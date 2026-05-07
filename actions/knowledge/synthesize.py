"""L3 framework synthesizer.

For each domain with 15+ active L1 records that has no recent L3,
distill a high-level knowledge framework using DeepSeek.
Writes to Qdrant (level=3) + Obsidian L3/{domain}/.
"""
import hashlib
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc
from storage.obsidian_writer import get_category_for_domain
from utils.llm import deepseek_generate, parse_json_block
from utils.embed import get_embedding

LOG = logging.getLogger("atlas.synthesize")

VAULT           = Path("/Volumes/data/obsidian-vault")
MIN_L1_FOR_L3   = 15   # minimum L1 records to justify L3
MAX_PER_CYCLE   = 1    # L3 is expensive — one per cycle
SAMPLE_SIZE     = 10   # how many L1 records to feed to DeepSeek


def _stable_id(seed: str) -> int:
    h = hashlib.md5(seed.encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 53)


def _domains_needing_l3() -> list[tuple[str, int]]:
    """Return (domain, l1_count) for domains ready for L3 synthesis."""
    l1_pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 1}},
            {"key": "status", "match": {"value": "active"}},
        ],
        "must_not": [
            {"key": "record_type", "match": {"value": "entity"}},
            {"key": "record_type", "match": {"value": "relation"}},
        ],
    }, limit=500)

    # Count L1 per domain
    counts: dict[str, int] = defaultdict(int)
    for pt in l1_pts:
        d = pt["payload"].get("domain") or "未分类"
        counts[d] += 1

    # Count existing L3 per domain + track most recent synthesis time
    l3_pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 3}},
            {"key": "status", "match": {"value": "active"}},
        ]
    }, limit=500)
    l3_counts: dict[str, int] = defaultdict(int)
    l3_latest: dict[str, str] = {}  # domain → latest created_at ISO string
    for pt in l3_pts:
        d = pt["payload"].get("domain") or "未分类"
        l3_counts[d] += 1
        ts = pt["payload"].get("created_at") or ""
        if ts > l3_latest.get(d, ""):
            l3_latest[d] = ts

    now_iso = datetime.now(timezone.utc).isoformat()

    # Qualify: enough L1, not too many L3 relative to L1, and 7-day cooldown per domain
    result = []
    for domain, l1_count in counts.items():
        if domain in ("未分类", "META"):
            continue
        if l1_count < MIN_L1_FOR_L3:
            continue
        existing_l3 = l3_counts.get(domain, 0)

        # Hard cap: max 1 L3 per 50 L1 records (so 418 L1 → max 8 L3)
        max_l3_for_domain = max(1, l1_count // 50)
        if existing_l3 >= max_l3_for_domain:
            continue

        # 7-day cooldown: don't re-synthesize a domain if already done recently
        last_ts = l3_latest.get(domain, "")
        if last_ts:
            try:
                from datetime import timedelta
                last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                if (datetime.now(timezone.utc) - last_dt) < timedelta(days=7):
                    continue
            except Exception:
                pass

        # Synthesize if: no L3, or L1 grew a lot since last synthesis (ratio > 6:1)
        if existing_l3 == 0 or (l1_count / max(existing_l3, 1)) > 6:
            result.append((domain, l1_count))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def _sample_l1_for_domain(domain: str, n: int = SAMPLE_SIZE) -> list[dict]:
    """Get top-quality L1 records for a domain."""
    pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 1}},
            {"key": "status", "match": {"value": "active"}},
            {"key": "domain", "match": {"value": domain}},
        ],
        "must_not": [
            {"key": "record_type", "match": {"value": "entity"}},
        ],
    }, limit=100)

    # Sort by completeness_score descending, take top n
    pts.sort(key=lambda p: p["payload"].get("completeness_score", 0), reverse=True)
    return pts[:n]


def _synthesize_framework(domain: str, samples: list[dict]) -> dict | None:
    """Call DeepSeek to distill a framework from L1 samples."""
    excerpts = []
    for i, pt in enumerate(samples, 1):
        pay  = pt["payload"]
        title = pay.get("title") or pay.get("topic") or ""
        summary = pay.get("summary") or ""
        rule = pay.get("rule_statement") or ""
        text = f"{title}: {summary or rule}"
        excerpts.append(f"{i}. {text[:200]}")

    excerpts_str = "\n".join(excerpts)

    from actions.evolution.prompt_optimizer import get_system_prompt
    system = get_system_prompt("l3_synthesis") or "你是知识提炼专家。从多条具体知识中提炼高阶框架和规律。严格输出JSON，不加解释。"
    user = f"""领域：{domain}
以下是该领域的 {len(samples)} 条核心知识点：

{excerpts_str}

请提炼这个领域的高阶知识框架，输出（严格JSON）：
{{
  "framework_title": "框架名称（10字以内）",
  "core_principle": "该领域最核心的原则（2-3句，高度概括）",
  "mental_model": "分析/解决该领域问题的思维模型（3-5句）",
  "key_patterns": ["规律1", "规律2", "规律3", "规律4"],
  "decision_framework": "在该领域做决策的框架（分步骤）",
  "common_mistakes": ["常见误区1", "常见误区2", "常见误区3"],
  "meta_insight": "该领域知识的底层逻辑（1-2句，最深的洞见）",
  "applicable_to": ["其他可迁移的领域1", "领域2"],
  "confidence": 0.85
}}"""

    try:
        raw = deepseek_generate(system, user, max_tokens=1200)
        result = parse_json_block(raw)
        if not result or not result.get("core_principle"):
            return None
        return result
    except Exception as e:
        LOG.error(f"Framework synthesis failed for {domain}: {e}")
        return None


def _write_l3_obsidian(domain: str, framework: dict, samples: list[dict]) -> str | None:
    category  = get_category_for_domain(domain)
    l3_dir    = VAULT / "L3" / category
    l3_dir.mkdir(parents=True, exist_ok=True)

    title = framework.get("framework_title") or f"{domain}框架"
    slug  = title.replace("/", "-")
    for ch in r'\:*?"<>|':
        slug = slug.replace(ch, "-")
    slug = slug[:50].strip()

    now       = datetime.now(timezone.utc)
    patterns  = "\n".join(f"- {p}" for p in (framework.get("key_patterns") or []))
    mistakes  = "\n".join(f"- {m}" for m in (framework.get("common_mistakes") or []))
    transfer  = "、".join(framework.get("applicable_to") or [])
    sample_links = "\n".join(
        f"- {pt['payload'].get('title') or pt['payload'].get('topic', '')}"
        for pt in samples[:5]
    )

    content = f"""---
level: L3
domain: {domain}
framework_title: {title}
confidence: {framework.get('confidence', 0.8)}
source_count: {len(samples)}
created: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}
generated_by: atlas-agent-v1
---

# {title}

> 领域框架 · 基于 {len(samples)} 条知识点蒸馏 · 置信度 {int(framework.get('confidence', 0.8) * 100)}%

## 核心原则

{framework.get('core_principle', '')}

## 思维模型

{framework.get('mental_model', '')}

## 关键规律

{patterns}

## 决策框架

{framework.get('decision_framework', '')}

## 常见误区

{mistakes}

## 底层洞见

{framework.get('meta_insight', '')}

## 可迁移领域

{transfer}

## 蒸馏来源（样本）

{sample_links}

---
*由 ATLAS 知识代理自动蒸馏 · {now.strftime("%Y-%m-%d %H:%M")} UTC*
"""

    md_path  = l3_dir / f"{slug}.md"
    rel_path = f"L3/{category}/{slug}.md"
    md_path.write_text(content, encoding="utf-8")
    return rel_path


def run_l3_synthesize(max_per_cycle: int = MAX_PER_CYCLE) -> dict:
    LOG.info(f"L3 synthesis: max {max_per_cycle} frameworks")

    domains = _domains_needing_l3()
    if not domains:
        LOG.info("No domains ready for L3 synthesis")
        return {"synthesized": 0}

    synthesized = 0

    for domain, l1_count in domains[:max_per_cycle]:
        LOG.info(f"Synthesizing L3 for: {domain} (L1 count={l1_count})")

        samples = _sample_l1_for_domain(domain)
        if len(samples) < 5:
            LOG.warning(f"Not enough quality L1 records for {domain}")
            continue

        framework = _synthesize_framework(domain, samples)
        if not framework:
            continue

        title = framework.get("framework_title") or f"{domain}框架"
        now   = datetime.now(timezone.utc).isoformat()

        embed_text = f"{title}\n{framework.get('core_principle', '')}\n{framework.get('meta_insight', '')}"
        vector = get_embedding(embed_text)
        if not vector:
            LOG.warning(f"Failed to embed L3 framework for {domain}")
            continue

        payload = {
            "level":             3,
            "status":            "active",
            "title":             title,
            "topic":             title,
            "domain":            domain,
            "content_type":      "framework",
            "core_principle":    framework.get("core_principle", ""),
            "mental_model":      framework.get("mental_model", ""),
            "key_patterns":      framework.get("key_patterns", []),
            "decision_framework": framework.get("decision_framework", ""),
            "common_mistakes":   framework.get("common_mistakes", []),
            "meta_insight":      framework.get("meta_insight", ""),
            "applicable_to":     framework.get("applicable_to", []),
            "confidence":        framework.get("confidence", 0.8),
            "source_count":      len(samples),
            "source":            "atlas-agent-synthesis",
            "created_at":        now,
            "generated_by":      "atlas-agent-v1",
        }

        point_id = _stable_id(f"l3-{domain}-{title}")
        try:
            qc.upsert_point(point_id, vector, payload)
        except Exception as e:
            LOG.error(f"Failed to upsert L3 {title}: {e}")
            continue

        try:
            rel_path = _write_l3_obsidian(domain, framework, samples)
            if rel_path:
                qc.patch_payload(point_id, {"obsidian_path": rel_path})
        except Exception as e:
            LOG.warning(f"L3 Obsidian write failed: {e}")

        synthesized += 1
        LOG.info(f"L3 synthesized: '{title}' for {domain}")
        time.sleep(0.3)

    LOG.info(f"L3 synthesis done: {synthesized} frameworks")
    return {"synthesized": synthesized}
