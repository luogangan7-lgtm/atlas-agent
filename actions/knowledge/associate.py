"""L2 cross-domain insight generator.

Algorithm:
1. Sample active L1 records from Qdrant (with vectors)
2. For each record, search for similar records in OTHER domains (0.55–0.88 similarity)
3. Call DeepSeek to generate a cross-domain insight
4. Write to Qdrant (level=2) + Obsidian (L2/{domainA}/ and L2/{domainB}/)
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc
from storage.obsidian_writer import get_category_for_domain
from utils.llm import deepseek_generate, parse_json_block
from utils.embed import get_embedding

LOG = logging.getLogger("atlas.associate")

VAULT      = Path("/Volumes/data/obsidian-vault")
MIN_SIM    = 0.55
MAX_SIM    = 0.88  # above this → probably same concept, not cross-domain insight
MAX_PER_CYCLE = 3  # new L2 records per cycle (LLM-heavy operation)


def _stable_id(seed: str) -> int:
    h = hashlib.md5(seed.encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 53)


def _already_exists(topic_a: str, topic_b: str) -> bool:
    """Check if L2 record for this pair already exists."""
    existing = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 2}},
            {"key": "status", "match": {"value": "active"}},
            {"key": "source_topic_a", "match": {"value": topic_a}},
            {"key": "source_topic_b", "match": {"value": topic_b}},
        ]
    }, limit=1)
    return len(existing) > 0


def _generate_insight(node_a: dict, node_b: dict) -> dict | None:
    """Call DeepSeek to generate cross-domain insight."""
    pay_a = node_a["payload"]
    pay_b = node_b["payload"]

    title_a   = pay_a.get("title") or pay_a.get("topic") or ""
    title_b   = pay_b.get("title") or pay_b.get("topic") or ""
    domain_a  = pay_a.get("domain") or "未知"
    domain_b  = pay_b.get("domain") or "未知"
    summary_a = pay_a.get("summary") or ""
    summary_b = pay_b.get("summary") or ""

    from actions.evolution.prompt_optimizer import get_system_prompt
    system = get_system_prompt("l2_generation") or "你是跨域知识连接专家。从两条不同领域的知识中发现深层关联。严格输出JSON，不加解释。"
    user = f"""知识A（{domain_a}）：{title_a}
{summary_a[:600]}

知识B（{domain_b}）：{title_b}
{summary_b[:600]}

请分析这两条知识的深层关联，输出（严格JSON）：
{{
  "insight_topic": "洞见标题（10字以内，格式：主题A×主题B）",
  "relation_type": "analogy|complement|tension|cascade|meta_pattern",
  "insight_summary": "跨域洞见的核心发现（3-5句）",
  "decision_implication": "对决策的启示（2-3句，实用建议）",
  "bridge_concept": "连接两个域的核心桥接概念（一个词/短语）",
  "confidence": 0.75
}}

relation_type说明：
- analogy: A和B是同一模式的不同表现
- complement: A和B互补，合用更强
- tension: A和B存在张力/矛盾，需要情境选择
- cascade: A的结论自然引发B，或B解释了A的机制
- meta_pattern: A和B揭示了更高层次的规律"""

    try:
        raw = deepseek_generate(system, user, max_tokens=600)
        result = parse_json_block(raw)
        if not result or not result.get("insight_summary"):
            return None
        return result
    except Exception as e:
        LOG.error(f"Insight generation failed: {e}")
        return None


def _write_l2_obsidian(domain: str, topic: str, insight: dict, src_a: dict, src_b: dict) -> str | None:
    """Write L2 Obsidian file under L2/{category}/{slug}.md"""
    category = get_category_for_domain(domain)
    l2_dir   = VAULT / "L2" / category
    l2_dir.mkdir(parents=True, exist_ok=True)

    slug = topic.replace("/", "-").replace("×", "x")
    for ch in r'\:*?"<>|':
        slug = slug.replace(ch, "-")
    slug = slug[:60].strip()

    pay_a = src_a.get("payload", {})
    pay_b = src_b.get("payload", {})
    title_a  = pay_a.get("title") or pay_a.get("topic") or ""
    title_b  = pay_b.get("title") or pay_b.get("topic") or ""
    path_a   = pay_a.get("obsidian_path", "")
    path_b   = pay_b.get("obsidian_path", "")
    link_a   = f"[[{path_a}|{title_a}]]" if path_a else title_a
    link_b   = f"[[{path_b}|{title_b}]]" if path_b else title_b

    now = datetime.now(timezone.utc)
    confidence = insight.get("confidence", 0.7)
    rel_type   = insight.get("relation_type", "analogy")
    bridge     = insight.get("bridge_concept", "")

    content = f"""---
level: L2
domain: {domain}
topic: {topic}
relation_type: {rel_type}
bridge_concept: {bridge}
confidence: {confidence}
source_a: {pay_a.get('domain', '')} / {title_a}
source_b: {pay_b.get('domain', '')} / {title_b}
created: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}
generated_by: atlas-agent-v1
---

# {topic}

> 关联类型：{rel_type}  置信度：{int(confidence * 100)}%  桥接概念：{bridge}

## 跨域洞见

{insight.get('insight_summary', '')}

## 决策启示

{insight.get('decision_implication', '')}

## 来源知识

- {link_a}（{pay_a.get('domain', '')}）
- {link_b}（{pay_b.get('domain', '')}）

---
*由 ATLAS 知识代理自动生成*
"""

    md_path  = l2_dir / f"{slug}.md"
    rel_path = f"L2/{category}/{slug}.md"
    md_path.write_text(content, encoding="utf-8")
    return rel_path


def run_l2_generate(max_per_cycle: int = MAX_PER_CYCLE) -> dict:
    LOG.info(f"L2 generation: max {max_per_cycle} new insights")

    # Sample L1 records to use as seeds — fetch with vectors
    seeds = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 1}},
            {"key": "status", "match": {"value": "active"}},
        ],
        "must_not": [
            {"key": "record_type", "match": {"value": "entity"}},
            {"key": "record_type", "match": {"value": "relation"}},
        ],
    }, limit=50, with_vector=True)

    # Only keep records with vectors and non-empty summaries
    seeds = [s for s in seeds if s.get("vector") and (s["payload"].get("summary") or s["payload"].get("title"))]

    if len(seeds) < 2:
        LOG.info("Not enough L1 records with vectors to generate L2")
        return {"generated": 0}

    generated = 0
    tried_pairs: set[tuple] = set()

    import random
    random.shuffle(seeds)

    for seed in seeds:
        if generated >= max_per_cycle:
            break

        domain_a = seed["payload"].get("domain") or "未分类"
        title_a  = seed["payload"].get("title") or seed["payload"].get("topic") or ""

        # Search for similar records in other domains
        candidates = qc.search(
            vector=seed["vector"],
            filter_body={
                "must": [
                    {"key": "level",  "match": {"value": 1}},
                    {"key": "status", "match": {"value": "active"}},
                ],
                "must_not": [
                    {"key": "domain", "match": {"value": domain_a}},
                    {"key": "record_type", "match": {"value": "entity"}},
                    {"key": "record_type", "match": {"value": "relation"}},
                ],
            },
            limit=5,
            score_threshold=MIN_SIM,
        )

        # Filter to similarity range
        candidates = [c for c in candidates if MIN_SIM <= c["score"] <= MAX_SIM]
        if not candidates:
            continue

        for candidate in candidates:
            if generated >= max_per_cycle:
                break

            domain_b = candidate["payload"].get("domain") or "未分类"
            title_b  = candidate["payload"].get("title") or candidate["payload"].get("topic") or ""

            pair_key = tuple(sorted([title_a, title_b]))
            if pair_key in tried_pairs:
                continue
            tried_pairs.add(pair_key)

            if _already_exists(title_a, title_b) or _already_exists(title_b, title_a):
                LOG.debug(f"L2 pair already exists: {title_a} × {title_b}")
                continue

            LOG.info(f"Generating L2: {domain_a}/{title_a} × {domain_b}/{title_b} (sim={candidate['score']:.3f})")

            insight = _generate_insight(seed, candidate)
            if not insight:
                continue

            insight_topic = insight.get("insight_topic") or f"{title_a[:8]}×{title_b[:8]}"
            now           = datetime.now(timezone.utc).isoformat()
            confidence    = insight.get("confidence", 0.7)

            # Build embed text
            embed_text = f"{insight_topic}\n{insight.get('insight_summary', '')}"
            vector = get_embedding(embed_text)
            if not vector:
                LOG.warning("Failed to embed L2 insight")
                continue

            payload = {
                "level":              2,
                "status":             "active",
                "title":              insight_topic,
                "topic":              insight_topic,
                "domain":             domain_a,
                "domain_b":           domain_b,
                "relation_type":      insight.get("relation_type", "analogy"),
                "bridge_concept":     insight.get("bridge_concept", ""),
                "insight_summary":    insight.get("insight_summary", ""),
                "decision_implication": insight.get("decision_implication", ""),
                "confidence":         confidence,
                "source_topic_a":     title_a,
                "source_topic_b":     title_b,
                "source_id_a":        str(seed["id"]),
                "source_id_b":        str(candidate["id"]),
                "source_domain_a":    domain_a,
                "source_domain_b":    domain_b,
                "created_at":         now,
                "generated_by":       "atlas-agent-v1",
            }

            point_id = _stable_id(f"l2-{title_a}-{title_b}")
            try:
                qc.upsert_point(point_id, vector, payload)
            except Exception as e:
                LOG.error(f"Failed to upsert L2 point: {e}")
                continue

            # Write Obsidian — one file in domain_a's L2 folder
            try:
                rel_a = _write_l2_obsidian(domain_a, insight_topic, insight, seed, candidate)
                if rel_a:
                    qc.patch_payload(point_id, {"obsidian_path": rel_a})
            except Exception as e:
                LOG.warning(f"Obsidian write failed: {e}")

            # Mirror in domain_b's folder if different domain
            if domain_a != domain_b:
                try:
                    point_id_b = _stable_id(f"l2-{title_b}-{title_a}")
                    payload_b = dict(payload)
                    payload_b["domain"] = domain_b
                    rel_b = _write_l2_obsidian(domain_b, insight_topic, insight, candidate, seed)
                    payload_b["obsidian_path"] = rel_b
                    qc.upsert_point(point_id_b, vector, payload_b)
                except Exception as e:
                    LOG.warning(f"L2 mirror write failed: {e}")

            generated += 1
            LOG.info(f"L2 created: {insight_topic}")
            time.sleep(0.5)

    LOG.info(f"L2 generation done: {generated} new insights")
    return {"generated": generated}
