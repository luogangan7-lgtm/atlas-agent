"""Autonomous research — fill knowledge gaps.

When a domain has < 5 L1 records, the agent autonomously generates research
questions and produces knowledge using DeepSeek, stored as L0 pending_review.
"""
import hashlib
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc
from storage import state_manager as sm
from utils.llm import deepseek_generate, parse_json_block

LOG = logging.getLogger("atlas.research")

VAULT     = Path("/Volumes/data/obsidian-vault")
INBOX_DIR = VAULT / "agent-brain" / "inbox"
GAP_THRESHOLD = 5   # domains with fewer than this many L1 records get researched
MAX_RESEARCH_PER_CYCLE = 1  # one domain per cycle (expensive)


def _stable_id(seed: str) -> int:
    h = hashlib.md5(seed.encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 53)


def _find_gap_domains() -> list[tuple[str, int]]:
    """Return list of (domain, l1_count) for domains with < GAP_THRESHOLD L1.
    Excludes domains researched in the last 3 days (avoids infinite re-research
    while L0 nodes are still waiting to be organized into L1).
    """
    from datetime import datetime, timezone, timedelta

    pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 1}},
            {"key": "status", "match": {"value": "active"}},
        ],
        "must_not": [
            {"key": "record_type", "match": {"value": "entity"}},
            {"key": "record_type", "match": {"value": "relation"}},
        ],
    }, limit=500)

    counts: dict[str, int] = {}
    for pt in pts:
        d = pt["payload"].get("domain") or "未分类"
        counts[d] = counts.get(d, 0) + 1

    # Build recent_research set: domains researched in the last 3 days.
    # Check BOTH active_research_threads (primary) AND L0 pending_review nodes (fallback).
    # The L0 check alone was insufficient: once nodes are confirmed their status changes
    # from pending_review → pending → active, so the cooldown stopped blocking them.
    recent_research: set[str] = set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=3)

    # Primary: check context.json active_research_threads (survives node status changes)
    try:
        from storage import state_manager as sm
        ctx = sm.load_context()
        for thread in ctx.get("active_research_threads", []):
            domain = thread.get("domain", "")
            started = thread.get("started", "")
            if not domain or not started:
                continue
            try:
                dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                if dt > cutoff:
                    recent_research.add(domain)
            except Exception:
                recent_research.add(domain)
    except Exception:
        pass

    # Fallback: also check L0 pending_review nodes (catches in-flight research)
    l0_pending = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 0}},
            {"key": "status", "match": {"value": "pending_review"}},
            {"key": "source", "match": {"value": "atlas-agent-research"}},
        ]
    }, limit=200)
    for pt in l0_pending:
        ts = pt["payload"].get("created_at") or ""
        domain = pt["payload"].get("domain") or ""
        if not domain:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt > cutoff:
                recent_research.add(domain)
        except Exception:
            recent_research.add(domain)

    # Filter: small domains (but not "未分类" which is catch-all, and not recently researched)
    gaps = [
        (d, c) for d, c in counts.items()
        if c < GAP_THRESHOLD and d != "未分类" and d not in recent_research
    ]
    gaps.sort(key=lambda x: x[1])  # smallest first
    return gaps


def _web_context_for_domain(domain: str) -> str:
    """Fetch real web search results to ground LLM research in factual data."""
    try:
        from utils.web_search import text_search
        results = text_search(f"{domain} 核心知识 基础概念", max_results=4)
        if not results:
            results = text_search(domain, max_results=4)
        if not results:
            return ""
        lines = []
        for r in results[:4]:
            title   = r.get("title", "")
            snippet = r.get("snippet", "")[:200]
            url     = r.get("url", "")
            lines.append(f"- [{title}]({url}): {snippet}")
        return "\n".join(lines)
    except Exception as e:
        LOG.debug(f"web context fetch failed for {domain}: {e}")
        return ""


def _research_domain(domain: str, existing_count: int) -> dict | None:
    """Use web search + DeepSeek to generate grounded knowledge for a gap domain."""
    web_context = _web_context_for_domain(domain)
    web_section = (
        f"\n\n## 网络参考资料（基于此生成，确保事实准确）\n{web_context}"
        if web_context else ""
    )

    system = "你是知识研究专家。为知识库中知识不足的领域生成核心知识节点。严格输出JSON，不加解释。"
    user = f"""知识库中「{domain}」领域只有 {existing_count} 条记录，知识严重不足。{web_section}

请为该领域生成 3 个最重要的基础知识节点（JSON数组），每个节点包含：
{{
  "title": "知识点标题（8字以内）",
  "content_type": "principle/concept/fact/procedure",
  "summary": "核心摘要（3-5句话，客观准确）",
  "key_points": ["要点1", "要点2", "要点3"],
  "rule_statement": "核心原则/定义（1-2句）",
  "rationale": "背后的逻辑/机制",
  "applicable_scenarios": "适用场景",
  "examples": ["例子1", "例子2"],
  "tags": ["标签1", "标签2", "标签3"]
}}

输出格式：
{{"nodes": [节点1, 节点2, 节点3]}}

重要：基于真实知识，不要捏造，可信度要高。"""

    try:
        raw = deepseek_generate(system, user, max_tokens=3500)
        result = parse_json_block(raw)
        if not result or not isinstance(result.get("nodes"), list):
            return None
        return result
    except Exception as e:
        LOG.error(f"Research failed for domain {domain}: {e}")
        return None


def _write_research_inbox(domain: str, nodes: list[dict], l0_ids: list[int]) -> str:
    """Inform user about autonomous research, let them confirm."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    now  = datetime.now(timezone.utc)
    date = now.strftime("%Y%m%d-%H%M%S")

    node_lines = "\n".join(
        f"### {i+1}. {n.get('title', '')}\n{n.get('summary', '')[:200]}"
        for i, n in enumerate(nodes)
    )
    ids_str = ", ".join(str(i) for i in l0_ids)

    content = f"""---
type: user_decision_required
category: research_confirm
priority: low
created: {now.isoformat()}
status: pending
domain: {domain}
l0_ids: [{ids_str}]
---

# 我主动研究了：{domain}

## 研究背景
你的知识库中「{domain}」领域知识不足（少于 {GAP_THRESHOLD} 条），
我主动进行了研究并生成了以下知识节点，请确认后我将正式加入知识库。

## 研究成果（待审核）

{node_lines}

## 请你决定（勾选后保存，下次循环我会处理）

- [ ] 全部确认加入知识库
- [ ] 部分确认（在下面标注哪些）：
- [ ] 全部拒绝，删除这批研究结果

---
*由 ATLAS 知识代理自动研究 · {now.strftime("%Y-%m-%d %H:%M")} UTC*
"""

    filename = f"research-{domain}-{date}.md"
    (INBOX_DIR / filename).write_text(content, encoding="utf-8")
    return f"agent-brain/inbox/{filename}"


def run_autonomous_research(max_domains: int = MAX_RESEARCH_PER_CYCLE) -> dict:
    LOG.info("Autonomous research: scanning for knowledge gaps")

    gaps = _find_gap_domains()
    if not gaps:
        LOG.info("No knowledge gaps found (all domains have ≥5 L1 records)")
        return {"researched": 0, "nodes_queued": 0}

    # Pick a random gap domain (not always the same smallest one)
    sample_size = min(max_domains * 3, len(gaps))
    targets = random.sample(gaps[:sample_size], min(max_domains, sample_size))

    researched = 0
    total_nodes = 0

    for domain, count in targets:
        LOG.info(f"Researching gap domain: {domain} (has {count} L1 records)")

        result = _research_domain(domain, count)
        if not result or not result.get("nodes"):
            continue

        nodes = result["nodes"]
        now = datetime.now(timezone.utc).isoformat()
        l0_ids = []

        # Store research as L0 pending_review
        for node in nodes:
            title = (node.get("title") or "").strip()
            if not title:
                continue

            # Build L0 record with pending_review status
            payload = {
                "level":          0,
                "status":         "pending_review",  # not yet 'pending' — needs user confirmation
                "title":          title,
                "content":        node.get("summary") or "",
                "domain":         domain,
                "content_type":   node.get("content_type") or "principle",
                "source":         "atlas-agent-research",
                "created_at":     now,
                # Pre-fill structured fields
                "summary":        node.get("summary") or "",
                "key_points":     node.get("key_points") or [],
                "rule_statement": node.get("rule_statement") or "",
                "rationale":      node.get("rationale") or "",
                "applicable_scenarios": node.get("applicable_scenarios") or "",
                "examples":       node.get("examples") or [],
                "tags":           node.get("tags") or [],
            }

            point_id = _stable_id(f"research-{domain}-{title}-{now}")
            try:
                from utils.embed import get_embedding
                embed_text = f"{title}\n{node.get('summary', '')}\n{node.get('rule_statement', '')}"
                vector = get_embedding(embed_text) or [0.0] * 1024
                qc.upsert_point(point_id, vector, payload)
                l0_ids.append(point_id)
                total_nodes += 1
            except Exception as e:
                LOG.error(f"Failed to store research node '{title}': {e}")

        if l0_ids:
            # Write to inbox for user confirmation
            inbox_path = _write_research_inbox(domain, nodes, l0_ids)
            LOG.info(f"Research inbox written: {inbox_path}")

            ctx = sm.load_context()
            threads = ctx.get("active_research_threads", [])
            threads.append({"domain": domain, "started": now, "l0_ids": l0_ids, "status": "pending_review"})
            ctx["active_research_threads"] = threads
            ctx["pending_user_decisions"] = ctx.get("pending_user_decisions", 0) + 1
            sm.save_context(ctx)

        researched += 1

    LOG.info(f"Autonomous research done: {researched} domains, {total_nodes} nodes queued")
    return {"researched": researched, "nodes_queued": total_nodes}
