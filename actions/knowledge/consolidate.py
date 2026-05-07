"""L1 consolidation — merge near-duplicate / shallow records into deeper nodes.

Two jobs:
1. Group L1 records with identical or generic titles (核心要点总结 × 24 → 1 merged)
2. Find high-similarity L1 pairs (>0.90) in the same domain and merge them

Merged records: the source records are marked status=superseded, a single richer
record is upserted with merged content, higher completeness.
"""
import hashlib
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone

from storage import qdrant_client as qc
from storage import obsidian_writer as ow
from utils.llm import deepseek_generate, parse_json_block
from utils.embed import get_embedding

LOG = logging.getLogger("atlas.consolidate")

# Generic titles that should always be merged rather than kept as separate nodes
GENERIC_TITLES = {
    "核心要点总结", "核心要点", "避坑指南", "注意事项", "案例拆解",
    "使用场景", "适用场景", "注意事项与避坑", "要点总结", "总结",
    "概述", "关键点", "重点", "学习要点", "内容总结", "知识总结",
}

MAX_MERGE_PER_CYCLE = 3    # merges per cycle (LLM-heavy)
MIN_GROUP_SIZE      = 2    # min records in a group to trigger merge
MAX_GROUP_FEED      = 8    # max records to feed LLM for one merge


def _stable_id(seed: str) -> int:
    h = hashlib.md5(seed.encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 53)


def _find_generic_groups() -> list[tuple[str, str, list[dict]]]:
    """Return (domain, title, [records]) for generic-titled groups needing merge."""
    pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 1}},
            {"key": "status", "match": {"value": "active"}},
        ],
        "must_not": [
            {"key": "record_type", "match": {"value": "entity"}},
            {"key": "record_type", "match": {"value": "relation"}},
        ],
    }, limit=1000)

    # Group by (domain, title)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for pt in pts:
        pay   = pt["payload"]
        title = (pay.get("title") or pay.get("topic") or "").strip()
        domain = pay.get("domain") or "未分类"
        if title in GENERIC_TITLES:
            groups[(domain, title)].append(pt)

    result = []
    for (domain, title), records in groups.items():
        if len(records) >= MIN_GROUP_SIZE:
            result.append((domain, title, records))

    # Sort by group size descending
    result.sort(key=lambda x: len(x[2]), reverse=True)
    return result


def _merge_records(domain: str, title: str, records: list[dict]) -> dict | None:
    """Call DeepSeek to merge multiple shallow records into one deep node."""
    samples = records[:MAX_GROUP_FEED]
    excerpts = []
    for i, pt in enumerate(samples, 1):
        pay = pt["payload"]
        s = pay.get("summary") or pay.get("content") or ""
        kp = pay.get("key_points") or []
        rule = pay.get("rule_statement") or ""
        text = f"{s} {rule} {' '.join(kp[:3])}".strip()
        if text:
            excerpts.append(f"{i}. {text[:300]}")

    if not excerpts:
        return None

    excerpts_str = "\n".join(excerpts)
    ct = samples[0]["payload"].get("content_type") or "principle"

    system = "你是知识整合专家。将多条重复/碎片化的知识合并成一条深度、完整的知识节点。严格输出JSON，不加解释。"
    user = f"""领域：{domain}，主题：「{title}」

以下是 {len(samples)} 条重复/相似的碎片化知识：

{excerpts_str}

请整合为一条深度知识节点（JSON）：
{{
  "title": "更精确的标题（8字以内，不要用通用名称）",
  "content_type": "{ct}",
  "summary": "综合摘要（4-6句，整合所有核心内容）",
  "key_points": ["要点1", "要点2", "要点3", "要点4", "要点5"],
  "rule_statement": "核心原则表述（2-3句）",
  "rationale": "底层逻辑（2-3句）",
  "applicable_scenarios": "适用场景（具体描述）",
  "examples": ["整合后的具体例子1", "例子2"],
  "exceptions": "例外情况",
  "tags": ["标签1", "标签2", "标签3"]
}}"""

    try:
        raw = deepseek_generate(system, user, max_tokens=2000)
        result = parse_json_block(raw)
        if not result or not result.get("summary"):
            return None
        return result
    except Exception as e:
        LOG.error(f"Merge LLM call failed: {e}")
        return None


def run_l1_consolidate(max_per_cycle: int = MAX_MERGE_PER_CYCLE) -> dict:
    LOG.info("L1 consolidation: scanning for generic-titled groups")

    groups = _find_generic_groups()
    if not groups:
        LOG.info("No groups to consolidate")
        return {"merged": 0, "superseded": 0}

    LOG.info(f"Found {len(groups)} groups to consolidate (processing up to {max_per_cycle})")
    merged   = 0
    superseded_total = 0

    for domain, title, records in groups[:max_per_cycle]:
        LOG.info(f"Consolidating {len(records)}× '{title}' in [{domain}]")

        merged_node = _merge_records(domain, title, records)
        if not merged_node:
            continue

        new_title = merged_node.get("title") or title
        now       = datetime.now(timezone.utc).isoformat()

        embed_text = f"{new_title}\n{merged_node.get('summary', '')}\n{merged_node.get('rule_statement', '')}"
        vector = get_embedding(embed_text)
        if not vector:
            LOG.warning(f"Embed failed for merged '{new_title}'")
            continue

        # Compute completeness for merged node
        ct = merged_node.get("content_type") or "principle"
        from actions.knowledge.complete import _calc_completeness
        score, gaps = _calc_completeness(merged_node, ct)

        payload = {
            "level":                1,
            "status":               "active",
            "title":                new_title,
            "topic":                new_title,
            "domain":               domain,
            "content_type":         ct,
            "summary":              merged_node.get("summary", ""),
            "key_points":           merged_node.get("key_points", []),
            "rule_statement":       merged_node.get("rule_statement", ""),
            "rationale":            merged_node.get("rationale", ""),
            "applicable_scenarios": merged_node.get("applicable_scenarios", ""),
            "examples":             merged_node.get("examples", []),
            "exceptions":           merged_node.get("exceptions", ""),
            "tags":                 merged_node.get("tags", []),
            "completeness_score":   score,
            "completeness_gaps":    gaps,
            "source":               "atlas-consolidation",
            "merged_from_count":    len(records),
            "merged_from_titles":   [r["payload"].get("title", "") for r in records[:10]],
            "created_at":           now,
        }

        point_id = _stable_id(f"consolidated-{domain}-{new_title}-{now[:10]}")
        try:
            qc.upsert_point(point_id, vector, payload)
        except Exception as e:
            LOG.error(f"Failed to upsert consolidated node: {e}")
            continue

        # Write Obsidian file
        try:
            rel_path = ow.write_l1_obsidian(domain, new_title, payload)
            if rel_path:
                qc.patch_payload(point_id, {"obsidian_path": rel_path})
        except Exception as e:
            LOG.warning(f"Obsidian write failed for '{new_title}': {e}")

        # Supersede source records
        superseded = 0
        for rec in records:
            try:
                qc.patch_payload(rec["id"], {
                    "status":          "superseded",
                    "superseded_by":   str(point_id),
                    "superseded_at":   now,
                })
                superseded += 1
            except Exception as e:
                LOG.warning(f"Failed to supersede {rec['id']}: {e}")

        superseded_total += superseded
        merged += 1
        LOG.info(f"Consolidated '{new_title}': {superseded} records → 1 (score={score:.2f})")
        time.sleep(0.5)

    LOG.info(f"Consolidation done: {merged} merged, {superseded_total} superseded")
    return {"merged": merged, "superseded": superseded_total}
