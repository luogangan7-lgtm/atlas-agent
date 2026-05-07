"""L4 → L1 feedback loop.

Meta-principles (L4) enrich the interpretation of specific knowledge atoms (L1).
For each L4 record, find related L1 records via vector search, then use DeepSeek
to add cross-level insight — showing how the high-level pattern manifests in
the specific knowledge node.
"""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc
from storage import obsidian_writer as ow
from utils.llm import deepseek_generate, parse_json_block
from utils.embed import get_embedding

LOG = logging.getLogger("atlas.l4_feedback")

VAULT          = Path("/Volumes/data/obsidian-vault")
MAX_L4_PER_RUN = 3    # L4 records to process per evolution cycle
MAX_L1_PER_L4  = 5    # L1 records to enrich per L4


def _get_l4_records() -> list[dict]:
    return qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 4}},
            {"key": "status", "match": {"value": "active"}},
        ]
    }, limit=20)


def _find_related_l1(l4_content: str, limit: int = MAX_L1_PER_L4) -> list[dict]:
    """Find L1 records most related to an L4 principle via vector search."""
    vector = get_embedding(l4_content[:2000])
    if not vector:
        return []

    results = qc.search(
        vector=vector,
        filter_body={
            "must": [
                {"key": "level",  "match": {"value": 1}},
                {"key": "status", "match": {"value": "active"}},
            ],
            "must_not": [
                {"key": "record_type", "match": {"value": "entity"}},
                {"key": "record_type", "match": {"value": "relation"}},
            ],
        },
        limit=limit,
        score_threshold=0.55,
    )
    return results


def _already_enriched(l1_id: int | str, l4_id: int | str) -> bool:
    """Check if this L1 already has a reference to this L4."""
    pts = qc.scroll({
        "must": [{"key": "id", "match": {"value": l1_id}}]
    }, limit=1)
    if not pts:
        return False
    l4_refs = pts[0]["payload"].get("l4_refs", [])
    return str(l4_id) in [str(r) for r in l4_refs]


def _generate_cross_level_insight(l4: dict, l1: dict) -> dict | None:
    """Ask DeepSeek: how does this L4 meta-pattern manifest in this L1 knowledge?"""
    l4_pay = l4.get("payload", l4)  # handle both search result and scroll result
    l1_pay = l1.get("payload", l1)

    l4_title   = l4_pay.get("title") or l4_pay.get("topic") or ""
    l4_content = l4_pay.get("content") or l4_pay.get("core_principle") or ""
    l1_title   = l1_pay.get("title") or l1_pay.get("topic") or ""
    l1_domain  = l1_pay.get("domain") or ""
    l1_summary = l1_pay.get("summary") or ""

    system = "你是跨层次知识整合专家。分析高阶元规律如何在具体知识中体现。严格输出JSON，不加解释。"
    user = f"""元规律（L4）：{l4_title}
{l4_content[:500]}

具体知识（L1 · {l1_domain}）：{l1_title}
{l1_summary[:400]}

请分析这条元规律如何在该具体知识中体现，输出（JSON）：
{{
  "manifestation": "元规律在此知识中的具体体现（2-3句）",
  "enriched_insight": "基于元规律视角，对该知识的更深理解（1-2句）",
  "application_hint": "结合元规律和具体知识，给出一个实用建议（1句）",
  "relevance_score": 0.75
}}"""

    try:
        raw    = deepseek_generate(system, user, max_tokens=400)
        result = parse_json_block(raw)
        if not result or not result.get("manifestation"):
            return None
        return result
    except Exception as e:
        LOG.error(f"Cross-level insight failed: {e}")
        return None


def _update_l1_with_l4_insight(l1_id: int, l4_id: int, l4_title: str, insight: dict,
                                 l1_payload: dict) -> None:
    """Patch L1 with meta_insight cross-reference and update Obsidian file."""
    l4_refs = l1_payload.get("l4_refs") or []
    if str(l4_id) not in [str(r) for r in l4_refs]:
        l4_refs.append(str(l4_id))

    meta_insights = l1_payload.get("meta_insights") or []
    meta_insights.append({
        "l4_id":       str(l4_id),
        "l4_title":    l4_title,
        "manifestation":   insight.get("manifestation", ""),
        "enriched_insight": insight.get("enriched_insight", ""),
        "application_hint": insight.get("application_hint", ""),
    })

    patch = {
        "l4_refs":         l4_refs,
        "meta_insights":   meta_insights[-3:],  # keep last 3 to avoid bloat
        "last_enriched_at": datetime.now(timezone.utc).isoformat(),
    }
    qc.patch_payload(l1_id, patch)

    # Append meta-insight section to Obsidian file if it exists
    obsidian_path = l1_payload.get("obsidian_path")
    if obsidian_path:
        full_path = VAULT / obsidian_path
        if full_path.exists():
            try:
                existing = full_path.read_text("utf-8")
                # Only add if section doesn't already exist
                if "## 元规律视角" not in existing:
                    section = "\n## 元规律视角\n\n"
                else:
                    # Remove old section to re-add fresh
                    idx = existing.index("## 元规律视角")
                    existing = existing[:idx]
                    section = "## 元规律视角\n\n"

                lines = []
                for mi in meta_insights[-3:]:
                    lines.append(f"### {mi['l4_title']}")
                    lines.append(f"**体现**：{mi['manifestation']}")
                    lines.append(f"**深化**：{mi['enriched_insight']}")
                    lines.append(f"**应用提示**：{mi['application_hint']}")
                    lines.append("")

                new_content = existing + section + "\n".join(lines)
                full_path.write_text(new_content, "utf-8")
            except Exception as e:
                LOG.warning(f"Failed to update Obsidian for L1 {l1_id}: {e}")


def run_l4_feedback(max_l4: int = MAX_L4_PER_RUN) -> dict:
    LOG.info(f"L4→L1 feedback: processing up to {max_l4} L4 records")

    l4_records = _get_l4_records()
    if not l4_records:
        LOG.info("No L4 records found")
        return {"l4_processed": 0, "l1_enriched": 0}

    l4_processed = 0
    l1_enriched  = 0

    for l4 in l4_records[:max_l4]:
        l4_id      = l4["id"]
        l4_pay     = l4["payload"]
        l4_title   = l4_pay.get("title") or l4_pay.get("topic") or ""
        l4_content = l4_pay.get("content") or l4_pay.get("core_principle") or ""

        if not l4_content.strip():
            LOG.debug(f"L4 {l4_id} has no content, skipping")
            continue

        LOG.info(f"Processing L4: '{l4_title}'")

        # Find related L1 records
        related_l1 = _find_related_l1(f"{l4_title}\n{l4_content}")
        if not related_l1:
            LOG.debug(f"No related L1 found for L4 '{l4_title}'")
            continue

        l4_processed += 1

        for l1_result in related_l1:
            l1_id  = l1_result["id"]
            l1_pay = l1_result["payload"]

            # Skip if already enriched by this L4
            l4_refs = l1_pay.get("l4_refs") or []
            if str(l4_id) in [str(r) for r in l4_refs]:
                continue

            insight = _generate_cross_level_insight(l4_pay, l1_pay)
            if not insight:
                continue

            if insight.get("relevance_score", 0) < 0.6:
                LOG.debug(f"Low relevance ({insight['relevance_score']:.2f}), skipping")
                continue

            _update_l1_with_l4_insight(l1_id, l4_id, l4_title, insight, l1_pay)
            l1_enriched += 1
            LOG.info(
                f"  L1 '{l1_pay.get('title', '')}' enriched with L4 '{l4_title}' "
                f"(relevance={insight.get('relevance_score', 0):.2f})"
            )
            time.sleep(0.3)

    LOG.info(f"L4→L1 feedback done: {l4_processed} L4 processed, {l1_enriched} L1 enriched")
    return {"l4_processed": l4_processed, "l1_enriched": l1_enriched}
