"""L1 completeness filler — bring low-score L1 records up to ≥0.85."""
import logging
import time
from pathlib import Path

from storage import qdrant_client as qc
from storage import obsidian_writer as ow
from utils.llm import deepseek_generate, parse_json_block

LOG = logging.getLogger("atlas.complete")

VAULT = Path("/Volumes/data/obsidian-vault")


def _build_fill_prompt(payload: dict, gaps: list[str]) -> tuple[str, str]:
    ct = payload.get("content_type") or "principle"
    topic   = payload.get("title") or payload.get("topic") or ""
    summary = payload.get("summary") or ""
    content = payload.get("content") or summary

    field_descriptions = {
        "summary":            "核心摘要（3-5句话，概括最重要内容）",
        "key_points":         "核心要点列表（3-5条，每条一句话）",
        "tags":               "标签列表（3-6个关键词）",
        "definition":         "概念核心定义（2-4句）",
        "scope":              "适用范围描述",
        "examples":           "具体例子列表（2-3条）",
        "related_concepts":   "关联概念列表",
        "claim":              "核心主张（一句话）",
        "reasoning":          "推理链条（逻辑推导过程）",
        "evidence":           "支撑证据列表",
        "limitations":        "局限性说明",
        "steps":              "操作步骤列表",
        "preconditions":      "前提条件",
        "expected_outcome":   "预期结果",
        "edge_cases":         "边界情况说明",
        "statement":          "事实陈述",
        "source_context":     "来源背景",
        "temporal_scope":     "时效性说明",
        "confidence":         "可信度说明",
        "rule_statement":     "原则/核心观点表述（1-2句）",
        "rationale":          "底层逻辑/关键洞见",
        "applicable_scenarios": "适用场景描述",
        "exceptions":         "例外情况说明",
        "hook":               "钩子/开场内容",
        "structure":          "内容结构说明",
        "pain_points":        "痛点列表",
        "cta":                "行动呼吁(CTA)",
        "tools_required":     "所需工具列表",
    }

    fields_str = "\n".join(
        f'  "{g}": "{field_descriptions.get(g, g)}"'
        for g in gaps
    )

    from actions.evolution.prompt_optimizer import get_system_prompt
    system = get_system_prompt("l1_completion") or "你是知识补全专家。根据现有知识内容，补全缺失字段。严格输出JSON，不加解释。"
    user = f"""知识主题：{topic}
内容类型：{ct}
现有摘要：{summary}
现有内容：{content[:2000]}

请补全以下缺失字段（严格JSON）：
{{
{fields_str}
}}

注意：列表类型字段输出列表，文本类型输出字符串。根据现有内容推断，不要捏造事实。"""
    return system, user


def _calc_completeness(node: dict, content_type: str) -> tuple[float, list[str]]:
    univ = (
        (0.15 if (node.get("summary") or "").strip() else 0) +
        (0.20 if node.get("key_points") else 0) +
        (0.05 if node.get("tags") else 0)
    )
    type_checks = {
        "concept":      [("definition", 0.30), ("scope", 0.20), ("examples", 0.25), ("related_concepts", 0.25)],
        "argument":     [("claim", 0.20), ("reasoning", 0.30), ("evidence", 0.30), ("limitations", 0.20)],
        "procedure":    [("steps", 0.35), ("preconditions", 0.20), ("expected_outcome", 0.25), ("edge_cases", 0.20)],
        "fact":         [("statement", 0.35), ("source_context", 0.25), ("temporal_scope", 0.20), ("confidence", 0.20)],
        "principle":    [("rule_statement", 0.25), ("rationale", 0.20), ("applicable_scenarios", 0.25), ("exceptions", 0.15), ("examples", 0.15)],
        "course":       [("rule_statement", 0.25), ("applicable_scenarios", 0.25), ("steps", 0.20), ("exceptions", 0.15), ("examples", 0.15)],
        "book":         [("rule_statement", 0.30), ("rationale", 0.20), ("applicable_scenarios", 0.25), ("examples", 0.25)],
        "note":         [("rule_statement", 0.30), ("rationale", 0.25), ("applicable_scenarios", 0.25), ("exceptions", 0.20)],
        "article":      [("claim", 0.25), ("reasoning", 0.30), ("evidence", 0.25), ("limitations", 0.20)],
        "video_script": [("hook", 0.30), ("structure", 0.25), ("pain_points", 0.25), ("cta", 0.20)],
        "sop":          [("steps", 0.40), ("preconditions", 0.20), ("expected_outcome", 0.25), ("tools_required", 0.15)],
    }
    checks = type_checks.get(content_type, type_checks["principle"])
    gaps = []
    type_score = 0.0
    for field_name, weight in checks:
        v = node.get(field_name)
        ok = bool(v) if not isinstance(v, list) else len(v) > 0
        if ok:
            type_score += weight
        else:
            gaps.append(field_name)
    if not (node.get("summary") or "").strip():
        gaps.append("summary")
    if not node.get("key_points"):
        gaps.append("key_points")
    score = min(univ + type_score * 0.60, 1.0)
    return score, gaps


def run_l1_complete(max_per_cycle: int = 12) -> dict:
    LOG.info(f"L1 completion: max {max_per_cycle} records")

    pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 1}},
            {"key": "status", "match": {"value": "active"}},
        ],
        "must_not": [
            {"key": "record_type", "match": {"value": "entity"}},
            {"key": "record_type", "match": {"value": "relation"}},
        ],
        "should": [
            {"key": "completeness_score", "range": {"lt": 0.85}},
        ],
    }, limit=max_per_cycle * 3)  # fetch extra, filter by actual score

    # Filter: must have content to work with
    candidates = []
    for pt in pts:
        pay = pt["payload"]
        score = pay.get("completeness_score") or 0
        if score >= 0.85:
            continue
        content = pay.get("content") or pay.get("summary") or ""
        if not content.strip():
            continue
        candidates.append(pt)
        if len(candidates) >= max_per_cycle:
            break

    if not candidates:
        LOG.info("No L1 records need completion")
        return {"completed": 0}

    LOG.info(f"Completing {len(candidates)} L1 records")
    completed = 0

    for pt in candidates:
        pay = pt["payload"]
        pid = pt["id"]
        ct  = pay.get("content_type") or "principle"

        gaps = pay.get("completeness_gaps") or []
        if not gaps:
            _, gaps = _calc_completeness(pay, ct)

        if not gaps:
            continue

        system, user = _build_fill_prompt(pay, gaps)
        try:
            raw = deepseek_generate(system, user, max_tokens=2500)
            filled = parse_json_block(raw)
        except Exception as e:
            LOG.error(f"LLM call failed for {pid}: {e}")
            continue

        if not filled or not isinstance(filled, dict):
            continue

        # Merge filled fields into payload
        updated = dict(pay)
        for k, v in filled.items():
            if v:
                updated[k] = v

        new_score, new_gaps = _calc_completeness(updated, ct)
        updated["completeness_score"] = new_score
        updated["completeness_gaps"]  = new_gaps

        try:
            patch = {k: filled[k] for k in filled if filled[k]}
            patch["completeness_score"] = new_score
            patch["completeness_gaps"]  = new_gaps
            qc.patch_payload(pid, patch)
        except Exception as e:
            LOG.error(f"Qdrant patch failed for {pid}: {e}")
            continue

        # Sync Obsidian file
        topic  = updated.get("topic") or updated.get("title") or ""
        domain = updated.get("domain") or "未分类"
        if topic:
            try:
                rel_path = ow.write_l1_obsidian(domain, topic, updated)
                if rel_path and not updated.get("obsidian_path"):
                    qc.patch_payload(pid, {"obsidian_path": rel_path})
            except Exception as e:
                LOG.warning(f"Obsidian write failed for '{topic}': {e}")

        completed += 1
        LOG.info(f"L1 {pid} '{topic}' completeness: {pay.get('completeness_score', 0):.2f} → {new_score:.2f}")
        time.sleep(0.2)

    LOG.info(f"L1 completion done: {completed} records updated")
    return {"completed": completed}
