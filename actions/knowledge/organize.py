"""L0 → L1 organizer: extract knowledge nodes from raw L0 records."""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone

from storage import qdrant_client as qc
from storage import obsidian_writer as ow
from storage.obsidian_writer import get_category_for_domain
from utils.llm import deepseek_generate, parse_json_block
from utils.embed import get_embedding

LOG = logging.getLogger("atlas.organize")

CONTENT_TYPES = [
    "concept", "argument", "procedure", "fact", "principle",
    "course", "book", "note", "article", "video_script", "sop",
]


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


def _stable_id(seed: str) -> str:
    """Generate deterministic UUID-like int from seed string."""
    h = hashlib.md5(seed.encode()).hexdigest()
    return int(h[:16], 16) % (2**53)  # Qdrant accepts unsigned int


def _extract_l1_content(content: str, domain: str, content_type: str) -> dict | None:
    type_hints = {
        "concept":   "definition（核心定义）, scope（适用范围）, examples（具体例子列表）, related_concepts（关联概念列表）",
        "argument":  "claim（核心主张）, reasoning（推理过程）, evidence（支撑证据列表）, limitations（局限性）",
        "procedure": "steps（步骤列表）, preconditions（前提条件）, expected_outcome（预期结果）, edge_cases（边界情况）",
        "fact":      "statement（事实陈述）, source_context（来源背景）, temporal_scope（时效性）, confidence（可信度）",
        "principle": "rule_statement（原则表述）, rationale（底层逻辑）, applicable_scenarios（适用场景）, exceptions（例外）, examples（例子）",
        "course":    "rule_statement（核心原理）, applicable_scenarios（适用场景）, steps（操作要点列表）, exceptions（注意事项）",
        "book":      "rule_statement（核心主张）, rationale（底层逻辑）, key_points（可执行要点）, applicable_scenarios（适用场景）",
        "note":      "rule_statement（核心观点）, rationale（关键洞见）, applicable_scenarios（适用场景）, exceptions（例外）",
        "article":   "claim（核心论点）, reasoning（推理链条）, evidence（支撑证据）, limitations（局限性）",
        "video_script": "hook（钩子/开场）, structure（内容结构）, pain_points（痛点列表）, cta（行动呼吁）",
        "sop":       "steps（操作步骤列表）, preconditions（前提条件）, expected_outcome（预期结果）, tools_required（工具列表）",
    }
    hint = type_hints.get(content_type, type_hints["principle"])

    infer_domain = not domain or domain == "未分类"
    domain_field = '"suggested_domain": "推断的知识领域（2-6汉字）",' if infer_domain else ""
    domain_hint  = "领域未知，请在 suggested_domain 字段给出推断" if infer_domain else f"领域：{domain}"

    system = "你是知识萃取专家。从原始内容中提取结构化知识节点。严格输出中文JSON，不要解释。"
    user = f"""{domain_hint}  内容类型：{content_type}

原始内容：
{content[:4000]}

输出格式（严格JSON）：
{{
  "nodes": [
    {{
      "title": "节点标题（8字以内）",
      "summary": "核心摘要（3-5句）",
      "key_points": ["要点1", "要点2", "要点3"],
      "content_type": "{content_type}",
      "tags": ["标签1", "标签2"],
      "faithfulness_score": 0.85,
      {domain_field}
      {hint}
    }}
  ]
}}"""

    try:
        raw = deepseek_generate(system, user, max_tokens=4000)
        result = parse_json_block(raw)
        if not result or not isinstance(result.get("nodes"), list):
            return None
        if not result["nodes"]:
            return None
        return result
    except Exception as e:
        LOG.error(f"extractL1Content failed: {e}")
        return None


def run_l0_organize(max_per_cycle: int = 10) -> dict:
    LOG.info(f"L0→L1 organizer: max {max_per_cycle} records")

    # Fetch pending or active L0 records (active = written by Node.js atlas-memory)
    pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 0}},
            {"key": "status", "match": {"any": ["pending", "active"]}},
        ]
    }, limit=max_per_cycle)

    if not pts:
        LOG.info("No pending L0 records")
        return {"processed": 0, "nodes_created": 0}

    LOG.info(f"Found {len(pts)} pending L0 records")
    processed = 0
    nodes_created = 0

    for pt in pts:
        pay = pt["payload"]
        pid = pt["id"]
        content = pay.get("content") or pay.get("raw_content") or ""
        if not content.strip():
            qc.patch_payload(pid, {"status": "processed"})
            continue

        # Quality gate: skip fragments too short to yield meaningful knowledge
        if len(content.strip()) < 300:
            LOG.info(f"L0 {pid} too short ({len(content.strip())} chars), skipping")
            qc.patch_payload(pid, {"status": "skipped_short", "skip_reason": f"content_len={len(content.strip())}"})
            processed += 1
            continue

        domain       = (pay.get("domain") or "").strip() or None
        content_type = pay.get("content_type") or "principle"
        source       = pay.get("source") or pay.get("title") or str(pid)

        result = _extract_l1_content(content, domain or "", content_type)
        if not result:
            LOG.warning(f"Failed to extract from L0 {pid}")
            qc.patch_payload(pid, {"status": "processed", "process_error": "extract_failed"})
            processed += 1
            continue

        now = datetime.now(timezone.utc).isoformat()

        for i, node in enumerate(result["nodes"]):
            # Use LLM-suggested domain when original domain was empty
            if not domain:
                suggested = (node.pop("suggested_domain", None) or "").strip()
                domain = suggested if suggested and suggested != "未分类" else "未分类"
            title = (node.get("title") or "").strip()
            if not title:
                continue

            score, gaps = _calc_completeness(node, content_type)

            node_payload = {
                "level":              1,
                "status":             "active",
                "title":              title,
                "topic":              title,
                "domain":             domain,
                "category":           get_category_for_domain(domain),
                "content_type":       content_type,
                "summary":            node.get("summary") or "",
                "key_points":         node.get("key_points") or [],
                "tags":               node.get("tags") or [],
                "faithfulness_score": node.get("faithfulness_score") or 0.85,
                "completeness_score": score,
                "completeness_gaps":  gaps,
                "source_l0":          source,
                "source":             source,
                "created_at":         now,
                "processed_by":       "atlas-agent-v1",
            }
            # Copy type-specific fields
            for field in ["definition", "scope", "examples", "related_concepts",
                          "claim", "reasoning", "evidence", "limitations",
                          "steps", "preconditions", "expected_outcome", "edge_cases",
                          "statement", "source_context", "temporal_scope", "confidence",
                          "rule_statement", "rationale", "applicable_scenarios", "exceptions",
                          "hook", "structure", "pain_points", "cta", "tools_required"]:
                if field in node:
                    node_payload[field] = node[field]

            # Generate embedding
            embed_text = f"{title}\n{node.get('summary', '')}"
            vector = get_embedding(embed_text)
            if not vector:
                LOG.warning(f"Failed to embed node '{title}'")
                continue

            node_id = _stable_id(title + str(pid) + str(i))

            try:
                qc.upsert_point(node_id, vector, node_payload)
            except Exception as e:
                LOG.error(f"Failed to upsert node {node_id}: {e}")
                continue

            # Write Obsidian file
            try:
                rel_path = ow.write_l1_obsidian(domain, title, node_payload)
                if rel_path:
                    qc.patch_payload(node_id, {"obsidian_path": rel_path})
            except Exception as e:
                LOG.warning(f"Obsidian write failed for '{title}': {e}")

            nodes_created += 1
            time.sleep(0.1)

        # Mark L0 as processed
        qc.patch_payload(pid, {"status": "processed"})
        processed += 1
        LOG.info(f"L0 {pid} → {len(result['nodes'])} nodes")

    LOG.info(f"L0→L1 done: {processed} L0 processed, {nodes_created} L1 nodes created")
    return {"processed": processed, "nodes_created": nodes_created}
