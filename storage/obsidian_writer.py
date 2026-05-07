"""Write L1 Obsidian markdown files — ported from index.js."""
import json
from pathlib import Path
from typing import Any

VAULT         = Path("/Volumes/data/obsidian-vault")
TAXONOMY_FILE = VAULT / "agent-brain" / "state" / "taxonomy.json"

_taxonomy_cache: dict = {}


def _load_taxonomy_cached() -> dict:
    """Load taxonomy, refresh if file is newer than cache."""
    global _taxonomy_cache
    try:
        mtime = TAXONOMY_FILE.stat().st_mtime
        if _taxonomy_cache.get("_mtime") != mtime:
            data = json.loads(TAXONOMY_FILE.read_text("utf-8"))
            data["_mtime"] = mtime
            _taxonomy_cache = data
    except Exception:
        pass
    return _taxonomy_cache


def get_category_for_domain(domain: str) -> str:
    """Return the canonical category (大类) for a domain, or '其他' as fallback."""
    if not domain or domain == "未分类":
        return "其他"
    primary = domain.split("×")[0].strip()  # handle cross-domain compound names
    tax = _load_taxonomy_cached()
    for cat_name, cat_data in tax.get("categories", {}).items():
        if primary in cat_data.get("domains", []) or domain in cat_data.get("domains", []):
            return cat_name
    return "其他"


def _safe_slug(s: str) -> str:
    for ch in r'/\:*?"<>|':
        s = s.replace(ch, "-")
    return s[:50].strip()


def _safe(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ""
    return str(v).strip()


def _list_items(lst: Any) -> list[str]:
    if not lst:
        return []
    result = []
    for item in (lst if isinstance(lst, list) else [lst]):
        if isinstance(item, str) and item.strip():
            result.append(f"- {item.strip()}")
        elif isinstance(item, dict):
            t = (item.get("text") or item.get("step") or item.get("example")
                 or item.get("fact") or item.get("name") or json.dumps(item, ensure_ascii=False))
            if t:
                result.append(f"- {t}")
        elif str(item).strip():
            result.append(f"- {item}")
    return result


def _numbered_items(lst: Any) -> list[str]:
    if not lst:
        return []
    result = []
    for i, item in enumerate(lst if isinstance(lst, list) else [lst], 1):
        if isinstance(item, str) and item.strip():
            result.append(f"{i}. {item.strip()}")
        elif isinstance(item, dict):
            t = item.get("step") or item.get("text") or json.dumps(item, ensure_ascii=False)
            if t:
                result.append(f"{i}. {t}")
        elif str(item).strip():
            result.append(f"{i}. {item}")
    return result


def _sec(header: str, body: str) -> list[str]:
    """Return [header, body, ''] only when body is non-empty."""
    if body.strip():
        return [header, body, ""]
    return []


def _sec_list(header: str, items: list[str]) -> list[str]:
    """Return [header, *items, ''] only when items is non-empty."""
    if items:
        return [header, *items, ""]
    return []


def _type_blocks(ct: str, p: dict) -> list[str]:
    blocks: list[str] = []

    if ct == "concept":
        blocks += _sec("## 定义", _safe(p.get("definition")))
        blocks += _sec("## 适用范围", _safe(p.get("scope")))
        blocks += _sec_list("## 具体例子", _list_items(p.get("examples")))
        blocks += _sec_list("## 关联概念", _list_items(p.get("related_concepts")))

    elif ct == "argument":
        blocks += _sec("## 核心论点", _safe(p.get("claim")))
        blocks += _sec("## 推理链条", _safe(p.get("reasoning")))
        blocks += _sec_list("## 支撑证据", _list_items(p.get("evidence")))
        blocks += _sec("## 局限性", _safe(p.get("limitations")))

    elif ct == "procedure":
        blocks += _sec_list("## 操作步骤", _numbered_items(p.get("steps")))
        blocks += _sec("## 前提条件", _safe(p.get("preconditions")))
        blocks += _sec("## 预期结果", _safe(p.get("expected_outcome")))
        blocks += _sec("## 边界情况", _safe(p.get("edge_cases")))

    elif ct == "fact":
        blocks += _sec("## 事实陈述", _safe(p.get("statement")))
        blocks += _sec("## 来源背景", _safe(p.get("source_context")))
        blocks += _sec("## 时效性", _safe(p.get("temporal_scope")))
        blocks += _sec("## 可信度", _safe(p.get("confidence")))

    elif ct in ("principle", "note"):
        h1 = "## 原则表述" if ct == "principle" else "## 核心观点"
        h2 = "## 底层逻辑" if ct == "principle" else "## 关键洞见"
        blocks += _sec(h1, _safe(p.get("rule_statement")))
        blocks += _sec(h2, _safe(p.get("rationale")))
        blocks += _sec("## 适用场景", _safe(p.get("applicable_scenarios")))
        blocks += _sec("## 例外情况", _safe(p.get("exceptions")))
        blocks += _sec_list("## 具体例子", _list_items(p.get("examples")))

    elif ct == "course":
        body = _safe(p.get("rule_statement") or p.get("rationale"))
        blocks += _sec("## 核心原理", body)
        blocks += _sec("## 适用场景", _safe(p.get("applicable_scenarios") or p.get("scope")))
        blocks += _sec_list("## 操作要点", _numbered_items(p.get("steps")))
        blocks += _sec("## 注意事项", _safe(p.get("exceptions") or p.get("edge_cases")))
        blocks += _sec_list("## 课程案例", _list_items(p.get("examples")))

    elif ct == "book":
        blocks += _sec("## 核心主张", _safe(p.get("rule_statement")))
        blocks += _sec("## 底层逻辑", _safe(p.get("rationale")))
        blocks += _sec_list("## 可执行要点", _list_items(p.get("key_points")))
        blocks += _sec("## 适用场景", _safe(p.get("applicable_scenarios")))
        blocks += _sec_list("## 书中案例", _list_items(p.get("examples")))

    elif ct == "article":
        blocks += _sec("## 核心论点", _safe(p.get("claim") or p.get("rule_statement")))
        blocks += _sec("## 推理链条", _safe(p.get("reasoning") or p.get("rationale")))
        blocks += _sec_list("## 支撑证据", _list_items(p.get("evidence")))
        blocks += _sec("## 局限性", _safe(p.get("limitations") or p.get("exceptions")))
        blocks += _sec_list("## 具体例子", _list_items(p.get("examples")))

    elif ct == "video_script":
        pp = p.get("pain_points")
        pain_lines = _list_items(pp) if isinstance(pp, list) else (
            [f"- {_safe(pp)}"] if _safe(pp) else []
        )
        ap = p.get("applicable_scenarios")
        ap_str = "；".join(ap) if isinstance(ap, list) else _safe(ap)
        blocks += _sec("## 钩子/开场", _safe(p.get("hook")))
        blocks += _sec("## 内容结构", _safe(p.get("structure")))
        blocks += _sec_list("## 痛点/价值点", pain_lines)
        blocks += _sec("## 行动呼吁(CTA)", _safe(p.get("cta")))
        blocks += _sec("## 适用场景", ap_str)

    elif ct == "sop":
        blocks += _sec_list("## 操作步骤", _numbered_items(p.get("steps")))
        blocks += _sec_list("## 所需工具", _list_items(p.get("tools_required")))
        blocks += _sec("## 前提条件", _safe(p.get("preconditions")))
        blocks += _sec("## 预期结果", _safe(p.get("expected_outcome")))

    else:  # fallback = principle
        blocks += _sec("## 原则表述", _safe(p.get("rule_statement")))
        blocks += _sec("## 底层逻辑", _safe(p.get("rationale")))
        blocks += _sec("## 适用场景", _safe(p.get("applicable_scenarios")))
        blocks += _sec("## 例外情况", _safe(p.get("exceptions")))
        blocks += _sec_list("## 具体例子", _list_items(p.get("examples")))

    return blocks


def build_l1_md(payload: dict) -> str:
    ct    = payload.get("content_type") or "principle"
    score = payload.get("completeness_score") or 0
    faith = payload.get("faithfulness_score") or 1.0
    flag  = " ⚠️幻觉风险" if faith < 0.6 else ""

    tags     = payload.get("tags") or []
    tag_str  = ", ".join(f'"{t}"' for t in tags)
    gaps     = payload.get("completeness_gaps") or []
    gaps_str = ", ".join(f'"{g}"' for g in gaps)

    source_l0 = payload.get("source_l0") or payload.get("source") or ""
    topic     = payload.get("topic") or payload.get("title") or ""
    domain    = payload.get("domain") or "未分类"

    lines: list[str] = [
        "---", "level: L1", f"content_type: {ct}", f"domain: {domain}",
        f"topic: {topic}", f"source_l0: {source_l0}",
        f"created: {payload.get('created_at') or ''}",
        f"tags: [{tag_str}]", f"completeness_score: {score}",
        f"completeness_gaps: [{gaps_str}]", f"faithfulness_score: {faith}", "---",
        "", f"# {payload.get('title') or topic}",
        f"> 类型：{ct}  完整度：{int(score * 100)}%{flag}",
        "",
    ]

    # Summary — only if non-empty
    summary = _safe(payload.get("summary"))
    if summary:
        lines += ["## 摘要", summary, ""]

    # Key points — only if non-empty
    kp = _list_items(payload.get("key_points"))
    if kp:
        lines += ["## 核心要点", *kp, ""]

    # Type-specific blocks (all guarded by _sec/_sec_list)
    lines += _type_blocks(ct, payload)

    # L4 meta-insight section (appended by l4_feedback, don't strip it)
    meta_section = payload.get("_meta_insight_section")
    if meta_section:
        lines += ["", meta_section]

    return "\n".join(lines).rstrip() + "\n"


def write_l1_obsidian(domain: str, topic: str, payload: dict, source_l0_path: str | None = None) -> str | None:
    """Write L1 Obsidian file under L1/{category}/{slug}.md"""
    category = get_category_for_domain(domain)
    l1_dir   = VAULT / "L1" / category
    l1_dir.mkdir(parents=True, exist_ok=True)

    slug     = _safe_slug(topic)
    md_path  = l1_dir / f"{slug}.md"
    rel_path = f"L1/{category}/{slug}.md"

    md = build_l1_md(payload)
    md_path.write_text(md, encoding="utf-8")
    return rel_path
