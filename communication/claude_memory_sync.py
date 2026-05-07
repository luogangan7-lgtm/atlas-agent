"""Sync ATLAS current state into Claude Code's auto-memory system.

Called every fast cycle. Writes atlas_live_state.md so Claude always
has fresh pyramid stats, pending items, and knowledge gaps at session start.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc
from storage import state_manager as sm

LOG = logging.getLogger("atlas.claude_sync")

MEMORY_DIR = Path("/Users/luolimo/.claude/projects/-Users-luolimo/memory")
STATE_FILE = MEMORY_DIR / "atlas_live_state.md"
MEMORY_MD  = MEMORY_DIR / "MEMORY.md"

MEMORY_ENTRY = "- [ATLAS Live State](atlas_live_state.md) — ATLAS知识金字塔当前状态、研究线程、待确认项，每15分钟自动更新"


def _count(filter_dict: dict) -> int:
    try:
        r = qc._req("POST", f"collections/{qc.COLLECTION}/points/count",
                    {"exact": False, "filter": filter_dict})
        return r.get("result", {}).get("count", 0)
    except Exception:
        return 0


def sync_to_claude_memory() -> None:
    try:
        ctx = sm.load_context()
    except Exception:
        ctx = {}

    l1_active  = _count({"must": [{"key": "level", "match": {"value": 1}},
                                   {"key": "status", "match": {"value": "active"}}]})
    l2_total   = _count({"must": [{"key": "level", "match": {"value": 2}}]})
    l3_total   = _count({"must": [{"key": "level", "match": {"value": 3}},
                                   {"key": "status", "match": {"value": "active"}}]})
    l4_total   = _count({"must": [{"key": "level", "match": {"value": 4}}]})
    pending_l0 = _count({"must": [{"key": "level", "match": {"value": 0}},
                                   {"key": "status", "match": {"value": "pending"}}]})

    threads         = ctx.get("active_research_threads", [])
    pending_reviews = [t for t in threads if t.get("status") == "pending_review"]
    gaps            = ctx.get("knowledge_gaps", [])
    health          = ctx.get("knowledge_health", {})
    focus           = ctx.get("user_focus_inference", {})

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "---",
        "name: ATLAS Live State",
        "description: ATLAS知识金字塔当前状态、研究线程、待确认项，每15分钟自动更新",
        "type: project",
        "---",
        "",
        f"**更新时间**: {now}",
        "",
        "## 知识金字塔",
        f"- L0 待处理: {pending_l0}",
        f"- L1 active: {l1_active}",
        f"- L2 跨域洞见: {l2_total}",
        f"- L3 框架 (active): {l3_total}",
        f"- L4 元原则: {l4_total}",
        f"- L1 不完整: {health.get('incomplete_l1', '?')}",
        "",
    ]

    if focus:
        recent_topics = focus.get("recent_topics", [])
        if recent_topics:
            lines += ["## 用户近期关注（从对话推断）"]
            for t in recent_topics[:5]:
                lines.append(f"- {t}")
            lines.append("")

    if pending_reviews:
        lines += [f"## 待确认研究 ({len(pending_reviews)} 项)"]
        for t in pending_reviews[:5]:
            lines.append(f"- {t.get('domain', '?')} (started {t.get('started', '?')[:10]})")
        lines.append("> 这些研究结果需要进入知识库，如无异议将在48h后自动确认")
        lines.append("")

    if gaps:
        lines += ["## 当前知识缺口（ATLAS 正在研究）"]
        for g in gaps[:5]:
            lines.append(f"- {g}")
        lines.append("")

    STATE_FILE.write_text("\n".join(lines), "utf-8")
    _ensure_memory_index()
    LOG.info(f"Claude memory synced: L1={l1_active} L2={l2_total} L3={l3_total} L4={l4_total}")


def _ensure_memory_index() -> None:
    try:
        text = MEMORY_MD.read_text("utf-8") if MEMORY_MD.exists() else ""
        if "atlas_live_state.md" not in text:
            text = text.rstrip() + "\n" + MEMORY_ENTRY + "\n"
            MEMORY_MD.write_text(text, "utf-8")
    except Exception as e:
        LOG.warning(f"MEMORY.md update failed: {e}")
