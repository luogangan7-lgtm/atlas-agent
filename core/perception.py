"""Perception layer — scan Qdrant + Obsidian and build world state."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc

LOG = logging.getLogger("atlas.perception")
VAULT = Path("/Volumes/data/obsidian-vault")
BRAIN_INBOX = VAULT / "agent-brain" / "inbox"


@dataclass
class WorldState:
    timestamp: str = ""
    cycle_id: str = ""

    # Knowledge quantities
    pending_l0: int = 0
    incomplete_l1: int = 0
    l1_total: int = 0
    l2_total: int = 0
    l3_total: int = 0
    l4_total: int = 0

    # Quality signals
    knowledge_gaps: list[str] = field(default_factory=list)  # domains with < 5 L1
    stale_count: int = 0
    unclassified_count: int = 0   # nodes with domain="未分类"
    no_category_count: int = 0    # nodes without category field
    garbled_count: int = 0        # nodes with U+FFFD in title

    # Communication
    user_responses: int = 0        # inbox/ files with status:approved/rejected
    external_messages: int = 0     # messages/inbox/ files

    # Derived
    urgency_signals: list[dict] = field(default_factory=list)


def _count_l0_pending() -> int:
    return qc.count({
        "must": [
            {"key": "level",  "match": {"value": 0}},
            {"key": "status", "match": {"any": ["pending", "active"]}},
        ]
    })


def _count_incomplete_l1() -> int:
    _base_must = [
        {"key": "level",  "match": {"value": 1}},
        {"key": "status", "match": {"value": "active"}},
    ]
    _base_must_not = [
        {"key": "record_type", "match": {"value": "entity"}},
        {"key": "record_type", "match": {"value": "relation"}},
    ]
    # Nodes with score present but below threshold
    has_low_score = qc.count({
        "must": _base_must + [{"key": "completeness_score", "range": {"lt": 0.85}}],
        "must_not": _base_must_not,
    })
    # Nodes with no completeness_score field at all
    no_score = qc.count({
        "must": _base_must + [{"is_empty": {"key": "completeness_score"}}],
        "must_not": _base_must_not,
    })
    return has_low_score + no_score


def _count_level(level: int) -> int:
    f: dict = {"must": [{"key": "level", "match": {"value": level}},
                         {"key": "status", "match": {"value": "active"}}]}
    if level == 1:
        f["must_not"] = [{"key": "record_type", "match": {"value": "entity"}},
                          {"key": "record_type", "match": {"value": "relation"}}]
    return qc.count(f)


def _find_knowledge_gaps() -> list[str]:
    """Return domain names where active L1 count < 5."""
    # Scroll a sample of L1 records to see domain distribution
    pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 1}},
            {"key": "status", "match": {"value": "active"}},
        ],
        "must_not": [
            {"key": "record_type", "match": {"value": "entity"}},
            {"key": "record_type", "match": {"value": "relation"}},
        ],
    }, limit=200)

    domain_counts: dict[str, int] = {}
    for pt in pts:
        d = pt["payload"].get("domain") or "未分类"
        domain_counts[d] = domain_counts.get(d, 0) + 1

    return [d for d, c in domain_counts.items() if c < 5]


def _count_user_responses() -> int:
    if not BRAIN_INBOX.exists():
        return 0
    count = 0
    for md_file in BRAIN_INBOX.glob("*.md"):
        try:
            text = md_file.read_text("utf-8")
            if "status: approved" in text or "status: rejected" in text or "status: modified" in text:
                count += 1
        except Exception:
            pass
    return count


def _count_quality_issues() -> tuple[int, int, int]:
    """Return (unclassified_count, no_category_count, garbled_count)."""
    pts = qc.scroll({
        "must": [{"key": "status", "match": {"value": "active"}}]
    }, limit=5000)

    unclassified = sum(1 for p in pts if (p["payload"].get("domain") or "") == "未分类")
    no_category  = sum(1 for p in pts if not (p["payload"].get("category") or "").strip()
                       and (p["payload"].get("domain") or "") not in ("", "未分类"))
    garbled      = sum(1 for p in pts if "?" in (p["payload"].get("title") or ""))
    return unclassified, no_category, garbled


def _count_external_messages() -> int:
    msg_inbox = VAULT / "agent-brain" / "messages" / "inbox"
    if not msg_inbox.exists():
        return 0
    return len(list(msg_inbox.glob("*.json")))


def perceive() -> WorldState:
    now = datetime.now(timezone.utc)
    cycle_id = f"cycle-{now.strftime('%Y%m%d-%H%M')}"

    LOG.info("Perceiving world state...")

    ws = WorldState(
        timestamp=now.isoformat(),
        cycle_id=cycle_id,
    )

    try:
        ws.pending_l0 = _count_l0_pending()
    except Exception as e:
        LOG.warning(f"Failed to count L0: {e}")

    try:
        ws.incomplete_l1 = _count_incomplete_l1()
        ws.l1_total = _count_level(1)
        ws.l2_total = _count_level(2)
        ws.l3_total = _count_level(3)
        ws.l4_total = _count_level(4)
    except Exception as e:
        LOG.warning(f"Failed to count levels: {e}")

    try:
        ws.knowledge_gaps = _find_knowledge_gaps()
    except Exception as e:
        LOG.warning(f"Failed to find gaps: {e}")

    try:
        ws.unclassified_count, ws.no_category_count, ws.garbled_count = _count_quality_issues()
    except Exception as e:
        LOG.warning(f"Failed to count quality issues: {e}")

    try:
        ws.user_responses    = _count_user_responses()
        ws.external_messages = _count_external_messages()
    except Exception as e:
        LOG.warning(f"Failed to count messages: {e}")

    # Build urgency signals
    if ws.external_messages > 0:
        ws.urgency_signals.append({"type": "external_message", "count": ws.external_messages, "priority": "high"})
    if ws.pending_l0 > 0:
        ws.urgency_signals.append({"type": "new_l0", "count": ws.pending_l0})
    if ws.user_responses > 0:
        ws.urgency_signals.append({"type": "user_response", "count": ws.user_responses})
    if ws.incomplete_l1 > 10:
        ws.urgency_signals.append({"type": "incomplete_l1", "count": ws.incomplete_l1})

    LOG.info(
        f"World state: L0_pending={ws.pending_l0} L1={ws.l1_total}(incomplete={ws.incomplete_l1}) "
        f"L2={ws.l2_total} L3={ws.l3_total} L4={ws.l4_total} gaps={len(ws.knowledge_gaps)} "
        f"quality(unclassified={ws.unclassified_count} no_cat={ws.no_category_count} garbled={ws.garbled_count})"
    )
    return ws
