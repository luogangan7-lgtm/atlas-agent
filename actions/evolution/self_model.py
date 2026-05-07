"""Dynamic self-model updater.

Runs during the evolution cycle. Computes:
- domain_expertise: depth score per domain based on real Qdrant stats
- capability_assessment: quality trends from recent cycle logs
- behavioral_preferences: what domains/types the agent has been processing most
"""
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc
from storage import state_manager as sm

LOG = logging.getLogger("atlas.self_model")

CYCLE_LOG = Path("/Volumes/data/obsidian-vault/agent-brain/state/cycle-log.jsonl")


def _compute_domain_expertise() -> dict:
    """Compute depth score per domain from Qdrant stats."""
    l1_pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 1}},
            {"key": "status", "match": {"value": "active"}},
        ],
        "must_not": [
            {"key": "record_type", "match": {"value": "entity"}},
            {"key": "record_type", "match": {"value": "relation"}},
        ],
    }, limit=1000)

    l3_pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 3}},
            {"key": "status", "match": {"value": "active"}},
        ]
    }, limit=500)

    l1_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "score_sum": 0.0})
    for pt in l1_pts:
        d = pt["payload"].get("domain") or "未分类"
        l1_stats[d]["count"] += 1
        l1_stats[d]["score_sum"] += pt["payload"].get("completeness_score", 0)

    l3_counts: dict[str, int] = defaultdict(int)
    for pt in l3_pts:
        d = pt["payload"].get("domain") or "未分类"
        l3_counts[d] += 1

    expertise = {}
    for domain, stats in l1_stats.items():
        count = stats["count"]
        avg_completeness = stats["score_sum"] / count if count > 0 else 0
        l3_count = l3_counts.get(domain, 0)

        # depth formula: L1 saturation × 0.4 + L3 coverage × 0.4 + completeness × 0.2
        l1_saturation  = min(count / 30.0, 1.0)   # 30 L1 = full saturation
        l3_coverage    = min(l3_count / 5.0, 1.0)  # 5 L3 = full coverage
        depth = l1_saturation * 0.4 + l3_coverage * 0.4 + avg_completeness * 0.2

        expertise[domain] = {
            "depth":            round(depth, 3),
            "l1_count":         count,
            "l3_count":         l3_count,
            "avg_completeness": round(avg_completeness, 3),
        }

    return expertise


def _compute_capability_assessment() -> dict:
    """Analyse recent cycle logs for task quality metrics."""
    if not CYCLE_LOG.exists():
        return {}

    lines = CYCLE_LOG.read_text("utf-8").strip().splitlines()
    recent = lines[-50:]  # last 50 cycles

    task_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "errors": 0, "total_out": 0})

    for line in recent:
        try:
            entry = json.loads(line)
        except Exception:
            continue
        for task in entry.get("tasks_completed", []):
            t = task.get("type", "")
            task_stats[t]["count"] += 1
            if task.get("error"):
                task_stats[t]["errors"] += 1
            # Count outputs
            for key in ("nodes_created", "completed", "generated", "synthesized"):
                task_stats[t]["total_out"] += task.get(key, 0)

    assessment = {}
    for task_type, stats in task_stats.items():
        count = stats["count"]
        if count == 0:
            continue
        success_rate = 1.0 - (stats["errors"] / count)
        avg_output   = stats["total_out"] / count
        assessment[task_type] = {
            "runs":         count,
            "success_rate": round(success_rate, 3),
            "avg_output":   round(avg_output, 2),
        }

    return assessment


def _compute_preferences(expertise: dict) -> dict:
    """Identify preferred/curiosity domains from expertise profile."""
    sorted_domains = sorted(expertise.items(), key=lambda x: x[1]["depth"], reverse=True)
    preferred   = [d for d, v in sorted_domains[:5] if v["depth"] > 0.5]
    curiosity   = [d for d, v in sorted_domains if v["depth"] < 0.3 and v["l1_count"] >= 3]
    weak        = [d for d, v in sorted_domains if v["depth"] < 0.2]

    return {
        "preferred_domains": preferred,
        "curiosity_domains": curiosity[:5],
        "weak_domains":      weak[:5],
    }


def update_self_model() -> dict:
    LOG.info("Updating self-model...")

    domain_expertise  = _compute_domain_expertise()
    capability        = _compute_capability_assessment()
    preferences       = _compute_preferences(domain_expertise)

    model = sm.load_self_model()
    model["domain_expertise"]      = domain_expertise
    model["capability_assessment"] = capability
    model["behavioral_preferences"] = preferences
    model["last_updated"]           = datetime.now(timezone.utc).isoformat()

    sm.save_self_model(model)

    top_domains = sorted(domain_expertise.items(), key=lambda x: x[1]["depth"], reverse=True)[:3]
    LOG.info(f"Self-model updated. Top domains: {[(d, v['depth']) for d, v in top_domains]}")

    return {
        "domains_tracked":  len(domain_expertise),
        "top_domain":       top_domains[0][0] if top_domains else "",
        "capability_tasks": len(capability),
    }
