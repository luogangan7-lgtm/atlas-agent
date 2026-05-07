"""Strategy adjuster — tune task limits and priorities based on performance history.

Reads the last N cycle logs, computes trends, and writes adjustments
to state/strategy.json. decision.py reads strategy.json to override defaults.
"""
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from storage import state_manager as sm
from core.health import HealthReport

LOG = logging.getLogger("atlas.strategy")

STRATEGY_FILE = Path("/Volumes/data/obsidian-vault/agent-brain/state/strategy.json")
CYCLE_LOG     = Path("/Volumes/data/obsidian-vault/agent-brain/state/cycle-log.jsonl")


def _load_strategy() -> dict:
    if STRATEGY_FILE.exists():
        try:
            return json.loads(STRATEGY_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {
        "max_l0_per_cycle":        10,
        "max_l1_complete_per_cycle": 5,
        "max_l2_per_cycle":        3,
        "max_l3_per_cycle":        1,
        "contradiction_check_every_n_cycles": 4,
        "research_cooldown_cycles": 3,
        "priority_overrides":      {},
        "updated_at":              "",
        "reasoning":               [],
    }


def _save_strategy(strategy: dict) -> None:
    STRATEGY_FILE.parent.mkdir(parents=True, exist_ok=True)
    strategy["updated_at"] = datetime.now(timezone.utc).isoformat()
    STRATEGY_FILE.write_text(json.dumps(strategy, ensure_ascii=False, indent=2), "utf-8")


def load_strategy() -> dict:
    return _load_strategy()


def _read_recent_cycles(n: int = 40) -> list[dict]:
    if not CYCLE_LOG.exists():
        return []
    lines = CYCLE_LOG.read_text("utf-8").strip().splitlines()
    recent = lines[-n:]
    result = []
    for line in recent:
        try:
            result.append(json.loads(line))
        except Exception:
            pass
    return result


def _analyze_task_trend(cycles: list[dict], task_type: str) -> dict:
    """Compute trend stats for a specific task type across cycles."""
    outputs    = []
    durations  = []
    error_count = 0

    for cycle in cycles:
        duration_ms = cycle.get("duration_ms", 0)
        for task in cycle.get("tasks_completed", []):
            if task.get("type") != task_type:
                continue
            if task.get("error"):
                error_count += 1
            for key in ("nodes_created", "completed", "generated", "synthesized"):
                v = task.get(key, 0)
                if v:
                    outputs.append(v)
        if any(t.get("type") == task_type for t in cycle.get("tasks_completed", [])):
            durations.append(duration_ms)

    runs = len(outputs) + error_count
    return {
        "runs":        runs,
        "error_rate":  error_count / runs if runs > 0 else 0,
        "avg_output":  sum(outputs) / len(outputs) if outputs else 0,
        "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
    }


def adjust_strategy(health: HealthReport) -> dict:
    LOG.info("Adjusting strategy based on performance data...")

    cycles   = _read_recent_cycles(40)
    strategy = _load_strategy()
    reasoning: list[str] = []

    # ── L0 organizing ────────────────────────────────────────────
    # If no pending L0 recently, keep at 10; if backlog builds fast, increase
    strategy["max_l0_per_cycle"] = 10  # keep default

    # ── L1 completion ────────────────────────────────────────────
    if health.l1_incomplete_count > 50:
        strategy["max_l1_complete_per_cycle"] = 8
        reasoning.append(f"L1 incomplete={health.l1_incomplete_count} > 50 → raised l1_complete limit to 8")
    elif health.l1_incomplete_count < 10:
        strategy["max_l1_complete_per_cycle"] = 3
        reasoning.append(f"L1 incomplete={health.l1_incomplete_count} < 10 → reduced l1_complete limit to 3")
    else:
        strategy["max_l1_complete_per_cycle"] = 5

    # ── L2 generation ─────────────────────────────────────────────
    l2_trend = _analyze_task_trend(cycles, "l2_generate")
    if l2_trend["avg_output"] < 0.5 and l2_trend["runs"] > 5:
        # Not generating much → maybe increase batch to try harder
        strategy["max_l2_per_cycle"] = 4
        reasoning.append("L2 avg output low → raised to 4 per cycle")
    elif health.l2_per_l1 > 0.5:
        strategy["max_l2_per_cycle"] = 2
        reasoning.append(f"L2/L1 ratio={health.l2_per_l1:.2f} > 0.5 → reduced L2 gen to 2")
    else:
        strategy["max_l2_per_cycle"] = 3

    # ── L3 synthesis ──────────────────────────────────────────────
    if health.l3_total > 200:
        strategy["max_l3_per_cycle"] = 0  # enough L3, focus elsewhere
        reasoning.append(f"L3 count={health.l3_total} > 200 → paused L3 synthesis")
    else:
        strategy["max_l3_per_cycle"] = 1

    # ── Contradiction detection cadence ──────────────────────────
    cont_trend = _analyze_task_trend(cycles, "contradiction_detect")
    if cont_trend.get("avg_output", 0) == 0 and cont_trend["runs"] > 5:
        # No contradictions found → run less often
        strategy["contradiction_check_every_n_cycles"] = 8
        reasoning.append("No contradictions found recently → check every 8 cycles")
    else:
        strategy["contradiction_check_every_n_cycles"] = 4

    # ── Research cooldown ──────────────────────────────────────────
    if len(health.gap_domains) > 10:
        strategy["research_cooldown_cycles"] = 2  # more frequent research
        reasoning.append(f"{len(health.gap_domains)} gap domains → research every 2 cycles")
    else:
        strategy["research_cooldown_cycles"] = 4

    # ── Priority overrides based on health grade ───────────────────
    if health.grade in ("C", "D"):
        # Focus on fundamentals: L1 completion + L0 processing
        strategy["priority_overrides"] = {
            "l1_complete": 0.65,
            "l2_generate": 0.30,
        }
        reasoning.append(f"Health grade={health.grade} → boosted L1 completion priority")
    elif health.grade == "A":
        # All good, focus on discovery
        strategy["priority_overrides"] = {
            "autonomous_research": 0.55,
            "l2_generate":         0.55,
        }
        reasoning.append("Health grade=A → boosted research and L2 generation")
    else:
        strategy["priority_overrides"] = {}

    strategy["reasoning"] = reasoning[-10:]  # keep last 10 reasons
    _save_strategy(strategy)

    LOG.info(f"Strategy adjusted: {reasoning}")
    return {"adjustments": len(reasoning), "grade": health.grade}
