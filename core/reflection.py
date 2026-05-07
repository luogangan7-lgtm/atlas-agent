"""Reflection layer — log cycle results and update state."""
import logging
from datetime import datetime, timezone

from storage import state_manager as sm

LOG = logging.getLogger("atlas.reflection")


def reflect(cycle_id: str, duration_ms: int, tasks_completed: list[dict], errors: list[str]) -> None:
    entry = {
        "cycle_id": cycle_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "tasks_completed": tasks_completed,
        "errors": errors,
    }
    sm.append_cycle_log(entry)

    # Update agenda completed_today counter
    agenda = sm.load_agenda()
    done_count = sum(t.get("count", 1) for t in tasks_completed)
    agenda["completed_today"] = agenda.get("completed_today", 0) + done_count
    agenda["current_cycle"] = cycle_id
    sm.save_agenda(agenda)

    LOG.info(
        f"Cycle {cycle_id} reflected: {len(tasks_completed)} task types done, "
        f"{len(errors)} errors, duration={duration_ms}ms"
    )
