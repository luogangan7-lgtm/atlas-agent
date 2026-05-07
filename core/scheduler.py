"""Cycle scheduler — track fast / deep / evolution timing."""
import json
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path("/Volumes/data/obsidian-vault/agent-brain/state/scheduler.json")


def _load() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {"last_deep": "", "last_evolution": ""}


def _save(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), "utf-8")


def should_run_deep(deep_hour: int = 23) -> bool:
    state = _load()
    now   = datetime.now(timezone.utc)
    last  = state.get("last_deep", "")
    if not last:
        return now.hour >= deep_hour
    try:
        last_dt = datetime.fromisoformat(last)
        return (now - last_dt).total_seconds() >= 20 * 3600 and now.hour >= deep_hour
    except Exception:
        return False


def should_run_evolution(evolution_weekday: int = 6) -> bool:
    """evolution_weekday: 6=Sunday."""
    state = _load()
    now   = datetime.now(timezone.utc)
    last  = state.get("last_evolution", "")
    if not last:
        return now.weekday() == evolution_weekday
    try:
        last_dt = datetime.fromisoformat(last)
        return (now - last_dt).total_seconds() >= 6 * 24 * 3600 and now.weekday() == evolution_weekday
    except Exception:
        return False


def mark_deep_done() -> None:
    state = _load()
    state["last_deep"] = datetime.now(timezone.utc).isoformat()
    _save(state)


def mark_evolution_done() -> None:
    state = _load()
    state["last_evolution"] = datetime.now(timezone.utc).isoformat()
    _save(state)
