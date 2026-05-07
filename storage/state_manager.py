"""Manage agent state files: agenda.json, context.json, self-model.json, cycle-log.jsonl."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

BRAIN_DIR = Path("/Volumes/data/obsidian-vault/agent-brain")
STATE_DIR = BRAIN_DIR / "state"


def _ensure() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (BRAIN_DIR / "daily").mkdir(parents=True, exist_ok=True)
    (BRAIN_DIR / "weekly").mkdir(parents=True, exist_ok=True)
    (BRAIN_DIR / "inbox").mkdir(parents=True, exist_ok=True)
    (BRAIN_DIR / "outbox").mkdir(parents=True, exist_ok=True)
    (BRAIN_DIR / "messages" / "inbox").mkdir(parents=True, exist_ok=True)
    (BRAIN_DIR / "messages" / "outbox").mkdir(parents=True, exist_ok=True)


def load(name: str, default: Any = None) -> Any:
    _ensure()
    path = STATE_DIR / f"{name}.json"
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return default if default is not None else {}


def save(name: str, data: Any) -> None:
    _ensure()
    path = STATE_DIR / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def append_cycle_log(entry: dict) -> None:
    _ensure()
    log_path = STATE_DIR / "cycle-log.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_agenda() -> dict:
    return load("agenda", {
        "last_updated": "",
        "current_cycle": "",
        "queue": [],
        "completed_today": 0,
        "blocked": [],
    })


def save_agenda(agenda: dict) -> None:
    agenda["last_updated"] = datetime.utcnow().isoformat() + "Z"
    save("agenda", agenda)


def load_context() -> dict:
    return load("context", {
        "last_updated": "",
        "user_focus_inference": {},
        "active_research_threads": [],
        "pending_user_decisions": 0,
        "knowledge_health": {},
    })


def save_context(ctx: dict) -> None:
    ctx["last_updated"] = datetime.utcnow().isoformat() + "Z"
    save("context", ctx)


def load_self_model() -> dict:
    return load("self-model", {
        "last_updated": "",
        "domain_expertise": {},
        "capability_assessment": {
            "l0_to_l1_quality": 0.8,
            "l2_generation_quality": 0.7,
            "l3_synthesis_quality": 0.6,
        },
        "active_prompts": {
            "l2_generation": "v1",
            "l1_completion": "v1",
        },
        "known_weaknesses": [],
    })


def save_self_model(model: dict) -> None:
    model["last_updated"] = datetime.utcnow().isoformat() + "Z"
    save("self-model", model)
