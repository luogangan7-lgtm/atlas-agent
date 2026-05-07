#!/usr/bin/env python3
"""Claude Code Stop hook — push current session content to ATLAS.

Runs after every assistant response. Tracks pushed message count to avoid
redundant writes; only pushes when ≥5 new user messages have accumulated
since the last push, or at natural breakpoints.

Writes a task_request JSON to agent-brain/messages/inbox/ for ATLAS to
process next cycle as ingest_session. ATLAS extracts topics, updates
user_focus_inference, and optionally creates L0 nodes.
"""
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

CLAUDE_PROJECTS = Path("/Users/luolimo/.claude/projects/-Users-luolimo")
ATLAS_INBOX     = Path("/Volumes/data/obsidian-vault/agent-brain/messages/inbox")
STATE_FILE      = Path("/tmp/atlas_session_push_state.json")

MIN_NEW_MESSAGES = 5   # only push when ≥N new user messages since last push
MIN_MSG_LENGTH   = 15  # ignore very short / system messages


def find_active_jsonl() -> Path | None:
    jsonls = list(CLAUDE_PROJECTS.glob("*.jsonl"))
    if not jsonls:
        return None
    return max(jsonls, key=lambda p: p.stat().st_mtime)


def extract_user_messages(jsonl_path: Path) -> list[str]:
    messages = []
    try:
        for line in jsonl_path.read_text("utf-8").strip().split("\n"):
            try:
                d = json.loads(line)
                if d.get("type") != "user":
                    continue
                content = d.get("message", {}).get("content", "")
                if isinstance(content, str):
                    text = content.strip()
                elif isinstance(content, list):
                    text = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ).strip()
                else:
                    continue
                if len(text) >= MIN_MSG_LENGTH:
                    messages.append(text)
            except Exception:
                continue
    except Exception:
        pass
    return messages


def load_state(jsonl_path: Path) -> dict:
    try:
        if STATE_FILE.exists():
            s = json.loads(STATE_FILE.read_text())
            if s.get("jsonl") == str(jsonl_path):
                return s
    except Exception:
        pass
    return {"jsonl": str(jsonl_path), "pushed_count": 0, "session_notified": False}


def save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state))
    except Exception:
        pass


def push_to_atlas(messages: list[str], jsonl_path: Path) -> None:
    ATLAS_INBOX.mkdir(parents=True, exist_ok=True)

    msg_id = str(uuid.uuid4())
    now    = datetime.now(timezone.utc).isoformat()

    session_content = "\n\n".join(f"[用户] {m}" for m in messages)

    payload = {
        "protocol_version": "1.0",
        "message_id":       msg_id,
        "timestamp":        now,
        "from":             {"agent_id": "claude-code", "type": "user_session"},
        "to":               {"agent_id": "atlas-knowledge-agent", "type": "knowledge"},
        "message_type":     "task_request",
        "payload": {
            "task_type": "ingest_session",
            "content": {
                "session_content": session_content,
                "message_count":   len(messages),
                "source":          "claude-code-session",
                "session_file":    jsonl_path.name,
                "timestamp":       now,
            },
        },
    }

    fname = f"{msg_id[:8]}-claude-code.json"
    (ATLAS_INBOX / fname).write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


def notify_pending_items() -> None:
    """On first stop of a new session, print any pending ATLAS inbox items."""
    import subprocess
    try:
        script = Path(__file__).parent / "pending_items_check.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout.strip()
        if output:
            print("\n" + output + "\n", flush=True)
    except Exception:
        pass


def main() -> None:
    active = find_active_jsonl()
    if not active:
        sys.exit(0)

    messages = extract_user_messages(active)
    total    = len(messages)

    state = load_state(active)

    # Notify pending items once per new session (first Stop after a new conversation starts)
    if not state.get("session_notified"):
        notify_pending_items()
        state["session_notified"] = True
        save_state(state)

    new_since_last = total - state["pushed_count"]
    if new_since_last < MIN_NEW_MESSAGES:
        sys.exit(0)

    push_to_atlas(messages, active)

    state["pushed_count"] = total
    save_state(state)


if __name__ == "__main__":
    main()
