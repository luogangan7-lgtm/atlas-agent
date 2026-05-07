"""External agent file-based communication protocol.

Reads messages from agent-brain/messages/inbox/*.json
Writes responses to agent-brain/messages/outbox/*.json

Message format (per design doc):
{
  "protocol_version": "1.0",
  "message_id": "uuid",
  "timestamp": "...",
  "from": {"agent_id": "...", "type": "..."},
  "to":   {"agent_id": "atlas-knowledge-agent", "type": "knowledge"},
  "message_type": "task_request|query|handshake",
  "payload": {"task_type": "research|summarize|query", "content": {}, ...}
}
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from storage import state_manager as sm

LOG = logging.getLogger("atlas.protocol")

VAULT      = Path("/Volumes/data/obsidian-vault")
MSG_INBOX  = VAULT / "agent-brain" / "messages" / "inbox"
MSG_OUTBOX = VAULT / "agent-brain" / "messages" / "outbox"
AGENT_ID   = "atlas-knowledge-agent"


def _send(to_agent: str, message_type: str, payload: dict) -> None:
    MSG_OUTBOX.mkdir(parents=True, exist_ok=True)
    msg = {
        "protocol_version": "1.0",
        "message_id":       str(uuid.uuid4()),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "from":             {"agent_id": AGENT_ID, "type": "knowledge"},
        "to":               {"agent_id": to_agent, "type": "unknown"},
        "message_type":     message_type,
        "payload":          payload,
    }
    fname = f"{msg['message_id'][:8]}-{to_agent}.json"
    (MSG_OUTBOX / fname).write_text(json.dumps(msg, ensure_ascii=False, indent=2), "utf-8")
    LOG.info(f"Sent {message_type} to {to_agent}: {fname}")


def _handle_task_request(msg: dict) -> str:
    """Handle task_request messages from external agents."""
    payload   = msg.get("payload", {})
    task_type = payload.get("task_type", "")
    from_id   = msg.get("from", {}).get("agent_id", "unknown")

    LOG.info(f"External task request from {from_id}: {task_type}")

    if task_type == "research":
        topic = payload.get("content", {}).get("topic") or payload.get("topic", "")
        if topic:
            # Store as L0 pending for processing
            from storage import qdrant_client as qc
            import hashlib
            seed = f"ext-task-{topic}-{msg.get('message_id', '')}"
            h = hashlib.md5(seed.encode()).hexdigest()
            point_id = int(h[:16], 16) % (2 ** 53)
            now = datetime.now(timezone.utc).isoformat()
            qc._req("PUT", f"collections/{qc.COLLECTION}/points?wait=true", {
                "points": [{
                    "id":      point_id,
                    "vector":  [0.0] * 1024,
                    "payload": {
                        "level":     0,
                        "status":    "pending",
                        "content":   f"研究主题：{topic}",
                        "source":    f"external-agent:{from_id}",
                        "created_at": now,
                        "ext_task_id": msg.get("message_id"),
                        "ext_from":    from_id,
                    },
                }]
            })
            return f"queued_research:{point_id}"

    elif task_type == "query":
        # Simple knowledge query — search Qdrant
        query_text = payload.get("content", {}).get("query") or payload.get("query", "")
        if query_text:
            from utils.embed import get_embedding
            from storage import qdrant_client as qc
            vector = get_embedding(query_text)
            if vector:
                results = qc.search(vector, limit=3, score_threshold=0.6)
                if results:
                    answer = results[0]["payload"].get("summary") or ""
                    _send(from_id, "task_result", {
                        "task_id":    msg.get("message_id"),
                        "status":     "completed",
                        "answer":     answer,
                        "confidence": results[0].get("score", 0),
                    })
                    return "query_answered"

    elif task_type == "ingest_session":
        return _handle_ingest_session(payload, from_id)

    return "unhandled"


def _handle_ingest_session(payload: dict, from_id: str) -> str:
    """Process a Claude Code session dump: extract topics, update user_focus, queue L0."""
    content = payload.get("content", {})
    session_text = content.get("session_content", "")
    if not session_text or len(session_text) < 50:
        return "skipped:too_short"

    now = datetime.now(timezone.utc).isoformat()

    # Extract user-focused lines for topic inference
    user_lines = [
        line[len("[用户] "):].strip()
        for line in session_text.split("\n\n")
        if line.startswith("[用户] ")
    ]

    # Update user_focus_inference in context
    try:
        ctx = sm.load_context()
        focus = ctx.get("user_focus_inference", {})
        recent = focus.get("recent_topics", [])
        # Add new user messages as recent topics (deduplicated, last 20)
        for msg_text in user_lines[-5:]:
            short = msg_text[:80]
            if short not in recent:
                recent.append(short)
        focus["recent_topics"]  = recent[-20:]
        focus["last_session"]   = now
        focus["session_source"] = from_id
        ctx["user_focus_inference"] = focus
        sm.save_context(ctx)
    except Exception as e:
        LOG.warning(f"user_focus update failed: {e}")

    # Store session as L0 pending for knowledge extraction
    try:
        from storage import qdrant_client as qc
        import hashlib
        seed     = f"session-{from_id}-{content.get('session_file', '')}-{content.get('message_count', 0)}"
        h        = hashlib.md5(seed.encode()).hexdigest()
        point_id = int(h[:16], 16) % (2 ** 53)

        qc._req("PUT", f"collections/{qc.COLLECTION}/points?wait=true", {
            "points": [{
                "id":      point_id,
                "vector":  [0.0] * 1024,
                "payload": {
                    "level":      0,
                    "status":     "pending",
                    "content":    session_text[:4000],
                    "source":     f"session:{from_id}",
                    "domain":     "用户会话",
                    "created_at": now,
                },
            }]
        })
        LOG.info(f"Session ingested as L0 {point_id} ({len(user_lines)} user msgs)")
        return f"ingested:{point_id}"
    except Exception as e:
        LOG.error(f"Session L0 write failed: {e}")
        return f"error:{e}"


def _handle_handshake(msg: dict) -> None:
    from_id = msg.get("from", {}).get("agent_id", "unknown")
    _send(from_id, "handshake", {
        "agent_id":   AGENT_ID,
        "capabilities": ["knowledge_storage", "l0_to_l4_processing", "research", "query"],
        "status":     "active",
        "knowledge_stats": sm.load_context().get("knowledge_health", {}),
    })


def process_external_messages() -> dict:
    """Scan messages/inbox/ and handle incoming messages."""
    if not MSG_INBOX.exists():
        return {"processed": 0}

    processed = 0

    for json_file in sorted(MSG_INBOX.glob("*.json")):
        try:
            msg = json.loads(json_file.read_text("utf-8"))
        except Exception as e:
            LOG.warning(f"Invalid message {json_file.name}: {e}")
            json_file.unlink(missing_ok=True)
            continue

        msg_type = msg.get("message_type", "")
        LOG.info(f"Processing external message: {json_file.name} type={msg_type}")

        try:
            if msg_type == "task_request":
                _handle_task_request(msg)
            elif msg_type == "handshake":
                _handle_handshake(msg)
            elif msg_type == "task_result":
                # External agent completed a task we delegated
                LOG.info(f"Task result received: {msg.get('payload', {}).get('status')}")
            else:
                LOG.warning(f"Unknown message type: {msg_type}")

        except Exception as e:
            LOG.error(f"Handle message failed: {e}")

        # Archive processed message
        archive = MSG_INBOX.parent / "inbox_processed"
        archive.mkdir(exist_ok=True)
        json_file.rename(archive / json_file.name)
        processed += 1

    if processed > 0:
        LOG.info(f"External messages processed: {processed}")

    return {"processed": processed}
