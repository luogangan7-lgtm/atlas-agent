"""Todo Agent — process task requests from Obsidian todo files.

Reads agent-brain/todo/pending.md for tasks tagged with #atlas.
Supported task types:
  - research:<domain>  → triggers autonomous research for that domain
  - summarize:<topic>  → queues a knowledge synthesis request
  - remind:<text>      → writes a reminder to the daily report context

Completed tasks are moved to done.md.
"""
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from storage import state_manager as sm

LOG = logging.getLogger("atlas.todo_agent")

VAULT     = Path("/Volumes/data/obsidian-vault")
TODO_DIR  = VAULT / "agent-brain" / "todo"
PENDING   = TODO_DIR / "pending.md"
DONE      = TODO_DIR / "done.md"


def _ensure_todo_dir() -> None:
    TODO_DIR.mkdir(parents=True, exist_ok=True)
    if not PENDING.exists():
        PENDING.write_text(
            "# ATLAS 任务列表\n\n"
            "在此添加任务，格式：`- [ ] <任务描述> #atlas`\n"
            "支持的命令：\n"
            "- `research:<领域>` — 研究某领域知识\n"
            "- `summarize:<主题>` — 汇总某主题知识\n"
            "- `remind:<文字>` — 添加提醒\n\n",
            encoding="utf-8",
        )


def _parse_tasks(text: str) -> list[tuple[int, str]]:
    """Return list of (line_index, task_text) for unchecked #atlas tasks."""
    results = []
    for i, line in enumerate(text.splitlines()):
        if re.match(r"\s*-\s*\[x\]", line, re.IGNORECASE):
            continue
        m = re.match(r"\s*-\s*\[\s*\]\s*(.+)", line)
        if m and "#atlas" in line:
            results.append((i, m.group(1).strip()))
    return results


def _handle_task(task_text: str) -> str | None:
    """Execute a single todo task. Returns result description or None."""
    text = task_text.replace("#atlas", "").strip()

    if text.lower().startswith("research:"):
        domain = text[len("research:"):].strip()
        if domain:
            try:
                from actions.maintenance.research import run_autonomous_research, _find_gap_domains, _research_domain, _write_research_inbox, _stable_id
                from datetime import datetime, timezone
                LOG.info(f"Todo: research requested for domain '{domain}'")
                now = datetime.now(timezone.utc).isoformat()
                result = _research_domain(domain, 0)
                if result and result.get("nodes"):
                    nodes = result["nodes"]
                    l0_ids = []
                    from storage import qdrant_client as qc
                    from utils.embed import get_embedding
                    for node in nodes:
                        title = (node.get("title") or "").strip()
                        if not title:
                            continue
                        payload = {
                            "level": 0, "status": "pending_review",
                            "title": title, "domain": domain,
                            "content": node.get("summary") or "",
                            "content_type": node.get("content_type") or "principle",
                            "source": "atlas-todo-agent",
                            "created_at": now,
                            "summary": node.get("summary") or "",
                            "key_points": node.get("key_points") or [],
                            "rule_statement": node.get("rule_statement") or "",
                            "tags": node.get("tags") or [],
                        }
                        pid = _stable_id(f"todo-research-{domain}-{title}-{now}")
                        vec = get_embedding(f"{title}\n{node.get('summary', '')}") or [0.0] * 1024
                        try:
                            qc.upsert_point(pid, vec, payload)
                            l0_ids.append(pid)
                        except Exception as e:
                            LOG.error(f"Todo research store failed: {e}")
                    if l0_ids:
                        _write_research_inbox(domain, nodes, l0_ids)
                    return f"已研究领域「{domain}」，生成 {len(l0_ids)} 个知识节点等待审核"
            except Exception as e:
                LOG.error(f"Todo research task failed: {e}")
                return f"研究「{domain}」失败: {e}"

    elif text.lower().startswith("remind:"):
        reminder = text[len("remind:"):].strip()
        if reminder:
            ctx = sm.load_context()
            reminders = ctx.get("user_reminders", [])
            reminders.append({
                "text": reminder,
                "created": datetime.now(timezone.utc).isoformat(),
            })
            ctx["user_reminders"] = reminders[-10:]  # keep last 10
            sm.save_context(ctx)
            return f"提醒已记录：{reminder}"

    elif text.lower().startswith("summarize:"):
        topic = text[len("summarize:"):].strip()
        if topic:
            ctx = sm.load_context()
            queue = ctx.get("summarize_queue", [])
            queue.append({"topic": topic, "created": datetime.now(timezone.utc).isoformat()})
            ctx["summarize_queue"] = queue
            sm.save_context(ctx)
            return f"摘要请求已加入队列：{topic}"

    return None


def _mark_done(lines: list[str], line_idx: int, result: str) -> list[str]:
    lines[line_idx] = re.sub(r"(\s*-\s*)\[\s*\]", r"\1[x]", lines[line_idx], count=1)
    return lines


def _append_done(task_text: str, result: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    entry = f"- [x] {task_text}  *(完成于 {now} — {result})*\n"
    if not DONE.exists():
        DONE.write_text("# 已完成任务\n\n", encoding="utf-8")
    with DONE.open("a", encoding="utf-8") as f:
        f.write(entry)


def process_todo_tasks() -> dict:
    _ensure_todo_dir()

    if not PENDING.exists():
        return {"processed": 0}

    text = PENDING.read_text(encoding="utf-8")
    tasks = _parse_tasks(text)
    if not tasks:
        return {"processed": 0}

    LOG.info(f"Todo agent: found {len(tasks)} pending task(s)")
    lines = text.splitlines()
    processed = 0

    for line_idx, task_text in tasks:
        result = _handle_task(task_text)
        if result is not None:
            lines = _mark_done(lines, line_idx, result)
            _append_done(task_text, result)
            LOG.info(f"Todo task done: {task_text[:60]} → {result}")
            processed += 1

    if processed > 0:
        PENDING.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"processed": processed}
