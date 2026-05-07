#!/usr/bin/env python3
"""Check ATLAS inbox for pending user decisions.

Called by agent hooks (Claude Code Stop, openclaw before_prompt_build) to notify
the user at the start of each new session.

CLI usage:
  python3 pending_items_check.py           # human-readable notification (exit 0 always)
  python3 pending_items_check.py --json    # JSON array (exit 0 always)
  python3 pending_items_check.py --count   # just the integer count
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

INBOX_DIR = Path("/Volumes/data/obsidian-vault/agent-brain/inbox")

# Categories that require explicit user confirmation — never auto-processed
DELETION_CATEGORIES = {"node_delete", "obsolete_cleanup", "delete_confirm", "mass_delete"}

AUTO_CONFIRM_HOURS = 24


def _parse_frontmatter(text: str) -> dict:
    fm: dict = {}
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return fm
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "未知"


def _hours_pending(created_str: str) -> float:
    if not created_str:
        return 0.0
    try:
        created = datetime.fromisoformat(created_str)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created).total_seconds() / 3600
    except Exception:
        return 0.0


def get_pending_items() -> list[dict]:
    if not INBOX_DIR.exists():
        return []
    items = []
    for md_file in sorted(INBOX_DIR.glob("*.md")):
        try:
            text = md_file.read_text("utf-8")
            fm = _parse_frontmatter(text)
            status = fm.get("status", "pending")
            if status in ("processed", "approved", "rejected", "modified"):
                continue
            category = fm.get("category", "unknown")
            hours = _hours_pending(fm.get("created", ""))
            is_deletion = category in DELETION_CATEGORIES
            items.append({
                "file":              md_file.name,
                "category":          category,
                "domain":            fm.get("domain", ""),
                "created":           fm.get("created", ""),
                "hours_pending":     round(hours, 1),
                "is_deletion":       is_deletion,
                "title":             _extract_title(text),
                "auto_confirm_hours": None if is_deletion else AUTO_CONFIRM_HOURS,
            })
        except Exception:
            pass
    return items


def format_notification(items: list[dict]) -> str:
    if not items:
        return ""

    lines = [f"## ATLAS 有 {len(items)} 条待处理事项，需要你决定：", ""]

    for i, item in enumerate(items, 1):
        domain = f"[{item['domain']}]" if item["domain"] else ""
        hours  = item["hours_pending"]
        if item["is_deletion"]:
            auto_label = "需手动确认（删除操作不自动处理）"
        else:
            remaining = max(0.0, AUTO_CONFIRM_HOURS - hours)
            if remaining < 1:
                auto_label = "即将自动确认"
            else:
                auto_label = f"{remaining:.0f}h 后自动确认"

        lines.append(f"**{i}.** {item['title']} {domain}")
        lines.append(f"   类型: `{item['category']}` | 等待: {hours:.0f}h | {auto_label}")

    lines += [
        "",
        "如需现在处理，请告诉我：",
        "  - 「第1条 确认」「第2条 拒绝」",
        "  - 「全部确认」",
        "  - 「第N条 查看详情」",
        "",
        "不回复则等待自动处理（删除类除外）。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    items = get_pending_items()

    if "--count" in sys.argv:
        print(len(items))
    elif "--json" in sys.argv:
        print(json.dumps(items, ensure_ascii=False, indent=2))
    else:
        notif = format_notification(items)
        if notif:
            print(notif)
