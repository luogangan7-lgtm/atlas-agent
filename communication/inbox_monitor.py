"""Inbox monitor — process user replies to agent questions.

Detects two signal types:
  A) Checkbox: user ticked `[x]` or `[X]` in the file (status stays 'pending')
  B) Status field: user manually changed status: pending → approved/rejected/modified

Auto-timeout: if a file has been pending for > AUTO_CONFIRM_HOURS with no user
response, it is auto-confirmed with a default action and logged clearly.
  - research_confirm → 全部确认 (safe: additive only)
  - contradiction     → keep higher completeness_score; tie-break: newer created_at

After processing, rewrites the file's status field to 'processed' and moves to outbox/.
"""
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import json

from storage import qdrant_client as qc
from storage import state_manager as sm

AUTO_CONFIRM_HOURS = 24

# Categories that involve deletion — require explicit user confirmation, never auto-confirmed
_DELETION_CATEGORIES = {"node_delete", "obsolete_cleanup", "delete_confirm", "mass_delete"}

LOG = logging.getLogger("atlas.inbox")

VAULT         = Path("/Volumes/data/obsidian-vault")
INBOX_DIR     = VAULT / "agent-brain" / "inbox"
OUTBOX_DIR    = VAULT / "agent-brain" / "outbox"
TAXONOMY_FILE = VAULT / "agent-brain" / "state" / "taxonomy.json"


def _read_frontmatter(text: str) -> dict:
    fm: dict = {}
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return fm
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm


def _get_checked_choices(text: str) -> list[str]:
    """Return all lines where the user ticked [x] or [X]."""
    return [m.strip() for m in re.findall(r"-\s*\[[xX]\]\s*(.+)", text)]


def _update_status_in_file(path: Path, new_status: str) -> None:
    """Rewrite status field in frontmatter without altering rest of file."""
    try:
        text = path.read_text("utf-8")
        updated = re.sub(
            r"^(status:\s*).*$", f"\\g<1>{new_status}", text, count=1, flags=re.MULTILINE
        )
        path.write_text(updated, "utf-8")
    except Exception as e:
        LOG.warning(f"Could not update status in {path.name}: {e}")


def _move_to_outbox(inbox_file: Path, result: str) -> None:
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    try:
        content = inbox_file.read_text("utf-8")
        content += f"\n\n---\n**处理结果：** {result}\n"
        dest = OUTBOX_DIR / inbox_file.name
        dest.write_text(content, "utf-8")
        inbox_file.unlink()
    except Exception as e:
        LOG.warning(f"Move to outbox failed for {inbox_file.name}: {e}")


# ── Auto-timeout helpers ─────────────────────────────────────────────────────

def _is_expired(created_str: str, hours: int = AUTO_CONFIRM_HOURS) -> bool:
    """Return True if the item was created more than `hours` ago."""
    if not created_str:
        return False
    try:
        created = datetime.fromisoformat(created_str)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - created > timedelta(hours=hours)
    except Exception:
        return False


def _auto_resolve_contradiction(fm: dict) -> list[str]:
    """Pick which node to keep based on completeness_score; fall back to newer."""
    node_a_id = fm.get("node_a_id", "")
    node_b_id = fm.get("node_b_id", "")

    def get_score(node_id: str) -> float:
        if not node_id:
            return 0.0
        try:
            r = qc._req("GET", f"collections/{qc.COLLECTION}/points/{node_id}", None)
            payload = r.get("result", {}).get("payload", {})
            score = float(payload.get("completeness_score", 0) or 0)
            if score == 0:
                # Fallback: newer created_at wins
                ts = payload.get("created_at", "") or ""
                score = datetime.fromisoformat(ts).timestamp() if ts else 0
            return score
        except Exception:
            return 0.0

    score_a = get_score(node_a_id)
    score_b = get_score(node_b_id)
    return ["保留A"] if score_a >= score_b else ["保留B"]


# ── Category handlers ────────────────────────────────────────────────────────

def _handle_contradiction(fm: dict, choices: list[str], inbox_file: Path) -> str:
    node_a_id = fm.get("node_a_id", "")
    node_b_id = fm.get("node_b_id", "")
    choice    = choices[0] if choices else ""

    if "保留 A" in choice or "保留A" in choice:
        if node_b_id:
            try:
                qc.patch_payload(int(node_b_id), {"status": "superseded",
                                                    "superseded_reason": "用户决策: 保留A"})
            except Exception as e:
                LOG.error(f"Patch B failed: {e}")
        return "已保留A，标记B为过时"

    elif "保留 B" in choice or "保留B" in choice:
        if node_a_id:
            try:
                qc.patch_payload(int(node_a_id), {"status": "superseded",
                                                    "superseded_reason": "用户决策: 保留B"})
            except Exception as e:
                LOG.error(f"Patch A failed: {e}")
        return "已保留B，标记A为过时"

    else:
        return f"用户决策已记录：{choice}"


def _handle_research_confirm(fm: dict, choices: list[str], inbox_file: Path) -> str:
    l0_ids_str = fm.get("l0_ids", "").strip("[]")
    l0_ids     = [x.strip() for x in l0_ids_str.split(",") if x.strip()]
    choice     = choices[0] if choices else ""

    if not choice:
        return "用户未勾选，跳过"

    domain = fm.get("domain", "")

    if "全部确认" in choice:
        activated = 0
        missing   = 0
        for id_str in l0_ids:
            try:
                pid = int(id_str)
                if not qc.point_exists(pid):
                    LOG.warning(f"Activate {id_str} skipped: point not found in Qdrant")
                    missing += 1
                    continue
                qc.patch_payload(pid, {"status": "pending"})
                activated += 1
            except Exception as e:
                LOG.error(f"Activate {id_str} failed: {e}")
        _sync_research_thread(domain, "confirmed")
        suffix = f"（{missing} 条已不存在，跳过）" if missing else ""
        return f"已确认 {activated} 个研究节点，加入待处理队列{suffix}"

    elif "全部拒绝" in choice:
        for id_str in l0_ids:
            try:
                pid = int(id_str)
                if not qc.point_exists(pid):
                    continue
                qc.patch_payload(pid, {"status": "rejected"})
            except Exception as e:
                LOG.error(f"Reject {id_str} failed: {e}")
        _sync_research_thread(domain, "rejected")
        return "已拒绝所有研究结果"

    else:
        _sync_research_thread(domain, "partial")
        return f"部分确认已记录：{choice}"


def _sync_research_thread(domain: str, new_status: str) -> None:
    """Update active_research_threads in context.json when a research is resolved."""
    if not domain:
        return
    try:
        ctx = sm.load_context()
        threads = ctx.get("active_research_threads", [])
        for t in threads:
            if t.get("domain") == domain and t.get("status") == "pending_review":
                t["status"] = new_status
        ctx["active_research_threads"] = threads
        sm.save_context(ctx)
    except Exception as e:
        LOG.warning(f"Failed to sync research thread for {domain}: {e}")


# ── Taxonomy review handler ──────────────────────────────────────────────────

def _handle_taxonomy_review(fm: dict, choices: list[str], inbox_file: Path) -> str:
    """Apply user's decision about a proposed new taxonomy category."""
    cat_name  = fm.get("new_category", "").strip()
    domain    = fm.get("domain", "").strip()
    choice    = choices[0] if choices else ""

    if not cat_name or not choice:
        return "无有效选择，跳过"

    # Load taxonomy
    try:
        tax = json.loads(TAXONOMY_FILE.read_text("utf-8")) if TAXONOMY_FILE.exists() else {"categories": {}}
    except Exception as e:
        LOG.warning(f"taxonomy_review: failed to load taxonomy: {e}")
        return f"读取taxonomy失败: {e}"

    cats = tax.setdefault("categories", {})
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    if "确认创建" in choice:
        if cat_name not in cats:
            cats[cat_name] = {
                "description": fm.get("description", f"涵盖{cat_name}相关领域"),
                "domains":     [domain] if domain else [],
                "auto_generated": True,
                "pending_review": False,
                "created_at":  now,
            }
        else:
            # Category already exists — just ensure domain is registered
            cats[cat_name]["pending_review"] = False
            if domain and domain not in cats[cat_name].get("domains", []):
                cats[cat_name].setdefault("domains", []).append(domain)
        result = f"大类「{cat_name}」已确认创建并写入taxonomy"

    elif "归入「其他」" in choice or "归入其他" in choice and "大类" not in choice:
        target = "其他"
        cats.setdefault(target, {"description": "暂未分类的领域", "domains": [], "auto_generated": False, "pending_review": False})
        if domain and domain not in cats[target].get("domains", []):
            cats[target].setdefault("domains", []).append(domain)
        # Remove the provisional new category if it exists and is still pending
        if cat_name in cats and cats[cat_name].get("pending_review"):
            del cats[cat_name]
        result = f"领域「{domain}」已归入「其他」，移除临时大类「{cat_name}」"

    else:
        # Unrecognised choice — log and skip taxonomy change
        return f"已记录用户选择（未执行分类变更）：{choice}"

    tax["last_updated"] = now
    try:
        TAXONOMY_FILE.write_text(json.dumps(tax, ensure_ascii=False, indent=2), "utf-8")
        LOG.info(f"taxonomy_review: {result}")
    except Exception as e:
        LOG.warning(f"taxonomy_review: failed to save taxonomy: {e}")
        return f"写入taxonomy失败: {e}"

    return result


# ── Main scanner ─────────────────────────────────────────────────────────────

def process_user_responses() -> dict:
    if not INBOX_DIR.exists():
        return {"processed": 0, "pending": 0}

    processed    = 0
    pending_count = 0

    for md_file in sorted(INBOX_DIR.glob("*.md")):
        try:
            text = md_file.read_text("utf-8")
        except Exception as e:
            LOG.warning(f"Cannot read {md_file.name}: {e}")
            continue

        fm       = _read_frontmatter(text)
        status   = fm.get("status", "pending")
        category = fm.get("category", "")

        # Already processed → skip
        if status == "processed":
            continue

        # Detect user intent: checkbox ticked OR status manually changed
        checked_choices  = _get_checked_choices(text)
        status_responded = status in ("approved", "rejected", "modified")

        if not checked_choices and not status_responded:
            # No user response — check for timeout auto-confirmation
            created_str = fm.get("created", "")
            if category in _DELETION_CATEGORIES:
                # Deletion operations always require explicit user confirmation
                pending_count += 1
                continue
            if _is_expired(created_str, hours=AUTO_CONFIRM_HOURS):
                if category == "contradiction":
                    checked_choices = _auto_resolve_contradiction(fm)
                    LOG.info(f"Auto-resolving contradiction {md_file.name}: {checked_choices} (24h timeout)")
                else:
                    checked_choices = ["全部确认"]
                    LOG.info(f"Auto-confirming {md_file.name} category={category} (24h timeout)")
            else:
                pending_count += 1
                continue

        # Determine effective choices
        if status_responded and not checked_choices:
            # User only changed status field — treat as generic approval
            checked_choices = [status]

        LOG.info(f"Processing inbox: {md_file.name} | choices={checked_choices}")

        result = "已处理"
        try:
            if category == "contradiction":
                result = _handle_contradiction(fm, checked_choices, md_file)
            elif category == "research_confirm":
                result = _handle_research_confirm(fm, checked_choices, md_file)
            elif category == "taxonomy_review":
                result = _handle_taxonomy_review(fm, checked_choices, md_file)
            else:
                result = f"类型 {category}，用户选择：{checked_choices[0] if checked_choices else '(无)'}"
        except Exception as e:
            LOG.error(f"Handle {md_file.name} failed: {e}")
            result = f"处理出错: {e}"

        _update_status_in_file(md_file, "processed")
        auto_tag = " [自动确认·24h超时]" if (not status_responded and not _get_checked_choices(text)) else ""
        _move_to_outbox(md_file, result + auto_tag)
        processed += 1
        LOG.info(f"  → {result}{auto_tag}")

    # Update context
    ctx = sm.load_context()
    ctx["pending_user_decisions"] = pending_count
    sm.save_context(ctx)

    if processed:
        LOG.info(f"Inbox: {processed} processed, {pending_count} still pending")

    return {"processed": processed, "pending": pending_count}
