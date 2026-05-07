"""Outbox archiver — rolls up processed ATLAS outbox files into hierarchical summaries.

Archive levels (all under outbox/archive/):
  daily   → 2026-05-06.md        one file per calendar day
  weekly  → 2026-W19.md          one file per ISO week
  monthly → 2026-05.md           one file per calendar month
  yearly  → 2026.md              one file per year

Only files whose embedded date is BEFORE today are ever touched.
Archive files themselves (in outbox/archive/) are never treated as loose files.

Entry point: run_archiver() → {"daily": N, "weekly": N, "monthly": N, "yearly": N}
"""

import logging
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

LOG = logging.getLogger("atlas.archiver")

VAULT       = Path("/Volumes/data/obsidian-vault")
OUTBOX_DIR  = VAULT / "agent-brain" / "outbox"
ARCHIVE_DIR = OUTBOX_DIR / "archive"

# Directories that must never be deleted even when empty
_PROTECTED_DIRS = {
    VAULT / "agent-brain" / "inbox",
    VAULT / "agent-brain" / "outbox",
    VAULT / "agent-brain" / "messages" / "inbox",
    VAULT / "agent-brain" / "messages" / "outbox",
    VAULT / ".git",
}


def cleanup_empty_dirs() -> int:
    """Remove empty directories from the vault, skipping protected system dirs.

    Walks bottom-up so nested empties are cleared in one pass.
    Returns the number of directories removed.
    """
    removed = 0
    for dirpath in sorted(VAULT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not dirpath.is_dir():
            continue
        # Skip protected dirs and anything inside .git
        if dirpath in _PROTECTED_DIRS:
            continue
        if ".git" in dirpath.parts:
            continue
        try:
            # is_empty: no files and no subdirs
            children = list(dirpath.iterdir())
            if not children:
                dirpath.rmdir()
                LOG.info(f"Removed empty dir: {dirpath.relative_to(VAULT)}")
                removed += 1
        except Exception as e:
            LOG.warning(f"Could not remove {dirpath}: {e}")
    return removed

# ── Filename / frontmatter helpers ────────────────────────────────────────────

_DATE_RE = re.compile(r"(\d{8})")


def _date_from_filename(name: str) -> date | None:
    """Extract the first YYYYMMDD embedded in a filename and return a date, or None."""
    m = _DATE_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _read_frontmatter(text: str) -> dict:
    """Parse YAML-ish frontmatter block between the first pair of '---' lines."""
    fm: dict = {}
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return fm
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm


def _extract_result(text: str) -> str:
    """Return the content after '**处理结果：**' on the last such line, or ''."""
    matches = re.findall(r"\*\*处理结果：\*\*\s*(.+)", text)
    return matches[-1].strip() if matches else ""


def _is_auto(result: str) -> bool:
    """True when the result was auto-confirmed (contains the 48h-timeout tag)."""
    return "[自动确认·48h超时]" in result


def _extract_time_hhmm(filename: str) -> str:
    """Return HH:MM from the HHMMSS part of the filename, e.g. '15:38' from '153803'."""
    # Filename pattern: type-domain-YYYYMMDD-HHMMSS.md
    m = re.search(r"\d{8}-(\d{2})(\d{2})\d{2}", filename)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return "??:??"


# ── Loose-file discovery ──────────────────────────────────────────────────────

def _loose_files_before_today() -> list[Path]:
    """Return outbox .md files that are NOT inside archive/ and predate today."""
    today = date.today()
    result = []
    for p in sorted(OUTBOX_DIR.glob("*.md")):
        if p.parent == ARCHIVE_DIR:
            continue
        d = _date_from_filename(p.name)
        if d is None:
            LOG.debug(f"No date in filename, skipping: {p.name}")
            continue
        if d < today:
            result.append(p)
    return result


def _group_by_date(files: list[Path]) -> dict[date, list[Path]]:
    groups: dict[date, list[Path]] = defaultdict(list)
    for p in files:
        d = _date_from_filename(p.name)
        if d:
            groups[d].append(p)
    return groups


# ── Daily archiver ────────────────────────────────────────────────────────────

def _archive_daily(day: date, files: list[Path]) -> Path:
    """Produce outbox/archive/YYYY-MM-DD.md from `files`.  Return the archive path."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / f"{day.isoformat()}.md"

    # Parse every file
    rows: list[dict] = []
    for p in sorted(files, key=lambda x: x.name):
        try:
            text = p.read_text("utf-8")
        except Exception as e:
            LOG.warning(f"Cannot read {p.name}: {e}")
            continue
        fm      = _read_frontmatter(text)
        result  = _extract_result(text)
        rows.append({
            "file":     p,
            "time":     _extract_time_hhmm(p.name),
            "category": fm.get("category", "其他"),
            "domain":   fm.get("domain", "—"),
            "result":   result,
            "auto":     _is_auto(result),
        })

    # Group by category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    auto_count = sum(1 for r in rows if r["auto"])
    user_count = len(rows) - auto_count

    lines: list[str] = []
    lines.append(f"# ATLAS 处理日志 · {day.isoformat()}")
    lines.append("")
    lines.append(f"> 共处理 {len(rows)} 项")
    lines.append("")

    # Known categories first, then catch-all
    known_order = ["research_confirm", "contradiction"]
    categories = known_order + [c for c in by_cat if c not in known_order]

    for cat in categories:
        if cat not in by_cat:
            continue
        cat_rows = by_cat[cat]
        label = cat if cat not in ("research_confirm", "contradiction", "其他") else {
            "research_confirm": "research_confirm",
            "contradiction":    "contradiction",
            "其他":             "其他",
        }.get(cat, cat)

        display_label = "其他" if cat == "其他" else label
        lines.append(f"## {display_label}（{len(cat_rows)}项）")
        lines.append("")
        lines.append("| 时间 | 域 | 结果 |")
        lines.append("|---|---|---|")
        for r in cat_rows:
            result_cell = r["result"]
            if r["auto"]:
                # Strip the raw tag and add a cleaner marker
                result_cell = result_cell.replace(" [自动确认·48h超时]", "") + " [自动]"
            lines.append(f"| {r['time']} | {r['domain']} | {result_cell} |")
        lines.append("")

    lines.append("---")
    lines.append(f"*自动确认: {auto_count}项 · 用户确认: {user_count}项*")
    lines.append("")

    content = "\n".join(lines)
    dest.write_text(content, "utf-8")
    LOG.info(f"Daily archive written: {dest.name} ({len(rows)} items)")

    # Remove the loose source files
    for r in rows:
        try:
            r["file"].unlink()
            LOG.debug(f"Deleted loose file: {r['file'].name}")
        except Exception as e:
            LOG.warning(f"Could not delete {r['file'].name}: {e}")

    return dest


# ── Weekly archiver ───────────────────────────────────────────────────────────

def _week_label(year: int, week: int) -> str:
    return f"{year}-W{week:02d}"


def _week_date_range(year: int, week: int) -> tuple[date, date]:
    """Return (monday, sunday) for the given ISO year/week."""
    monday = date.fromisocalendar(year, week, 1)
    sunday = monday + timedelta(days=6)
    return monday, sunday


def _archive_weekly(year: int, week: int) -> Path | None:
    """Roll up daily archive files for the given ISO week into a weekly summary."""
    monday, sunday = _week_date_range(year, week)
    label = _week_label(year, week)
    dest  = ARCHIVE_DIR / f"{label}.md"

    # Find daily archive files whose date falls in this week
    day_files: list[tuple[date, Path]] = []
    current = monday
    while current <= sunday:
        dp = ARCHIVE_DIR / f"{current.isoformat()}.md"
        if dp.exists():
            day_files.append((current, dp))
        current += timedelta(days=1)

    if not day_files:
        return None

    # Parse each daily summary for its stats line and table rows
    day_summaries: list[dict] = []
    total_items = 0
    for day, dp in day_files:
        try:
            text = dp.read_text("utf-8")
        except Exception as e:
            LOG.warning(f"Cannot read {dp.name}: {e}")
            continue

        # Extract total from "> 共处理 N 项"
        m = re.search(r"共处理\s+(\d+)\s+项", text)
        day_total = int(m.group(1)) if m else 0
        total_items += day_total

        # Extract per-category counts from "## category（N项）" headings
        cat_counts: list[str] = []
        for hm in re.finditer(r"^## (.+?)（(\d+)项）", text, re.MULTILINE):
            cat_counts.append(f"{hm.group(1)}×{hm.group(2)}")

        # Auto / user split from footer
        auto_m = re.search(r"自动确认:\s*(\d+)项\s*·\s*用户确认:\s*(\d+)项", text)
        auto_c = int(auto_m.group(1)) if auto_m else 0
        user_c = int(auto_m.group(2)) if auto_m else 0

        day_summaries.append({
            "day":        day,
            "total":      day_total,
            "cat_counts": cat_counts,
            "auto":       auto_c,
            "user":       user_c,
        })

    if not day_summaries:
        return None

    lines: list[str] = []
    lines.append(f"# ATLAS 周报 · {label}")
    lines.append("")
    lines.append(
        f"> {monday.isoformat()} ~ {sunday.isoformat()} · 共处理 {total_items} 项"
    )
    lines.append("")
    lines.append("## 每日汇总")
    lines.append("")
    for ds in day_summaries:
        cat_str = ", ".join(ds["cat_counts"]) if ds["cat_counts"] else "无"
        lines.append(f"- {ds['day'].isoformat()}: {ds['total']}项（{cat_str}）")
    lines.append("")

    total_auto = sum(ds["auto"] for ds in day_summaries)
    total_user = sum(ds["user"] for ds in day_summaries)
    lines.append("---")
    lines.append(f"*自动确认: {total_auto}项 · 用户确认: {total_user}项*")
    lines.append("")

    content = "\n".join(lines)
    dest.write_text(content, "utf-8")
    LOG.info(f"Weekly archive written: {dest.name} ({total_items} items)")

    # Remove daily files that were rolled up
    for _, dp in day_files:
        try:
            dp.unlink()
            LOG.debug(f"Deleted daily archive: {dp.name}")
        except Exception as e:
            LOG.warning(f"Could not delete {dp.name}: {e}")

    return dest


# ── Monthly archiver ──────────────────────────────────────────────────────────

def _archive_monthly(year: int, month: int) -> Path | None:
    """Roll up weekly archive files for the given month into a monthly summary."""
    label = f"{year}-{month:02d}"
    dest  = ARCHIVE_DIR / f"{label}.md"

    # Determine which ISO weeks overlap with this month
    # A week belongs here if its Monday falls in [year-month] OR its Friday does.
    # Simpler: collect all weeks that contain at least one day in this month.
    first_day = date(year, month, 1)
    # last day of month
    if month == 12:
        last_day = date(year, 12, 31)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)

    # Walk day-by-day and collect unique (iso_year, iso_week) pairs
    seen_weeks: set[tuple[int, int]] = set()
    current = first_day
    while current <= last_day:
        iso = current.isocalendar()
        seen_weeks.add((iso[0], iso[1]))
        current += timedelta(days=1)

    # Find week archive files that exist for those weeks
    week_files: list[tuple[int, int, Path]] = []
    for (wy, ww) in sorted(seen_weeks):
        wp = ARCHIVE_DIR / f"{_week_label(wy, ww)}.md"
        if wp.exists():
            week_files.append((wy, ww, wp))

    if not week_files:
        return None

    week_summaries: list[dict] = []
    total_items = 0
    for wy, ww, wp in week_files:
        try:
            text = wp.read_text("utf-8")
        except Exception as e:
            LOG.warning(f"Cannot read {wp.name}: {e}")
            continue

        m = re.search(r"共处理\s+(\d+)\s+项", text)
        wk_total = int(m.group(1)) if m else 0
        total_items += wk_total

        monday, sunday = _week_date_range(wy, ww)
        auto_m = re.search(r"自动确认:\s*(\d+)项\s*·\s*用户确认:\s*(\d+)项", text)
        auto_c = int(auto_m.group(1)) if auto_m else 0
        user_c = int(auto_m.group(2)) if auto_m else 0

        # Count distinct days mentioned in the daily section
        day_mentions = re.findall(r"- (\d{4}-\d{2}-\d{2}):", text)

        week_summaries.append({
            "label":  _week_label(wy, ww),
            "monday": monday,
            "sunday": sunday,
            "total":  wk_total,
            "days":   len(day_mentions),
            "auto":   auto_c,
            "user":   user_c,
        })

    if not week_summaries:
        return None

    lines: list[str] = []
    lines.append(f"# ATLAS 月报 · {label}")
    lines.append("")
    lines.append(f"> {first_day.isoformat()} ~ {last_day.isoformat()} · 共处理 {total_items} 项")
    lines.append("")
    lines.append("## 每周汇总")
    lines.append("")
    for ws in week_summaries:
        lines.append(
            f"- {ws['label']} ({ws['monday'].isoformat()} ~ {ws['sunday'].isoformat()}): "
            f"{ws['total']}项，活跃 {ws['days']} 天"
        )
    lines.append("")

    total_auto = sum(ws["auto"] for ws in week_summaries)
    total_user = sum(ws["user"] for ws in week_summaries)
    lines.append("---")
    lines.append(f"*自动确认: {total_auto}项 · 用户确认: {total_user}项*")
    lines.append("")

    content = "\n".join(lines)
    dest.write_text(content, "utf-8")
    LOG.info(f"Monthly archive written: {dest.name} ({total_items} items)")

    # Remove weekly files that were rolled up
    for _, _, wp in week_files:
        try:
            wp.unlink()
            LOG.debug(f"Deleted weekly archive: {wp.name}")
        except Exception as e:
            LOG.warning(f"Could not delete {wp.name}: {e}")

    return dest


# ── Yearly archiver ───────────────────────────────────────────────────────────

def _archive_yearly(year: int) -> Path | None:
    """Roll up monthly archive files for the given year into an annual summary."""
    dest = ARCHIVE_DIR / f"{year}.md"

    month_files: list[tuple[int, Path]] = []
    for month in range(1, 13):
        label = f"{year}-{month:02d}"
        mp    = ARCHIVE_DIR / f"{label}.md"
        if mp.exists():
            month_files.append((month, mp))

    if not month_files:
        return None

    month_summaries: list[dict] = []
    total_items = 0
    for month, mp in month_files:
        try:
            text = mp.read_text("utf-8")
        except Exception as e:
            LOG.warning(f"Cannot read {mp.name}: {e}")
            continue

        m = re.search(r"共处理\s+(\d+)\s+项", text)
        mo_total = int(m.group(1)) if m else 0
        total_items += mo_total

        auto_m = re.search(r"自动确认:\s*(\d+)项\s*·\s*用户确认:\s*(\d+)项", text)
        auto_c = int(auto_m.group(1)) if auto_m else 0
        user_c = int(auto_m.group(2)) if auto_m else 0

        month_summaries.append({
            "month": month,
            "label": f"{year}-{month:02d}",
            "total": mo_total,
            "auto":  auto_c,
            "user":  user_c,
        })

    if not month_summaries:
        return None

    lines: list[str] = []
    lines.append(f"# ATLAS 年报 · {year}")
    lines.append("")
    lines.append(f"> 共处理 {total_items} 项")
    lines.append("")
    lines.append("## 每月汇总")
    lines.append("")
    lines.append("| 月份 | 处理量 | 自动确认 | 用户确认 |")
    lines.append("|---|---|---|---|")
    for ms in month_summaries:
        lines.append(
            f"| {ms['label']} | {ms['total']} | {ms['auto']} | {ms['user']} |"
        )
    lines.append("")

    total_auto = sum(ms["auto"] for ms in month_summaries)
    total_user = sum(ms["user"] for ms in month_summaries)
    lines.append("---")
    lines.append(f"*自动确认: {total_auto}项 · 用户确认: {total_user}项*")
    lines.append("")

    content = "\n".join(lines)
    dest.write_text(content, "utf-8")
    LOG.info(f"Yearly archive written: {dest.name} ({total_items} items)")

    # Remove monthly files that were rolled up
    for _, mp in month_files:
        try:
            mp.unlink()
            LOG.debug(f"Deleted monthly archive: {mp.name}")
        except Exception as e:
            LOG.warning(f"Could not delete {mp.name}: {e}")

    return dest


# ── Weekly / monthly / yearly discovery helpers ───────────────────────────────

def _daily_archives_before_this_week() -> dict[tuple[int, int], list[date]]:
    """Return {(iso_year, iso_week): [dates]} for daily archives in previous weeks."""
    today      = date.today()
    this_iso   = today.isocalendar()
    this_yw    = (this_iso[0], this_iso[1])

    result: dict[tuple[int, int], list[date]] = defaultdict(list)
    pattern = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.md$")
    if not ARCHIVE_DIR.exists():
        return result

    for p in ARCHIVE_DIR.glob("????-??-??.md"):
        m = pattern.match(p.name)
        if not m:
            continue
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        iso = d.isocalendar()
        yw  = (iso[0], iso[1])
        if yw < this_yw:
            result[yw].append(d)

    return result


def _weekly_archives_before_this_month() -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Return {(year, month): [(iso_year, iso_week)]} for weekly archives in previous months."""
    today = date.today()
    this_ym = (today.year, today.month)

    result: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    pattern = re.compile(r"^(\d{4})-W(\d{2})\.md$")
    if not ARCHIVE_DIR.exists():
        return result

    for p in ARCHIVE_DIR.glob("????-W??.md"):
        m = pattern.match(p.name)
        if not m:
            continue
        wy, ww = int(m.group(1)), int(m.group(2))
        # Use the Friday of this week to assign it to a month
        try:
            friday = date.fromisocalendar(wy, ww, 5)
        except ValueError:
            continue
        ym = (friday.year, friday.month)
        if ym < this_ym:
            result[ym].append((wy, ww))

    return result


def _monthly_archives_before_this_year() -> dict[int, list[int]]:
    """Return {year: [months]} for monthly archives in previous years."""
    today     = date.today()
    this_year = today.year

    result: dict[int, list[int]] = defaultdict(list)
    pattern = re.compile(r"^(\d{4})-(\d{2})\.md$")
    if not ARCHIVE_DIR.exists():
        return result

    for p in ARCHIVE_DIR.glob("????-??.md"):
        m = pattern.match(p.name)
        if not m:
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        if y < this_year:
            result[y].append(mo)

    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def run_archiver() -> dict[str, int]:
    """Run all archiving passes and return counts of archives created at each level."""
    counts = {"daily": 0, "weekly": 0, "monthly": 0, "yearly": 0}

    # ── Pass 1: loose files → daily archives ─────────────────────────────────
    loose = _loose_files_before_today()
    if not loose:
        LOG.info("Archiver: no loose outbox files to process")
    else:
        LOG.info(f"Archiver: found {len(loose)} loose file(s) to archive")

    by_date = _group_by_date(loose)
    for day in sorted(by_date):
        try:
            _archive_daily(day, by_date[day])
            counts["daily"] += 1
        except Exception as e:
            LOG.error(f"Daily archive failed for {day}: {e}")

    # ── Pass 2: daily archives → weekly archives ──────────────────────────────
    daily_by_week = _daily_archives_before_this_week()
    for (wy, ww) in sorted(daily_by_week):
        try:
            result = _archive_weekly(wy, ww)
            if result:
                counts["weekly"] += 1
        except Exception as e:
            LOG.error(f"Weekly archive failed for {_week_label(wy, ww)}: {e}")

    # ── Pass 3: weekly archives → monthly archives ────────────────────────────
    weekly_by_month = _weekly_archives_before_this_month()
    for (year, month) in sorted(weekly_by_month):
        try:
            result = _archive_monthly(year, month)
            if result:
                counts["monthly"] += 1
        except Exception as e:
            LOG.error(f"Monthly archive failed for {year}-{month:02d}: {e}")

    # ── Pass 4: monthly archives → yearly archives ────────────────────────────
    monthly_by_year = _monthly_archives_before_this_year()
    for year in sorted(monthly_by_year):
        try:
            result = _archive_yearly(year)
            if result:
                counts["yearly"] += 1
        except Exception as e:
            LOG.error(f"Yearly archive failed for {year}: {e}")

    counts["empty_dirs_removed"] = cleanup_empty_dirs()

    LOG.info(
        f"Archiver complete: daily={counts['daily']} weekly={counts['weekly']} "
        f"monthly={counts['monthly']} yearly={counts['yearly']} "
        f"empty_dirs={counts['empty_dirs_removed']}"
    )
    return counts


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_archiver()
    print(result)
