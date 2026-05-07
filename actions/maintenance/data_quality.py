"""Data Quality — unified maintenance action.

Runs every fast cycle, handling four sub-tasks in priority order:

  1. clean_empty_dirs()        — remove empty Obsidian folders
  2. fix_garbled()             — scan/repair U+FFFD in titles (batch)
  3. reclassify_unclassified() — assign domain to 未分类 nodes (batch)
  4. assign_categories()       — assign category (大类) to classified nodes (batch)

Taxonomy is stored in a living JSON file at STATE_DIR/taxonomy.json.
The file is bootstrapped on first run via LLM clustering over all known domains,
then evolves automatically as new domains appear.

Compound cross-domain names (e.g. "营销策略×情感学") are handled by extracting
the primary domain (before the first "×") for category lookup.
"""
import json
import logging
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc
from utils.llm import deepseek_generate, parse_json_block

LOG   = logging.getLogger("atlas.data_quality")
VAULT = Path("/Volumes/data/obsidian-vault")

TAXONOMY_FILE = VAULT / "agent-brain" / "state" / "taxonomy.json"
INBOX_DIR     = VAULT / "agent-brain" / "inbox"

_PROTECTED_DIRS = {
    VAULT / "agent-brain" / "inbox",
    VAULT / "agent-brain" / "outbox",
    VAULT / "agent-brain" / "intake",
    VAULT / "agent-brain" / "intake" / "processed",
    VAULT / "agent-brain" / "messages" / "inbox",
    VAULT / "agent-brain" / "messages" / "outbox",
    VAULT / "agent-brain" / "state",
    VAULT / ".git",
}

# Per-cycle batch sizes — small to avoid LLM overload
GARBLED_BATCH      = 20
UNCLASSIFIED_BATCH = 20   # increased: 289 backlog, 20/cycle clears in ~15 cycles
CATEGORY_BATCH     = 30   # increased: 807 backlog


# ── Taxonomy I/O ──────────────────────────────────────────────────────────────

def _load_taxonomy() -> dict:
    if TAXONOMY_FILE.exists():
        try:
            return json.loads(TAXONOMY_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {}


def _save_taxonomy(tax: dict) -> None:
    TAXONOMY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tax["last_updated"] = datetime.now(timezone.utc).isoformat()
    TAXONOMY_FILE.write_text(json.dumps(tax, ensure_ascii=False, indent=2), "utf-8")


def _primary_domain(domain: str) -> str:
    """For compound cross-domain names like 'A×B', return primary domain A."""
    return domain.split("×")[0].strip()


def _domain_to_category(domain: str, tax: dict) -> str | None:
    """Look up which category a domain belongs to (checks primary domain too)."""
    cats = tax.get("categories", {})
    # Direct match
    for cat_name, cat_data in cats.items():
        if domain in cat_data.get("domains", []):
            return cat_name
    # Primary domain match (for compound domains)
    primary = _primary_domain(domain)
    if primary != domain:
        for cat_name, cat_data in cats.items():
            if primary in cat_data.get("domains", []):
                return cat_name
    return None


# ── Taxonomy bootstrap ────────────────────────────────────────────────────────

def _get_all_active_domains() -> dict[str, int]:
    """Return {domain: count} for all active nodes, excluding 未分类."""
    pts = qc.scroll({
        "must": [{"key": "status", "match": {"value": "active"}}],
        "must_not": [{"key": "domain", "match": {"value": "未分类"}}],
    }, limit=5000)
    counts: Counter = Counter()
    for p in pts:
        d = (p["payload"].get("domain") or "").strip()
        if d:
            counts[d] += 1
    return dict(counts)


def bootstrap_taxonomy() -> dict:
    """Generate initial taxonomy.json from all current domains via LLM."""
    LOG.info("Taxonomy bootstrap: generating from current domains")
    domain_counts = _get_all_active_domains()

    # Only pass leaf domains (no compound) for clustering
    leaf_domains = sorted(
        {_primary_domain(d) for d in domain_counts if "×" not in d},
        key=lambda d: -domain_counts.get(d, 0)
    )

    # Pass top-50 leaf domains; remaining will be discovered via assign_missing_categories
    domain_list = "\n".join(f"- {d} ({domain_counts.get(d, 0)} 条)" for d in leaf_domains[:50])

    system = "你是知识体系架构师。严格输出中文JSON，不加解释。"
    user = (
        f"以下是知识库中的领域列表：\n{domain_list}\n\n"
        "请归为 5-8 个大类，要求：\n"
        "1. 大类名称 2-5 汉字，能涵盖旗下所有领域\n"
        "2. 大类要有前瞻性，能容纳未来出现的相似新领域\n"
        "3. 保留「其他」大类兜底\n"
        '输出格式：{"categories":{"大类名":{"description":"涵盖范围及未来扩展方向","domains":["领域1","领域2"]}}}'
    )

    try:
        raw = deepseek_generate(system, user, max_tokens=2000)
        result = parse_json_block(raw)
        if not result or "categories" not in result:
            raise ValueError("Invalid taxonomy response")
    except Exception as e:
        LOG.error(f"Bootstrap LLM failed: {e}, using minimal fallback")
        result = {"categories": {"其他": {"description": "暂未分类的领域", "domains": []}}}

    now = datetime.now(timezone.utc).isoformat()
    tax = {
        "version": 1,
        "created_at": now,
        "last_updated": now,
        "evolution_rules": {
            "split_threshold": 20,
            "merge_similarity_threshold": 0.85,
            "min_domains_per_category": 2,
            "new_domain_confidence_threshold": 0.65,
        },
        "categories": {},
    }

    for cat_name, cat_data in result["categories"].items():
        tax["categories"][cat_name] = {
            "description": cat_data.get("description", ""),
            "domains": cat_data.get("domains", []),
            "auto_generated": False,
            "created_at": now,
            "pending_review": False,
        }

    _save_taxonomy(tax)
    LOG.info(f"Taxonomy bootstrapped: {len(tax['categories'])} categories, "
             f"{sum(len(v['domains']) for v in tax['categories'].values())} domains mapped")
    return tax


def _discover_category(domain: str, tax: dict) -> str:
    """Map a new domain to the best-matching existing category (create new only as last resort)."""
    cats = tax.get("categories", {})
    # Include ALL categories (including pending_review) so LLM sees the full picture
    cat_summaries = "\n".join(
        f"- {name}: {data.get('description', '')} (现有: {', '.join(data.get('domains', [])[:6])})"
        for name, data in cats.items()
    )

    system = "你是知识分类专家。严格输出中文JSON，不加解释。"
    user = f"""新出现的知识领域：「{domain}」

现有大类（必须优先归入）：
{cat_summaries}

请判断该领域最适合归入哪个现有大类。只有当没有任何大类匹配（置信度 < 0.55）时，才建议新大类。

输出：{{"category": "大类名", "confidence": 0.8, "is_new": false}}
若确实无法归类：{{"category": "新大类名", "confidence": 0.4, "is_new": true, "new_description": "涵盖范围"}}"""

    try:
        raw = deepseek_generate(system, user, max_tokens=800)
        res = parse_json_block(raw)
        if not res or not res.get("category"):
            return "其他"

        cat_name   = res["category"].strip()
        confidence = float(res.get("confidence", 0))
        is_new     = bool(res.get("is_new", False))

        # Hard rule: only create new category when truly no match (confidence < 0.55) AND is_new=True
        if is_new and cat_name not in cats and confidence < 0.55:
            now      = datetime.now(timezone.utc).isoformat()
            new_desc = res.get("new_description", f"自动发现的新领域大类，包含「{domain}」等")
            tax["categories"][cat_name] = {
                "description": new_desc,
                "domains":     [domain],
                "auto_generated": True,
                "created_at":  now,
                "pending_review": True,
            }
            LOG.info(f"Taxonomy: new category '{cat_name}' for '{domain}' (conf={confidence:.2f})")
            _notify_new_category(cat_name, domain, new_desc, confidence)
        else:
            # Map to existing category (even if LLM said is_new — trust confidence over flag)
            target = cat_name if cat_name in cats else "其他"
            if domain not in tax["categories"][target]["domains"]:
                tax["categories"][target]["domains"].append(domain)
            cat_name = target

        return cat_name

    except Exception as e:
        LOG.warning(f"discover_category failed for '{domain}': {e}")
        return "其他"


def _notify_new_category(cat_name: str, domain: str, description: str, confidence: float) -> None:
    """Write inbox notification for low-confidence new category discoveries."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    now  = datetime.now(timezone.utc)
    date = now.strftime("%Y%m%d-%H%M%S")
    content = f"""---
type: user_decision_required
category: taxonomy_review
priority: low
created: {now.isoformat()}
status: pending
new_category: {cat_name}
domain: {domain}
confidence: {confidence:.2f}
---

# 发现新知识大类：{cat_name}

## 背景
知识库中出现了新领域「{domain}」，无法以较高置信度归入现有大类，
系统建议创建新大类「{cat_name}」。

**建议描述：** {description}
**分类置信度：** {confidence:.0%}

## 请你决定

- [ ] 确认创建新大类「{cat_name}」
- [ ] 归入其他已有大类（在下方注明）：
- [ ] 暂时归入「其他」

---
*由 ATLAS 数据质量模块自动发现 · {now.strftime("%Y-%m-%d %H:%M")} UTC*
"""
    fname = f"taxonomy-{cat_name}-{date}.md"
    (INBOX_DIR / fname).write_text(content, "utf-8")


# ── Sub-task 1: Empty directories ─────────────────────────────────────────────

_EMPTY_SENTINELS = {".gitkeep", ".DS_Store"}


def clean_empty_dirs() -> int:
    """Remove empty directories in the vault (bottom-up, skip protected).

    Treats directories that contain only .gitkeep / .DS_Store as empty.
    """
    removed = 0
    for dirpath in sorted(VAULT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not dirpath.is_dir():
            continue
        if dirpath == VAULT:
            continue
        if any(dirpath == p or p in dirpath.parents for p in _PROTECTED_DIRS):
            continue
        if any(part.startswith(".") for part in dirpath.parts):
            continue
        try:
            children = list(dirpath.iterdir())
            # Consider effectively empty if only sentinel files remain
            real_children = [c for c in children if c.name not in _EMPTY_SENTINELS]
            if not real_children:
                for sentinel in children:
                    sentinel.unlink(missing_ok=True)
                dirpath.rmdir()
                LOG.info(f"Removed empty dir: {dirpath.relative_to(VAULT)}")
                removed += 1
        except Exception:
            pass
    return removed


# ── Sub-task 2: Garbled titles ────────────────────────────────────────────────

def fix_garbled(batch: int = GARBLED_BATCH) -> int:
    """Scan active nodes for U+FFFD in title and attempt LLM repair."""
    pts = qc.scroll({
        "must": [{"key": "status", "match": {"value": "active"}}]
    }, limit=5000)

    bad = [p for p in pts if "�" in (p["payload"].get("title") or "")]
    if not bad:
        return 0

    LOG.info(f"fix_garbled: {len(bad)} corrupted titles found, repairing batch={batch}")
    fixed = 0

    for p in bad[:batch]:
        pid     = p["id"]
        payload = p["payload"]
        title   = payload.get("title", "")
        summary = (payload.get("summary") or payload.get("content") or "")[:200]
        tags    = ", ".join(payload.get("tags") or [])
        domain  = payload.get("domain", "")

        system = "你是文本修复专家。根据上下文推断损坏的标题原文。严格输出中文JSON。"
        user = f"""以下知识节点的标题含有乱码字符（用「?」表示）：

损坏标题：{title.replace(chr(0xFFFD), "?")}
摘要：{summary}
标签：{tags}
领域：{domain}

请推断正确的标题。输出：{{"title": "正确标题", "confidence": 0.9}}"""

        try:
            raw = deepseek_generate(system, user, max_tokens=800)
            res = parse_json_block(raw)
            if res and res.get("title") and "�" not in res["title"]:
                new_title = res["title"].strip()
                qc.patch_payload(pid, {"title": new_title})
                LOG.info(f"  Fixed L{payload.get('level')} {pid}: '{title}' → '{new_title}'")
                fixed += 1
            else:
                # Fallback: strip replacement chars
                clean = title.replace("�", "")
                if clean and clean != title:
                    qc.patch_payload(pid, {"title": clean})
                    LOG.info(f"  Stripped L{payload.get('level')} {pid}: '{title}' → '{clean}'")
                    fixed += 1
        except Exception as e:
            LOG.warning(f"  fix_garbled failed for {pid}: {e}")

    return fixed


# ── Sub-task 3: Reclassify 未分类 ─────────────────────────────────────────────

def reclassify_unclassified(batch: int = UNCLASSIFIED_BATCH, tax: dict | None = None) -> dict:
    """Assign domain (and category) to nodes stuck in 未分类."""
    if tax is None:
        tax = _load_taxonomy()

    pts = qc.scroll({
        "must": [
            {"key": "status", "match": {"value": "active"}},
            {"key": "domain", "match": {"value": "未分类"}},
        ]
    }, limit=500)

    if not pts:
        return {"reclassified": 0, "remaining": 0}

    LOG.info(f"reclassify_unclassified: {len(pts)} nodes, processing batch={batch}")

    # Build candidate domain list from taxonomy
    known_domains: list[str] = []
    for cat_data in tax.get("categories", {}).values():
        known_domains.extend(cat_data.get("domains", []))
    domain_list = "\n".join(f"- {d}" for d in sorted(set(known_domains))[:50])

    reclassified = 0

    for p in pts[:batch]:
        pid     = p["id"]
        payload = p["payload"]
        level   = payload.get("level", 1)
        title   = (payload.get("title") or payload.get("topic") or "").strip()
        summary = (payload.get("summary") or payload.get("content") or "")[:200]
        tags    = ", ".join(payload.get("tags") or [])

        if not title:
            continue

        system = "你是知识分类专家。严格输出中文JSON，不加解释。"
        user = f"""为以下 L{level} 知识节点分配领域（domain）。

标题：{title}
摘要：{summary}
标签：{tags}

现有领域候选（优先从中选择）：
{domain_list}

若无合适的现有领域，可提出新领域（2-6汉字）。不得返回「未分类」。

输出：{{"domain": "领域名"}}"""

        try:
            raw = deepseek_generate(system, user, max_tokens=800)
            res = parse_json_block(raw) if raw else None
            if not res or not res.get("domain"):
                continue
            new_domain = res["domain"].strip()
            if not new_domain or new_domain == "未分类":
                continue

            # Find / discover category
            cat = _domain_to_category(new_domain, tax)
            if not cat:
                cat = _discover_category(new_domain, tax)
                _save_taxonomy(tax)

            # Move Obsidian file if it exists in 未分类/
            patch: dict = {"domain": new_domain, "category": cat}
            old_path = payload.get("obsidian_path", "")
            if old_path and "未分类" in old_path:
                new_path = _move_obsidian_file(pid, old_path, level, new_domain)
                if new_path:
                    patch["obsidian_path"] = new_path

            qc.patch_payload(pid, patch)
            LOG.info(f"  L{level} {pid} '{title[:35]}' → domain='{new_domain}' cat='{cat}'")
            reclassified += 1

        except Exception as e:
            LOG.warning(f"  reclassify failed for {pid}: {e}")

    return {"reclassified": reclassified, "remaining": len(pts) - reclassified}


# ── Sub-task 4: Assign missing categories ─────────────────────────────────────

def assign_missing_categories(batch: int = CATEGORY_BATCH, tax: dict | None = None) -> int:
    """For nodes that have a domain but no category field, assign category."""
    if tax is None:
        tax = _load_taxonomy()

    pts = qc.scroll({
        "must": [
            {"key": "status",   "match": {"value": "active"}},
            {"is_empty":        {"key": "category"}},
        ],
        "must_not": [
            {"key": "domain",   "match": {"value": "未分类"}},
        ],
    }, limit=500)

    # Filter manually: no category field or empty category
    need_cat = [p for p in pts if not (p["payload"].get("category") or "").strip()]
    if not need_cat:
        return 0

    LOG.debug(f"assign_missing_categories: {len(need_cat)} nodes need category, batch={batch}")
    assigned  = 0
    tax_dirty = False

    for p in need_cat[:batch]:
        pid    = p["id"]
        domain = (p["payload"].get("domain") or "").strip()
        if not domain:
            continue

        cat = _domain_to_category(domain, tax)
        if not cat:
            cat = _discover_category(domain, tax)
            tax_dirty = True

        try:
            qc.patch_payload(pid, {"category": cat})
            assigned += 1
        except Exception as e:
            LOG.warning(f"  assign_category failed for {pid}: {e}")

    if tax_dirty:
        _save_taxonomy(tax)

    if assigned:
        LOG.info(f"assign_missing_categories: {assigned} nodes updated")
    return assigned


# ── Taxonomy evolution ────────────────────────────────────────────────────────

def evolve_taxonomy(tax: dict) -> bool:
    """Detect over-large categories that should split, or near-empty ones to merge.
    Returns True if taxonomy was modified."""
    rules    = tax.get("evolution_rules", {})
    split_at = rules.get("split_threshold", 20)
    min_size = rules.get("min_domains_per_category", 2)
    changed  = False

    cats = tax.get("categories", {})
    for cat_name, cat_data in list(cats.items()):
        domains = cat_data.get("domains", [])

        # Too large → ask LLM to propose split
        if len(domains) >= split_at and not cat_data.get("pending_review"):
            LOG.info(f"evolve_taxonomy: '{cat_name}' has {len(domains)} domains, suggesting split")
            _suggest_category_split(cat_name, domains, tax)
            changed = True

        # Too small → flag for potential merge
        if len(domains) < min_size and cat_data.get("auto_generated") and not cat_data.get("pending_review"):
            LOG.info(f"evolve_taxonomy: '{cat_name}' has only {len(domains)} domains, flagging")
            tax["categories"][cat_name]["pending_review"] = True
            changed = True

    return changed


def _suggest_category_split(cat_name: str, domains: list[str], tax: dict) -> None:
    domain_list = "\n".join(f"- {d}" for d in domains[:30])
    system = "你是知识体系架构师。严格输出中文JSON，不加解释。"
    user = f"""大类「{cat_name}」已包含 {len(domains)} 个领域，建议拆分为 2-3 个子大类。

当前领域：
{domain_list}

请建议拆分方案。输出：
{{
  "split_into": [
    {{"name": "新大类1", "description": "...", "domains": ["领域A", "领域B"]}},
    {{"name": "新大类2", "description": "...", "domains": ["领域C", "领域D"]}}
  ]
}}"""

    try:
        raw = deepseek_generate(system, user, max_tokens=800)
        res = parse_json_block(raw)
        if not res or not res.get("split_into"):
            return

        # Write inbox notification (user decides whether to accept split)
        INBOX_DIR.mkdir(parents=True, exist_ok=True)
        now  = datetime.now(timezone.utc)
        date = now.strftime("%Y%m%d-%H%M%S")
        split_lines = "\n".join(
            f"### {s['name']}\n{s.get('description','')}\n领域：{', '.join(s.get('domains',[]))}"
            for s in res["split_into"]
        )
        content = f"""---
type: user_decision_required
category: taxonomy_review
priority: low
created: {now.isoformat()}
status: pending
category_to_split: {cat_name}
---

# 建议拆分大类：{cat_name}

当前大类包含 {len(domains)} 个领域，建议拆分：

{split_lines}

## 请你决定

- [ ] 接受拆分方案
- [ ] 保持现状，不拆分
- [ ] 手动调整（在下方说明）：

---
*由 ATLAS 分类演化模块自动建议 · {now.strftime("%Y-%m-%d %H:%M")} UTC*
"""
        fname = f"taxonomy-split-{cat_name}-{date}.md"
        (INBOX_DIR / fname).write_text(content, "utf-8")
        tax["categories"][cat_name]["pending_review"] = True
        LOG.info(f"  Split suggestion written to inbox for '{cat_name}'")
    except Exception as e:
        LOG.warning(f"_suggest_category_split failed for '{cat_name}': {e}")


# ── Obsidian file relocation ──────────────────────────────────────────────────

def _move_obsidian_file(point_id: int, old_path_str: str, level: int, new_domain: str) -> str | None:
    from storage.obsidian_writer import get_category_for_domain
    old_abs  = VAULT / old_path_str
    if not old_abs.exists():
        return None
    category = get_category_for_domain(new_domain)
    new_dir  = VAULT / f"L{level}" / category
    new_dir.mkdir(parents=True, exist_ok=True)
    new_abs  = new_dir / old_abs.name
    if new_abs.exists():
        new_abs = new_dir / f"{old_abs.stem}-{str(point_id)[-6:]}{old_abs.suffix}"
    try:
        shutil.move(str(old_abs), str(new_abs))
        return str(new_abs.relative_to(VAULT))
    except Exception as e:
        LOG.warning(f"File move failed {old_abs} → {new_abs}: {e}")
        return None


def consolidate_obsidian_folders() -> dict:
    """One-time migration: move files from domain-based folders to category-based folders.

    Scans L1/, L2/, L3/ for any folder that is NOT a canonical category name,
    moves its files into the correct category folder, updates Qdrant obsidian_path.
    """
    from storage.obsidian_writer import get_category_for_domain
    tax = _load_taxonomy()
    canonical_cats = set(tax.get("categories", {}).keys())

    moved = 0
    for level in (1, 2, 3):
        level_dir = VAULT / f"L{level}"
        if not level_dir.exists():
            continue
        for folder in list(level_dir.iterdir()):
            if not folder.is_dir() or folder.name in canonical_cats:
                continue  # already a category folder or special dir
            # This is an old domain folder — find its category
            domain   = folder.name
            category = get_category_for_domain(domain)
            dst_dir  = level_dir / category
            dst_dir.mkdir(parents=True, exist_ok=True)

            for md_file in list(folder.glob("*.md")):
                dst = dst_dir / md_file.name
                if dst.exists():
                    dst = dst_dir / f"{md_file.stem}-{domain[:6]}{md_file.suffix}"
                try:
                    shutil.move(str(md_file), str(dst))
                    old_rel = str(md_file.relative_to(VAULT))
                    new_rel = str(dst.relative_to(VAULT))
                    # Update Qdrant records that reference the old path
                    pts = qc.scroll({
                        "must": [{"key": "obsidian_path", "match": {"value": old_rel}}]
                    }, limit=10)
                    for pt in pts:
                        qc.patch_payload(pt["id"], {"obsidian_path": new_rel})
                    moved += 1
                except Exception as e:
                    LOG.warning(f"consolidate: move failed {md_file}: {e}")

            # Remove now-empty domain folder
            try:
                if not any(folder.iterdir()):
                    folder.rmdir()
            except Exception:
                pass

    if moved:
        LOG.info(f"consolidate_obsidian_folders: {moved} files migrated to category folders")
    return {"files_consolidated": moved}


# ── Master entry point ────────────────────────────────────────────────────────

def purge_superseded(level: int = 3) -> int:
    """Delete superseded records for a given level. Safe to run periodically."""
    deleted = qc.delete_by_filter({
        "must": [
            {"key": "level",  "match": {"value": level}},
            {"key": "status", "match": {"value": "superseded"}},
        ]
    })
    if deleted:
        LOG.info(f"purge_superseded: deleted {deleted} superseded L{level} records")
    return deleted


def run_data_quality(evolve: bool = False) -> dict:
    """Run all data quality sub-tasks. Call from main cycle."""
    results: dict = {}

    # Ensure taxonomy exists
    tax = _load_taxonomy()
    if not tax:
        tax = bootstrap_taxonomy()
    if not tax:
        LOG.warning("data_quality: no taxonomy available, skipping category tasks")
        tax = {}

    # 0. Consolidate domain folders → category folders (fast file moves)
    c = consolidate_obsidian_folders()
    results["files_consolidated"] = c["files_consolidated"]

    # 0b. Purge superseded L3 records (no cap — always clean all)
    purged = purge_superseded(level=3)
    results["superseded_l3_purged"] = purged

    # 1. Empty directories (fast, no LLM)
    removed = clean_empty_dirs()
    results["empty_dirs_removed"] = removed

    # 2. Garbled titles (LLM, small batch)
    fixed = fix_garbled(batch=GARBLED_BATCH)
    results["garbled_fixed"] = fixed

    # 3. Reclassify 未分类 (LLM, small batch)
    rc = reclassify_unclassified(batch=UNCLASSIFIED_BATCH, tax=tax)
    results["unclassified_reclassified"] = rc["reclassified"]
    results["unclassified_remaining"]    = rc["remaining"]

    # 4. Assign missing categories (mostly local lookup, occasional LLM)
    assigned = assign_missing_categories(batch=CATEGORY_BATCH, tax=tax)
    results["categories_assigned"] = assigned

    # 5. Taxonomy evolution (only every N cycles, or when explicitly requested)
    if evolve:
        changed = evolve_taxonomy(tax)
        if changed:
            _save_taxonomy(tax)
        results["taxonomy_evolved"] = changed

    LOG.info(
        f"data_quality: dirs_removed={removed} garbled_fixed={fixed} "
        f"unclassified_reclassified={rc['reclassified']}(remaining={rc['remaining']}) "
        f"categories_assigned={assigned}"
    )
    return results
