"""Write / sync Obsidian files for L4 meta-principle records."""
import logging
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc

LOG  = logging.getLogger("atlas.l4_obsidian")
VAULT = Path("/Volumes/data/obsidian-vault")


def _write_l4_file(point_id: int, payload: dict) -> str | None:
    l4_dir = VAULT / "L4"
    l4_dir.mkdir(parents=True, exist_ok=True)

    title = payload.get("title") or payload.get("topic") or str(point_id)
    slug  = title
    for ch in r'/\:*?"<>|':
        slug = slug.replace(ch, "-")
    slug = slug[:60].strip()

    now       = datetime.now(timezone.utc)
    content_body = payload.get("content") or payload.get("core_principle") or ""
    patterns  = payload.get("key_patterns") or []
    pat_lines = "\n".join(f"- {p}" for p in patterns)
    domains   = payload.get("applicable_to") or []
    dom_str   = "、".join(domains)
    meta      = payload.get("meta_insight") or ""
    decision  = payload.get("decision_framework") or ""
    tags      = payload.get("tags") or []
    tag_str   = ", ".join(f'"{t}"' for t in tags)
    source_ids = payload.get("source_ids") or []

    content = f"""---
level: L4
domain: META
title: {title}
tags: [{tag_str}]
created: {payload.get('created_at') or now.isoformat()}
synced_at: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}
---

# {title}

> L4 元规律 · 跨域共性模式

## 核心内容

{content_body}

{"## 关键规律" + chr(10) + pat_lines if pat_lines else ""}

{"## 底层洞见" + chr(10) + meta if meta else ""}

{"## 决策框架" + chr(10) + decision if decision else ""}

{"## 可迁移领域" + chr(10) + dom_str if dom_str else ""}

## 蒸馏来源

{"共 " + str(len(source_ids)) + " 条下层知识" if source_ids else "来源记录暂缺"}

---
*L4 元规律 · 由 ATLAS 知识代理维护*
"""

    rel_path = f"L4/{slug}.md"
    (VAULT / rel_path).write_text(content.strip() + "\n", "utf-8")
    return rel_path


def sync_l4_obsidian() -> dict:
    """Write/update Obsidian files for all L4 records."""
    l4_pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 4}},
            {"key": "status", "match": {"value": "active"}},
        ]
    }, limit=50)

    written  = 0
    skipped  = 0

    for pt in l4_pts:
        pid = pt["id"]
        pay = pt["payload"]

        existing_path = pay.get("obsidian_path")
        if existing_path:
            full = VAULT / existing_path
            if full.exists():
                skipped += 1
                continue  # already has a good file

        try:
            rel_path = _write_l4_file(pid, pay)
            if rel_path:
                qc.patch_payload(pid, {"obsidian_path": rel_path})
                written += 1
                LOG.info(f"L4 file written: {rel_path}")
        except Exception as e:
            LOG.error(f"L4 Obsidian write failed for {pid}: {e}")

    LOG.info(f"L4 Obsidian sync: {written} written, {skipped} already exist")
    return {"written": written, "skipped": skipped}
