"""File-based knowledge intake: scan agent-brain/intake/ for .md/.txt files → L0.

Drop any markdown or text file into:
  /Volumes/data/obsidian-vault/agent-brain/intake/

The agent picks it up within 15 minutes, creates L0 records (status=pending),
then organize.py promotes them to L1 in the following cycle.

Filename convention (optional, all fields inferred if omitted):
  [domain]__[content_type]__<any title>.md
  e.g.  营销策略__course__克亚第9讲.md
        战略__principle__增长飞轮.txt

If the file contains YAML front-matter, those fields override filename inference:
  ---
  domain: 营销策略
  content_type: course
  ---
"""
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from storage import qdrant_client as qc
from utils.llm import deepseek_generate, parse_json_block
from utils.embed import get_embedding

LOG = logging.getLogger("atlas.intake")

VAULT       = Path("/Volumes/data/obsidian-vault")
INTAKE_DIR  = VAULT / "agent-brain" / "intake"
DONE_DIR    = INTAKE_DIR / "processed"
CONTENT_TYPES = {
    "concept", "argument", "procedure", "fact", "principle",
    "course", "book", "note", "article", "video_script", "sop",
}
_FNAME_RE = re.compile(r"^(?P<domain>[^_]+?)__(?P<ctype>[^_]+?)__.*$")


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML front-matter (simple key: value only) and return (meta, body)."""
    meta: dict = {}
    if not text.startswith("---"):
        return meta, text
    end = text.find("\n---", 3)
    if end == -1:
        return meta, text
    fm_block = text[3:end].strip()
    body = text[end + 4:].strip()
    for line in fm_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body


def _infer_meta(path: Path) -> dict:
    """Infer domain/content_type from filename pattern."""
    stem = path.stem
    m = _FNAME_RE.match(stem)
    if m:
        ctype = m.group("ctype").lower()
        return {
            "domain":       m.group("domain"),
            "content_type": ctype if ctype in CONTENT_TYPES else "note",
        }
    return {}


def _infer_domain_llm(title: str, snippet: str) -> str:
    """Use LLM to infer domain when not provided (fallback, cheap call)."""
    system = "你是知识分类专家。严格输出中文JSON，不加解释。"
    user   = f"文件标题：{title}\n内容片段：{snippet[:300]}\n\n输出该知识属于哪个领域（2-6汉字）：{{\"domain\": \"领域名\"}}"
    try:
        raw = deepseek_generate(system, user, max_tokens=800)
        res = parse_json_block(raw) if raw else None
        d = (res or {}).get("domain", "").strip()
        return d if d and d != "未分类" else "未分类"
    except Exception:
        return "未分类"


def _write_l0(path: Path, meta: dict, body: str) -> bool:
    """Write a single L0 record to Qdrant. Returns True on success."""
    title        = meta.get("title") or path.stem
    domain       = meta.get("domain") or ""
    content_type = meta.get("content_type") or "note"

    if content_type not in CONTENT_TYPES:
        content_type = "note"

    if not domain:
        domain = _infer_domain_llm(title, body)

    # Embed the title for dedup / search
    vector = get_embedding(f"{title}\n{body[:400]}")
    if not vector:
        LOG.warning(f"intake: failed to embed '{title}', skipping")
        return False

    # Stable ID based on file content hash to avoid duplicates
    import hashlib
    h = int(hashlib.md5((title + body[:200]).encode()).hexdigest()[:16], 16) % (2**53)

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "level":        0,
        "status":       "pending",
        "title":        title,
        "source":       title,
        "domain":       domain,
        "content_type": content_type,
        "content":      body,
        "raw_content":  body,
        "created_at":   now,
        "intake_file":  path.name,
    }
    try:
        qc.upsert_point(h, vector, payload)
        return True
    except Exception as e:
        LOG.warning(f"intake: upsert failed for '{title}': {e}")
        return False


def run_intake() -> dict:
    """Scan intake/ folder and ingest new files as L0 records."""
    INTAKE_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        f for f in INTAKE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in (".md", ".txt") and not f.name.startswith(".")
    ]
    if not files:
        return {"ingested": 0}

    LOG.info(f"intake: found {len(files)} files")
    ingested = 0

    for path in files:
        try:
            raw = path.read_text("utf-8", errors="replace")
            fm, body = _parse_frontmatter(raw)

            # Merge: frontmatter > filename inference > defaults
            file_meta = _infer_meta(path)
            meta = {**file_meta, **fm}
            meta.setdefault("title", path.stem)

            if not body.strip():
                LOG.info(f"intake: '{path.name}' is empty, skipping")
                path.rename(DONE_DIR / path.name)
                continue

            ok = _write_l0(path, meta, body)
            if ok:
                LOG.info(f"intake: '{path.name}' → L0 (domain={meta.get('domain')!r})")
                ingested += 1
            # Move to processed regardless (avoids retry loop on persistent failures)
            prefix = "ok__" if ok else "err__"
            path.rename(DONE_DIR / (prefix + path.name))

        except Exception as e:
            LOG.warning(f"intake: error processing '{path.name}': {e}")

    if ingested:
        LOG.info(f"intake done: {ingested} files → L0")
    return {"ingested": ingested}
