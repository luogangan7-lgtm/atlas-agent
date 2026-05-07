"""Web search via DuckDuckGo (ddgs) — no API key, no registration required.

Provides:
  text_search(query, max_results)       → list[dict]  general search
  news_search(query, max_results)       → list[dict]  news search
  fetch_page(url, max_chars)            → str          page text extraction
  search_and_fetch(query, ...)          → dict         search + optional page fetch

In-memory cache (10 min TTL) and rate limiting (1.5s between requests) are
built in to avoid triggering DuckDuckGo's IP-based throttling.
"""
import logging
import time
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser

LOG = logging.getLogger("atlas.web_search")

_CACHE: dict[tuple, tuple] = {}     # (kind, query, n) → (timestamp, results)
_CACHE_TTL   = 600                  # 10 min
_RATE_LIMIT  = 1.5                  # seconds between outbound searches
_last_call   = 0.0


class _TextExtractor(HTMLParser):
    """Minimal HTML → plain-text stripper using stdlib only."""
    _SKIP = {"script", "style", "nav", "footer", "header", "aside", "noscript"}

    def __init__(self):
        super().__init__()
        self._depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag, _attrs):
        if tag.lower() in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self._SKIP and self._depth > 0:
            self._depth -= 1

    def handle_data(self, data):
        if self._depth == 0:
            t = data.strip()
            if len(t) > 20:
                self._chunks.append(t)

    def result(self) -> str:
        return "\n".join(self._chunks)


def _throttle() -> None:
    global _last_call
    wait = _RATE_LIMIT - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def text_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web. Returns [{title, url, snippet}, ...]."""
    key = ("text", query, max_results)
    if key in _CACHE:
        ts, cached = _CACHE[key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    _throttle()
    try:
        from ddgs import DDGS
        raw = DDGS().text(query, max_results=max_results) or []
        results = [{"title": r.get("title", ""), "url": r.get("href", ""),
                    "snippet": r.get("body", "")} for r in raw]
        _CACHE[key] = (time.time(), results)
        LOG.info(f"web:text '{query[:50]}' → {len(results)} hits")
        return results
    except Exception as e:
        LOG.warning(f"web:text search failed: {e}")
        return []


def news_search(query: str, max_results: int = 5) -> list[dict]:
    """Search recent news. Returns [{title, url, snippet, date, source}, ...]."""
    key = ("news", query, max_results)
    if key in _CACHE:
        ts, cached = _CACHE[key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    _throttle()
    try:
        from ddgs import DDGS
        raw = DDGS().news(query, max_results=max_results) or []
        results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                    "snippet": r.get("body", ""), "date": r.get("date", ""),
                    "source": r.get("source", "")} for r in raw]
        _CACHE[key] = (time.time(), results)
        LOG.info(f"web:news '{query[:50]}' → {len(results)} hits")
        return results
    except Exception as e:
        LOG.warning(f"web:news search failed: {e}")
        return []


def fetch_page(url: str, max_chars: int = 3000) -> str:
    """Fetch URL and return extracted plain text (stdlib only, no external deps)."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            ct = resp.headers.get("Content-Type", "")
            if "html" not in ct and "plain" not in ct:
                return ""
            raw = resp.read(max_chars * 8).decode("utf-8", errors="replace")
    except Exception as e:
        LOG.debug(f"fetch_page {url}: {e}")
        return ""

    parser = _TextExtractor()
    try:
        parser.feed(raw)
        return parser.result()[:max_chars]
    except Exception:
        return raw[:max_chars]


def search_and_fetch(
    query: str,
    max_results: int = 5,
    fetch_top_n: int = 0,
    max_chars_per_page: int = 2000,
) -> dict:
    """Search + optionally fetch top-N pages for richer content.

    Returns:
      {"query", "results": [{title, url, snippet, content?}], "fetched", "timestamp"}
    """
    results = text_search(query, max_results=max_results)

    for i, r in enumerate(results[:fetch_top_n]):
        r["content"] = fetch_page(r["url"], max_chars=max_chars_per_page)
        if i < fetch_top_n - 1:
            time.sleep(0.5)

    return {
        "query":     query,
        "results":   results,
        "fetched":   min(fetch_top_n, len(results)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
