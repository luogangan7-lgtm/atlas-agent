"""Minimal MCP-compatible HTTP server for ATLAS knowledge queries.

Runs as a background thread on port 8766.
Exposes endpoints that follow the MCP JSON-RPC pattern:
  POST /  — JSON-RPC 2.0 dispatcher
  GET  /health — liveness check

Supported methods:
  memory_search   — semantic search over ATLAS knowledge base
  get_knowledge   — retrieve a specific knowledge item by ID or topic
  add_knowledge   — inject raw content as L0 pending for processing
  list_domains    — list all knowledge domains with stats
  get_health      — knowledge base health summary
"""
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

from storage import qdrant_client as qc
from utils.embed import get_embedding

LOG = logging.getLogger("atlas.mcp_server")
PORT = 8766
_server_thread: threading.Thread | None = None


# ─── JSON-RPC helpers ────────────────────────────────────────────────────────

def _ok(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _err(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


# ─── Method handlers ─────────────────────────────────────────────────────────

def _handle_memory_search(params: dict) -> dict:
    query  = params.get("query", "")
    limit  = min(int(params.get("limit", 5)), 20)
    level  = params.get("level")           # optional: 1/2/3/4
    domain = params.get("domain")          # optional: filter by domain

    if not query:
        return {"results": [], "error": "query required"}

    vector = get_embedding(query)
    if not vector:
        return {"results": [], "error": "embedding failed"}

    must = []
    if level is not None:
        must.append({"key": "level", "match": {"value": int(level)}})
    if domain:
        must.append({"key": "domain", "match": {"value": domain}})
    must.append({"key": "status", "match": {"value": "active"}})

    results = qc.search(
        vector=vector,
        filter_body={"must": must} if must else None,
        limit=limit,
        score_threshold=0.5,
    )

    return {
        "results": [
            {
                "id":       r["id"],
                "score":    round(r.get("score", 0), 4),
                "level":    r["payload"].get("level"),
                "domain":   r["payload"].get("domain", ""),
                "title":    r["payload"].get("title") or r["payload"].get("topic", ""),
                "summary":  r["payload"].get("summary") or r["payload"].get("content", "")[:200],
                "obsidian_path": r["payload"].get("obsidian_path", ""),
            }
            for r in results
        ],
        "total": len(results),
    }


def _handle_get_knowledge(params: dict) -> dict:
    point_id = params.get("id")
    topic    = params.get("topic", "")

    if point_id:
        # Direct ID lookup
        pts = qc.scroll({"must": [{"key": "id", "match": {"value": int(point_id)}}]}, limit=1)
        if pts:
            return {"item": pts[0]["payload"], "id": pts[0]["id"]}
        return {"error": "not found"}

    if topic:
        # Topic search
        vector = get_embedding(topic)
        if vector:
            results = qc.search(vector, limit=1, score_threshold=0.7)
            if results:
                return {"item": results[0]["payload"], "id": results[0]["id"], "score": results[0]["score"]}
        return {"error": "not found"}

    return {"error": "id or topic required"}


def _handle_add_knowledge(params: dict) -> dict:
    content      = params.get("content", "").strip()
    domain       = params.get("domain", "未分类")
    content_type = params.get("content_type", "note")
    source       = params.get("source", "external-mcp")

    if not content:
        return {"error": "content required"}
    if len(content) < 10:
        return {"error": "content too short"}

    h   = hashlib.md5((content + source).encode()).hexdigest()
    pid = int(h[:16], 16) % (2 ** 53)

    try:
        qc._req("PUT", f"collections/{qc.COLLECTION}/points?wait=true", {
            "points": [{
                "id":      pid,
                "vector":  [0.0] * 1024,
                "payload": {
                    "level":        0,
                    "status":       "pending",
                    "content":      content,
                    "domain":       domain,
                    "content_type": content_type,
                    "source":       source,
                    "created_at":   datetime.now(timezone.utc).isoformat(),
                },
            }]
        })
        return {"success": True, "id": pid, "message": "Added to L0 queue for processing"}
    except Exception as e:
        return {"error": str(e)}


def _handle_list_domains(params: dict) -> dict:
    from collections import defaultdict
    pts = qc.scroll({
        "must": [
            {"key": "level",  "match": {"value": 1}},
            {"key": "status", "match": {"value": "active"}},
        ],
        "must_not": [
            {"key": "record_type", "match": {"value": "entity"}},
        ],
    }, limit=2000)

    counts: dict[str, int] = defaultdict(int)
    for pt in pts:
        d = pt["payload"].get("domain") or "未分类"
        counts[d] += 1

    domains = sorted(
        [{"domain": d, "l1_count": c} for d, c in counts.items()],
        key=lambda x: x["l1_count"],
        reverse=True,
    )
    return {"domains": domains, "total": len(domains)}


def _handle_get_health(params: dict) -> dict:
    from core.health import assess_health
    try:
        report = assess_health()
        return {
            "score":            round(report.score, 3),
            "grade":            report.grade,
            "l1_total":         report.l1_total,
            "l2_total":         report.l2_total,
            "l3_total":         report.l3_total,
            "l4_total":         report.l4_total,
            "coverage_score":   round(report.coverage_score, 3),
            "avg_completeness": round(report.avg_l1_completeness, 3),
            "gap_domains":      report.gap_domains[:10],
        }
    except Exception as e:
        return {"error": str(e)}


def _handle_unified_search(params: dict) -> dict:
    """Search both ATLAS knowledge base and the web (DuckDuckGo), merge results.

    params:
      query       str   — search query (required)
      max_kb      int   — max knowledge-base results (default 5)
      max_web     int   — max web results (default 5)
      mode        str   — "all" | "kb_only" | "web_only" (default "all")
      news        bool  — also run a news search (default false)
      fetch_top   int   — fetch page content for top-N web results (default 0)
    """
    query    = params.get("query", "").strip()
    max_kb   = min(int(params.get("max_kb", 5)), 20)
    max_web  = min(int(params.get("max_web", 5)), 10)
    mode     = params.get("mode", "all")
    want_news = bool(params.get("news", False))
    fetch_top = int(params.get("fetch_top", 0))

    if not query:
        return {"error": "query required"}

    kb_results  = []
    web_results = []
    news_results = []

    # ── Knowledge-base search ─────────────────────────────────────────────────
    if mode in ("all", "kb_only"):
        try:
            vector = get_embedding(query)
            if vector:
                hits = qc.search(
                    vector=vector,
                    filter_body={"must": [{"key": "status", "match": {"value": "active"}}]},
                    limit=max_kb,
                    score_threshold=0.45,
                )
                kb_results = [
                    {
                        "source":   "atlas",
                        "level":    h["payload"].get("level"),
                        "domain":   h["payload"].get("domain", ""),
                        "title":    h["payload"].get("title") or h["payload"].get("topic", ""),
                        "summary":  (h["payload"].get("summary") or h["payload"].get("content", ""))[:300],
                        "score":    round(h.get("score", 0), 4),
                        "obsidian_path": h["payload"].get("obsidian_path", ""),
                    }
                    for h in hits
                ]
        except Exception as e:
            LOG.warning(f"unified_search: KB error: {e}")

    # ── Web search ────────────────────────────────────────────────────────────
    if mode in ("all", "web_only"):
        try:
            from utils.web_search import search_and_fetch, news_search as _news_search
            web_data    = search_and_fetch(query, max_results=max_web, fetch_top_n=fetch_top)
            web_results = [{"source": "web", **r} for r in web_data["results"]]

            if want_news:
                news_raw  = _news_search(query, max_results=5)
                news_results = [{"source": "news", **r} for r in news_raw]
        except Exception as e:
            LOG.warning(f"unified_search: web error: {e}")

    return {
        "query":        query,
        "kb_results":   kb_results,
        "web_results":  web_results,
        "news_results": news_results,
        "total_kb":     len(kb_results),
        "total_web":    len(web_results),
        "total_news":   len(news_results),
    }


_METHODS = {
    "memory_search":   _handle_memory_search,
    "get_knowledge":   _handle_get_knowledge,
    "add_knowledge":   _handle_add_knowledge,
    "list_domains":    _handle_list_domains,
    "get_health":      _handle_get_health,
    "unified_search":  _handle_unified_search,
}


# ─── HTTP request handler ────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        LOG.debug(f"HTTP {self.address_string()} {fmt % args}")

    def _send_json(self, code: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "agent": "atlas-knowledge-agent", "port": PORT})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length)

        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, _err(None, -32700, "Parse error"))
            return

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}

        handler = _METHODS.get(method)
        if not handler:
            self._send_json(200, _err(req_id, -32601, f"Method not found: {method}"))
            return

        try:
            result = handler(params)
            self._send_json(200, _ok(req_id, result))
        except Exception as e:
            LOG.error(f"Method {method} error: {e}", exc_info=True)
            self._send_json(200, _err(req_id, -32603, str(e)))


# ─── Server lifecycle ─────────────────────────────────────────────────────────

def start_server() -> None:
    global _server_thread

    def _run():
        server = HTTPServer(("127.0.0.1", PORT), Handler)
        LOG.info(f"ATLAS MCP server listening on http://127.0.0.1:{PORT}")
        server.serve_forever()

    _server_thread = threading.Thread(target=_run, daemon=True, name="atlas-mcp-server")
    _server_thread.start()
