import json
import urllib.request
from typing import Any

QDRANT     = "http://127.0.0.1:6333"
COLLECTION = "atlas_memories_v2"


def _req(method: str, path: str, body: Any = None, timeout: int = 30) -> dict:
    url = f"{QDRANT}/{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def scroll(filter_body: dict, limit: int = 100, with_vector: bool = False) -> list[dict]:
    points = []
    offset = None
    while True:
        body: dict = {
            "limit": limit,
            "with_payload": True,
            "with_vector": with_vector,
            "filter": filter_body,
        }
        if offset:
            body["offset"] = offset
        r = _req("POST", f"collections/{COLLECTION}/points/scroll", body)
        batch = r.get("result", {}).get("points", [])
        points.extend(batch)
        offset = r.get("result", {}).get("next_page_offset")
        if not batch or offset is None:
            break
    return points


def search(vector: list[float], filter_body: dict | None = None, limit: int = 10, score_threshold: float = 0.0) -> list[dict]:
    body: dict = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
        "score_threshold": score_threshold,
    }
    if filter_body:
        body["filter"] = filter_body
    r = _req("POST", f"collections/{COLLECTION}/points/search", body)
    return r.get("result", [])


def point_exists(point_id: str | int) -> bool:
    """Return True if the point exists in the collection."""
    try:
        r = _req("GET", f"collections/{COLLECTION}/points/{point_id}")
        return r.get("result") is not None
    except Exception:
        return False


def patch_payload(point_id: str | int, payload: dict) -> None:
    _req("POST", f"collections/{COLLECTION}/points/payload?wait=true", {
        "payload": payload,
        "points": [point_id],
    })


def upsert_point(point_id: str | int, vector: list[float], payload: dict) -> None:
    _req("PUT", f"collections/{COLLECTION}/points?wait=true", {
        "points": [{"id": point_id, "vector": vector, "payload": payload}]
    })


def get_collection_info() -> dict:
    r = _req("GET", f"collections/{COLLECTION}")
    return r.get("result", {})


def count(filter_body: dict) -> int:
    r = _req("POST", f"collections/{COLLECTION}/points/count", {"filter": filter_body})
    return r.get("result", {}).get("count", 0)
