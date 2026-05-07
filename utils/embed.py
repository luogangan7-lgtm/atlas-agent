import json
import urllib.request

OLLAMA_URL  = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
SAFE_CHARS  = 6000


def get_embedding(text: str, model: str = EMBED_MODEL) -> list[float] | None:
    if not text:
        return None
    text = text[:SAFE_CHARS]
    body = json.dumps({"model": model, "prompt": text}).encode()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data.get("embedding")
    except Exception:
        return None
