import json
import os
import re
import time
import urllib.request
from typing import Any


DEEPSEEK_URL   = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
TIMEOUT        = 120


def _api_key() -> str:
    return os.environ.get("DEEPSEEK_API_KEY", "")


def deepseek_generate(system: str, user: str, max_tokens: int = 800, retries: int = 2) -> str | None:
    key = _api_key()
    if not key:
        return None

    body = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"{DEEPSEEK_URL}/v1/chat/completions",
                data=body,
                method="POST",
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {key}",
                },
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise
    return None


def _sanitize(text: str) -> str:
    """Remove Unicode replacement characters that indicate encoding corruption."""
    return text.replace("�", "")


def parse_json_block(text: str) -> Any:
    """Extract and parse the first JSON object or array from LLM output."""
    if not text:
        return None
    # Strip markdown code fences
    cleaned = re.sub(r"```[a-z]*\n?", "", text).replace("```", "").strip()
    # Try full parse
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = None

    if data is None:
        for pat in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
            m = re.search(pat, cleaned)
            if m:
                try:
                    data = json.loads(m.group())
                    break
                except json.JSONDecodeError:
                    pass

    if data is None:
        return None

    # Recursively strip replacement characters from all string values
    return _clean_obj(data)


def _clean_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return _sanitize(obj)
    if isinstance(obj, list):
        return [_clean_obj(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _clean_obj(v) for k, v in obj.items()}
    return obj
