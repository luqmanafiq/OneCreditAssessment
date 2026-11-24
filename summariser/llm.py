"""A small client for any OpenAI-compatible chat API (Gemini, Groq, OpenRouter, Mistral, ...), with a response cache.

Every response is saved to runs/<version>/<registrationNumber>.json together with the exact messages sent, so a
version can be re-scored, or re-rendered, without calling the model again. A cached response is reused only if the
model and messages are identical.

Configure with environment variables, or a .env file at the repo root (it's in .gitignore):
    LLM_BASE_URL   e.g. https://generativelanguage.googleapis.com/v1beta/openai   or   https://api.groq.com/openai/v1
    LLM_API_KEY
    LLM_MODEL      check your provider's current free models
    LLM_MIN_INTERVAL   seconds between calls, to stay under a free tier's per-minute limit (default 4)
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from summariser.facts import ROOT

RUNS = ROOT / "runs"


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def config() -> dict[str, str]:
    _load_dotenv()
    missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} (see summariser/llm.py).")
    return {
        "base_url": os.environ["LLM_BASE_URL"].rstrip("/"),
        "api_key": os.environ["LLM_API_KEY"],
        "model": os.environ["LLM_MODEL"],
        "min_interval": os.environ.get("LLM_MIN_INTERVAL", "4"),
    }


def _key(model: str, messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps([model, messages], sort_keys=True).encode()).hexdigest()[:16]


def _post(cfg: dict[str, str], messages: list[dict[str, str]]) -> dict[str, Any]:
    body = json.dumps({"model": cfg["model"], "messages": messages, "temperature": 0.2}).encode()
    request = urllib.request.Request(  # noqa: S310 (the URL comes from the user's own config)
        f"{cfg['base_url']}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise SystemExit(f"LLM call failed: HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e
            wait = float(e.headers.get("Retry-After") or 2 ** (attempt + 2))
            print(f"  HTTP {e.code}, retrying in {wait:.0f}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


class Client:
    def __init__(self, version: str, dry_run: bool = False) -> None:
        self.dir = RUNS / version
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.cfg = {"model": "dry-run"} if dry_run else config()
        self._last = 0.0

    def complete(self, registration_number: str, messages: list[dict[str, str]]) -> str:
        path = self.dir / f"{registration_number}.json"
        key = _key(self.cfg["model"], messages)
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("key") == key:
                return cached["text"]
        if self.dry_run:
            path.write_text(json.dumps({"key": key, "messages": messages, "text": ""}, indent=1), encoding="utf-8")
            return ""
        wait = float(self.cfg["min_interval"]) - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        started = time.monotonic()
        raw = _post(self.cfg, messages)
        self._last = time.monotonic()
        text = clean(raw["choices"][0]["message"]["content"] or "")
        record = {
            "key": key,
            "model": self.cfg["model"],
            "messages": messages,
            "text": text,
            "usage": raw.get("usage"),
            "seconds": round(self._last - started, 2),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return text


def clean(text: str) -> str:
    """Models sometimes wrap Markdown in a code fence or add reasoning tags; strip both."""
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.removesuffix("```").strip()
    return text
