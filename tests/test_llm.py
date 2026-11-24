"""The LLM client against a fake OpenAI-compatible server: retries on 429, caches, and strips code fences."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import override

import pytest

from summariser import llm

CALLS: list[dict[str, object]] = []


class FakeAPI(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        CALLS.append(body)
        if len(CALLS) == 1:  # first call is rate-limited
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        reply = {"choices": [{"message": {"content": "```markdown\n**Verdict: Low concern** — fine.\n```"}}], "usage": None}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode())

    @override
    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    httpd = HTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.setenv("LLM_MIN_INTERVAL", "0")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(llm, "RUNS", tmp_path)
    CALLS.clear()
    yield
    httpd.shutdown()


@pytest.mark.usefixtures("server")
def test_retry_cache_and_cleanup(tmp_path: Path) -> None:
    client = llm.Client("vtest")
    messages = [{"role": "user", "content": "hi"}]
    assert client.complete("SYN-1", messages) == "**Verdict: Low concern** — fine."
    assert len(CALLS) == 2  # one 429, one success
    saved = json.loads((tmp_path / "vtest" / "SYN-1.json").read_text())
    assert saved["model"] == "fake-model" and saved["messages"] == messages
    # Same prompt again: served from the cache, no new call.
    assert client.complete("SYN-1", messages) == "**Verdict: Low concern** — fine."
    assert len(CALLS) == 2
    # A changed prompt is a cache miss.
    client.complete("SYN-1", [{"role": "user", "content": "changed"}])
    assert len(CALLS) == 3
