from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

SENTINEL_KEY = "sk-kneebench-sentinel-7f3a9c"

Route = tuple[int, str, str]  # status, content type, body


class FakeServer:
    """A tiny HTTP server with canned GET routes, for endpoint-probe tests."""

    def __init__(self, routes: dict[str, Route]) -> None:
        self.routes = routes
        self.seen_auth: list[str | None] = []
        handler = self._handler()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host!s}:{port}"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                server.seen_auth.append(self.headers.get("Authorization"))
                status, ctype, body = server.routes.get(self.path, (404, "text/plain", "nope"))
                payload = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler


def json_route(body: object, status: int = 200) -> Route:
    return status, "application/json", json.dumps(body)


def text_route(body: str, status: int = 200) -> Route:
    return status, "text/plain", body


@pytest.fixture
def fake_server() -> Iterator[type[FakeServer]]:
    started: list[FakeServer] = []

    class Factory(FakeServer):
        def __init__(self, routes: dict[str, Route]) -> None:
            super().__init__(routes)
            self.thread.start()
            started.append(self)

    yield Factory
    for s in started:
        s.httpd.shutdown()
        s.httpd.server_close()
