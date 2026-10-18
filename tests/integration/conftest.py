"""A local guidellm mock server, run in its own uvx environment (it can't share ours)."""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from collections.abc import Iterator

import httpx
import pytest

GUIDELLM_PIN = "0.8.1"
MOCK_MODEL = "mock-llm"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="session")
def mock_server_url() -> Iterator[str]:
    uvx = shutil.which("uvx")
    if uvx is None:
        pytest.skip("uvx not found; install uv to run integration tests")
    port = _free_port()
    # No --processor: guidellm's mock server crashes when given one.
    proc = subprocess.Popen(
        [
            uvx,
            "--from",
            f"guidellm=={GUIDELLM_PIN}",
            "guidellm",
            "mock-server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model",
            MOCK_MODEL,
            "--ttft-ms",
            "20",
            "--itl-ms",
            "2",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 300
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"guidellm mock-server exited with {proc.returncode}")
            try:
                if httpx.get(f"{url}/v1/models", timeout=2).is_success:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            pytest.fail("guidellm mock-server did not start within 300 s")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
