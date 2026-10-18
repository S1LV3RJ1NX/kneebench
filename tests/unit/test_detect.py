"""Auto-detection over real HTTP against a canned local server."""

from __future__ import annotations

from kneebench.detect.server import detect_server
from tests.builders import ENDPOINT_FIXTURE
from tests.conftest import SENTINEL_KEY, FakeServer, json_route, text_route


def test_detects_vllm_server_from_recorded_routes(fake_server: type[FakeServer]) -> None:
    server = fake_server(
        {
            "/version": json_route({"version": "0.30.1"}),
            "/v1/models": text_route((ENDPOINT_FIXTURE / "models.json").read_text()),
            "/metrics": text_route((ENDPOINT_FIXTURE / "metrics.prom").read_text()),
            "/health": text_route(""),
        }
    )
    snap = detect_server(f"{server.url}/v1", SENTINEL_KEY)

    assert snap.base_url == server.url
    assert snap.vllm_version == "0.30.1"
    model = snap.model()
    assert model is not None
    assert model.id == "served-model"
    assert model.max_model_len == 8192
    assert model.hf_id == "unsloth/Meta-Llama-3.1-8B-Instruct"
    assert snap.cache is not None
    assert snap.cache.kv_cache_size_tokens == 532_768
    assert snap.network is not None
    assert snap.network.http_rtt_ms is not None
    assert snap.network.tls_handshake_ms is None  # plain http
    assert set(server.seen_auth) == {f"Bearer {SENTINEL_KEY}"}


def test_detects_generic_openai_server(fake_server: type[FakeServer]) -> None:
    server = fake_server({"/v1/models": json_route({"data": [{"id": "my-model"}]})})
    snap = detect_server(server.url, None, measure_network=False)

    assert snap.reachable
    assert snap.vllm_version is None
    assert not snap.metrics_available
    assert snap.cache is None
    model = snap.model("my-model")
    assert model is not None
    assert model.hf_id is None
    assert snap.model("missing") is None
