from collections.abc import Callable

import pytest

from kneebench import doctor
from kneebench.detect.endpoint import ProbeResult
from kneebench.doctor import Status, exit_code, run_doctor

ProbeFn = Callable[..., ProbeResult]


def _fake_probe(responses: dict[str, ProbeResult]) -> ProbeFn:
    def fake(
        base: str, path: str, api_key: str | None = None, expect_json: bool = True
    ) -> ProbeResult:
        return responses[path]

    return fake


def test_offline_doctor_passes_on_a_supported_install() -> None:
    checks = run_doctor(offline=True, url=None, api_key=None, tokenizer=None)
    statuses = {c.name: c.status for c in checks}
    assert statuses["Python"] is Status.OK
    assert statuses["AIPerf"] is Status.OK
    assert exit_code(checks) == 0


@pytest.mark.parametrize(
    ("version", "status"),
    [
        ((3, 10, 9), Status.FAIL),
        ((3, 11, 0), Status.OK),
        ((3, 13, 5), Status.OK),
        ((3, 14, 0), Status.FAIL),
    ],
)
def test_python_range(version: tuple[int, int, int], status: Status) -> None:
    check = doctor.check_python(version)
    assert check.status is status
    assert check.fix if status is Status.FAIL else check.fix is None


def test_vllm_endpoint_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor,
        "probe",
        _fake_probe(
            {
                "/health": ProbeResult("/health", ok=True, status=200, elapsed_ms=12.0),
                "/version": ProbeResult(
                    "/version", ok=True, status=200, body={"version": "0.29.0"}
                ),
                "/v1/models": ProbeResult(
                    "/v1/models",
                    ok=True,
                    status=200,
                    body={
                        "data": [
                            {
                                "id": "served-name",
                                "max_model_len": 32768,
                                "root": "/models/models--acme--chat-7b/snapshots/1",
                            }
                        ]
                    },
                ),
                "/metrics": ProbeResult(
                    "/metrics",
                    ok=True,
                    status=200,
                    text='vllm:cache_config_info{block_size="16"} 1.0\n',
                ),
            }
        ),
    )
    checks, hf_id = doctor.check_endpoint("http://server:8000", None)
    assert hf_id == "acme/chat-7b"
    assert all(c.status is Status.OK for c in checks)
    assert any("32768" in c.detail for c in checks)


def test_non_vllm_endpoint_degrades_without_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor,
        "probe",
        _fake_probe(
            {
                "/health": ProbeResult("/health", ok=True, status=200, elapsed_ms=3.0),
                "/version": ProbeResult("/version", ok=False, status=404),
                "/v1/models": ProbeResult(
                    "/v1/models", ok=True, status=200, body={"data": [{"id": "mock"}]}
                ),
                "/metrics": ProbeResult("/metrics", ok=False, status=404),
            }
        ),
    )
    checks, hf_id = doctor.check_endpoint("http://server:8000", None)
    by_name = {c.name: c for c in checks}
    assert hf_id is None
    assert by_name["GET /version"].status is Status.WARN
    assert by_name["GET /metrics"].status is Status.WARN
    assert "not visible" in (by_name["GET /metrics"].fix or "")
    assert exit_code(checks) == 0


def test_unreachable_endpoint_fails_with_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor,
        "probe",
        _fake_probe(
            {
                "/health": ProbeResult("/health", ok=False, error="ConnectError: refused"),
            }
        ),
    )
    checks, _ = doctor.check_endpoint("http://nowhere:1", None)
    assert checks[0].status is Status.FAIL
    assert "VPN" in (checks[0].fix or "")


def test_bad_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor,
        "probe",
        _fake_probe(
            {
                "/health": ProbeResult("/health", ok=True, status=200, elapsed_ms=3.0),
                "/version": ProbeResult(
                    "/version", ok=True, status=200, body={"version": "0.30.0"}
                ),
                "/v1/models": ProbeResult("/v1/models", ok=False, status=401),
                "/metrics": ProbeResult("/metrics", ok=False, status=401),
            }
        ),
    )
    checks, _ = doctor.check_endpoint("http://server:8000", "wrong")
    models = next(c for c in checks if c.name == "GET /v1/models")
    assert models.status is Status.FAIL
    assert "KNEEBENCH_API_KEY" in (models.fix or "")
