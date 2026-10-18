from pathlib import Path

from kneebench.loadgen.aiperf.redact import REDACTED, redact, redact_argv, scrub_tree

SECRET = "sk-kneebench-sentinel-1111"


def test_redact_text_and_argv() -> None:
    assert redact(f"key={SECRET}", [SECRET]) == f"key={REDACTED}"
    assert redact("nothing here", [None, ""]) == "nothing here"
    assert redact_argv(["--api-key", SECRET], [SECRET]) == ("--api-key", REDACTED)


def test_scrub_tree_rewrites_only_text_files_with_secret(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    leaky = tmp_path / "logs" / "aiperf.log"
    leaky.write_text(f"input_value={{'apiKey': '{SECRET}'}}")
    clean = tmp_path / "profile_export_aiperf.json"
    clean.write_text('{"api_key": "<redacted>"}')
    binary = tmp_path / "data.parquet"
    binary.write_bytes(SECRET.encode())

    changed = scrub_tree(tmp_path, [SECRET])

    assert changed == [leaky]
    assert SECRET not in leaky.read_text()
    assert clean.read_text() == '{"api_key": "<redacted>"}'
    assert binary.read_bytes() == SECRET.encode()
