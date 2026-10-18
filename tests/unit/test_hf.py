import pytest

from kneebench.detect.hf import root_to_hf_id


@pytest.mark.parametrize(
    ("root", "expected"),
    [
        ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        (
            "/root/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/abc",
            "unsloth/Meta-Llama-3.1-8B-Instruct",
        ),
        ("/mnt/models/models--Qwen--Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
        ("/data/my-finetune", None),
        ("my-served-model", None),
        ("", None),
        (None, None),
    ],
)
def test_root_to_hf_id(root: str | None, expected: str | None) -> None:
    assert root_to_hf_id(root) == expected
