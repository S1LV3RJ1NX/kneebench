# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added (M1: detect, theory and baseline)
- Auto-detection from the endpoint: vLLM version (`/version`), served model, context length
  and Hugging Face id (`/v1/models`), KV cache size and settings (`vllm:cache_config_info`),
  and client distance (DNS, TCP, TLS and a keep-alive `/health` round trip).
- Model physics from Hugging Face metadata: exact weight bytes from safetensors (dense, FP8
  and int4 checkpoints), KV bytes per token (GQA, fp8 cache), tensor parallelism.
  MoE and MLA models are rejected with a clear message until after the MVP.
- GPU catalog (H100 SXM/NVL/PCIe, H200, B200, A100, L40S, L4, A10, MI300X) plus custom GPUs.
- vLLM defaults per version (`content/vllm_versions.yaml`), verified against vLLM source at
  v0.29.0 to v0.32.0, with a live test that re-reads the source.
- SQLite storage of setups, runs, points, per-request records, timeslices, server metrics,
  AIPerf warnings and compressed raw artifacts.
- Phase 0 (theory) and phase 1 (single-request baseline), with ETAs that include AIPerf
  startup overhead.
- Rules T01-T08, B01-B06, H03-H05, H07, H08, each citing its lesson note, with text in a
  reviewable content layer and a workload-ordered Markdown and JSON report.
- `kneebench run`, `runs`, `report` (with `--gpu` to correct the GPU entry without a rerun),
  `detect` and `gpus` commands.

### Fixed
- Work around AIPerf 0.13.0 dropping `networkLatency` from YAML configs by also passing
  `--network-latency-automatic`.

### Added (M0)
- Project skeleton managed with uv; Python 3.11 to 3.13; macOS and Linux.
- AIPerf 0.13.0 bundled as a pinned core dependency. kneebench always launches the
  copy in its own environment and refuses to run with any other version.
- Per-point AIPerf configs written as YAML. The API key reaches AIPerf only through
  the child process environment, never argv or files, and all artifacts and logs
  are scrubbed after each point.
- Measured AIPerf per-process startup overhead (about 8 s warm with a Hugging Face
  tokenizer, about 20 s on first launch) for the planner's ETA.
- `kneebench doctor` with `--offline`, `--url` and `--tokenizer`: checks Python,
  platform, the bundled AIPerf, the data directory, endpoint routes
  (`/health`, `/version`, `/v1/models`, `/metrics`) and the tokenizer download,
  with plain-English fixes.
- CI on Ubuntu and macOS for Python 3.11 to 3.13, including a clean-install job.

### Dependencies
- `aiperf==0.13.0` (pinned). Upgrade only together with `AIPERF_PIN` and a note here.
