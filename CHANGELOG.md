# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
