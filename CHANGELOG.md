# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Security
- certifi raised to 2026.10.19 or newer and uv to 0.13.4 or newer: older versions still trust
  the compromised eMudhra intermediate certificates, which allows TLS interception.

### Changed (report and UI)
- Reports open with a Setup block (model, GPU, server, cache, scheduler limits and where they
  came from, traffic, targets, where the test ran, what ran) and an Executive summary in
  bullets: capacity, how much traffic to plan for, the first server change to try, caveats.
- A light and dark mode toggle in the header; it follows the system until you pick one.
- The new-run form marks required fields and explains every field in one line behind a (?).

### Added (plan gaps)
- H10 checks kneebench was alone on the server: no requests running or waiting at the start,
  and never more than one running while kneebench sent one. The CLI and the run form warn
  before a run when the server is already busy.
- max_num_seqs and the per-step token budget are inferred from the engine when its behaviour
  pins them down (the running count flattening while requests wait; steps larger than the
  default, or a lower ceiling across queued levels), and labelled as inferred. E02, W06,
  I01 and H09 use them instead of assuming the version defaults.
- Comparisons and experiments measure their noise band: changes smaller than the larger
  start-to-end move of either run's repeated baseline count as noise, with 5% as the floor.

### Fixed (from the full live run)
- The latency wall is found at or above the operating point, so a noisy tail at a low level
  no longer reads "4 users comfortably, breaking at about 1".
- B04 and B06 suggest moving kneebench closer to the server when the network caused the
  spread, not checking the GPU; rule actions can now depend on the finding's case.
- B06 and B04 compare the client's time per token with the server's own histogram: when the
  server held steady they say the network path changed, instead of blaming the GPU.
- H04 trusts the server's output-token count over AIPerf's recount of long streamed answers,
  and expects the mixed lengths of a scheduled point. H05 skips multi-turn points, where
  every message adds its own chat-template tokens.
- "1 users" in sentences.

### Changed
- The bundled demo and README screenshots come from a fresh full run with every phase.

### Added (quality gate)
- Guided experiments can record answers before a change and compare them after (lesson 54):
  the same questions twice for the noise floor, then question by question with flips, the
  paired accuracy change and its 95% interval, an exact McNemar test and per-category
  results. The pass rule is saved before any answer exists; changed answers are listed to read.
- A 290-question starter set with program-checkable answers, or your own JSONL. Answers are
  graded by rules, never by a model. `kneebench quality ask | runs | compare` on the command
  line, which also covers two endpoints side by side.

### Added (roofline and precision what-ifs)
- F01 places every load level on the GPU's roofline: FLOPs per byte of a decode step against
  the ridge, the throughput each level reached against the line, and where the cache reads
  cap decode at long contexts (lesson 61). With a roofline chart.
- F02 projects an FP8 KV cache, FP8 or INT8 weights and activations (only where the GPU has
  those tensor cores) and 4-bit weight-only: weight memory, requests that fit, decode and
  prefill speed at your load (lessons 50, 51, 53). When the cache is the limit it suggests
  `--kv-cache-dtype fp8`, which the guided experiment can verify from outside.

### Added (prompt reuse)
- New phase at the operating point: the same load with prompts that never repeat, with one
  shared 1,500-token system prompt, and as six-turn conversations that resend the real
  answers (lessons 47 and 48).
- P01 reports what the shared prompt saves in first-token time and the server's prefix-cache
  hit rate, and suggests `--enable-prefix-caching` when it is off. P02 shows first-token time
  per turn as the history grows. P03 estimates how many distinct system prompts the cache
  holds before reuse collapses (lesson 49).
- P04: the same retrieved passages in ranked order and sorted by document id, with the
  reuse and first-token time each gets (lesson 48 section 5).
- New phase: abandoned requests. Every request asks for a long answer and hangs up after a
  second; kneebench reads the server's token counters before and after. A01 warns when the
  server finishes answers nobody is reading, usually a gateway not passing disconnects on
  (lesson 36 section 3).

### Added (long-prompt interference)
- New phase: 32 streaming chats, once alone and once with a long prompt arriving every 10 s
  (lesson 44), replayed on a fixed schedule. kneebench rebuilds every streamed chunk's arrival
  time and reports the longest pause, P99.9 and stalls, which an average or P99 hides.
- Rule I01 ties pauses to the long prompts only when they exceed anything the streams showed
  alone, says so when the network is too noisy to tell, and otherwise suggests a smaller
  `--max-num-batched-tokens` sized from your per-token target (lesson 45).

### Added (guided tuning and the engine view)
- Guided experiments: every server change in a report has "Try this change". kneebench shows
  it for your deployment (vllm serve, Kubernetes/Helm or a managed platform), checks from
  outside that the server restarted and that visible settings changed, runs a short
  verification over the load the change can move, and compares it with the run that proposed
  it. Keep or revert; the Experiments page is the ledger.
- Engine view: E01 splits first-token time into network, queue and prompt read from vLLM's
  own histograms; E02 names the limit that makes requests wait (sequence cap, per-step token
  budget or KV cache).
- Physics and rules updated for lessons 30-61: KV-aware decode and attention-aware prefill
  floors, p95 block-rounded admission sizing, req/s headroom with a failure margin,
  evictions as a rate, quantization by compute path.

### Added (demo run)
- The UI opens on a real, sanitized demo run with every phase on first start. It is a normal
  run: delete it any time, `kneebench demo` restores it, `KNEEBENCH_NO_DEMO=1` skips it.
  `scripts/make_demo.py` refreshes it from any finished run.

### Added (M5: what-ifs and compare)
- What-if targets: re-judge a stored run against other latency targets from the report page
  or `kneebench report <id> --slo-ttft-ms/--slo-tpot-ms/--slo-percentile`. Nothing is rerun
  or saved, and exports keep the what-if.
- Request-size estimates in the report: first-token time, per-token speed, answer time and
  cache fit for a prompt and output length the run didn't test, from the shape sweep.
- Compare two runs (`kneebench compare <before> <after>`, or the form on the run list): setup
  and workload diff, results with ratios and a 5% noise band, overlaid curves, and rules X01
  (what changed, and whether the comparison is clean) and X02 (what it did).

### Added (M4: shape sweep)
- Your own request size (`--shapes 2000/500`, or "Typical request" in the form) becomes the
  main shape, tested next to the standard chat, reasoning and summarization shapes.
- Request-size sweep: prompt length (output fixed at 256) and output length (prompt fixed at
  1,024), each at 1 user and at the operating point, capped to the context window, plus the
  shape map: every shape at the same load, never averaged.
- Rules S01-S07 (first-token cost per prompt token, long prompts slowing decode, answer time
  per output token, long answers holding cache, your traffic, the shape map, the prompt to
  output exchange rate), with charts and their own report section ordered by product type.

### Added (M3: request-rate sweep)
- Poisson arrivals at 25% to 110% of the measured capacity and bursty (gamma) arrivals near
  it, 90 seconds each; rates come from the concurrency sweep at run time.
- Rules R01-R04 (sustainable request rate, run-length dependence past capacity, client
  shortfall, burst penalty).

### Added (M2: concurrency sweep, walls, web UI and PDF)
- Concurrency sweep (1, 2, 4 ... 256, capped by memory and `max_num_seqs`) with 5% length
  variation, a time budget, and early stop once the walls are found or on errors and
  gateway failures. ETAs include AIPerf startup and a physics model of per-token cost.
- Steady-state throughput that excludes ramp-up and drain.
- Rules C01-C05 (curve, operating point, peak vs capacity, tails), W01-W09 (saturation, SLO
  wall, capacity as the lowest wall, preemption, `max_num_seqs` advice, errors, headroom)
  and H01, H02, H06, H09 (steady state, weak client, prefix caching, gateways).
- First-token targets judged on server-side time, without the client's network round trip.
- Web UI (`kneebench` or `kneebench ui`): detection-first new-run form, live progress with
  stop, doctor page, run list with delete, and the report with a contents sidebar,
  collapsible findings and interactive Plotly charts.
- Exports: PDF (headless Chromium, with an appendix of raw numbers and the Markdown and
  JSON reports attached), self-contained HTML, Markdown and JSON. Browser printing works
  without Chromium.
- One run at a time across the CLI and the UI (shared lock); failed, cancelled and
  interrupted runs are recorded with the reason. `kneebench delete` removes a run.
- Design: warm paper palette with bundled Geist, Geist Mono and Source Serif 4 fonts.

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
  `--network-latency-automatic` (reported upstream as
  [ai-dynamo/aiperf#1583](https://github.com/ai-dynamo/aiperf/issues/1583)).

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
