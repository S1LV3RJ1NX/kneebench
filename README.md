# kneebench

**Point it at your vLLM server. Get back a report that says how many users it can really serve, where it breaks, why, and which flag to change next.**

> **Status: pre-alpha (milestone M0).** Install, `kneebench doctor` and the bundled AIPerf runner work today. The benchmark phases and the report land in M1 and M2 (see the [roadmap](#roadmap)).

<!-- Screenshot of the report goes here once M2 ships. -->

## Why this exists

Your team deployed a model on a GPU. Someone asks: "How many users can it take?" You run a benchmark and get a wall of numbers: TTFT P99, ITL, tokens per second at twelve concurrency levels. Which one is the answer? Is 6 ms per token good for this GPU? Did the server run out of memory, or just get slow?

Benchmark tools measure well. They don't explain. kneebench runs the measurements that matter, checks them against what the hardware can physically do, and writes the conclusion in plain words with your own numbers:

> For your chat traffic, this server comfortably handles about 40 users at your latency target. It breaks at 48 because per-token latency crosses 50 ms, not because the GPU is full.

Every finding says what was measured, what it means, why it happens (with a link to a short lesson) and what to try next. You change your server, rerun, and compare.

## Quickstart (macOS or Linux)

```bash
git clone https://github.com/S1LV3RJ1NX/kneebench && cd kneebench
uv sync                                   # installs kneebench and its bundled AIPerf
uv run kneebench doctor --url http://your-vllm-host:8000
```

Need uv? `curl -LsSf https://astral.sh/uv/install.sh | sh`. If your server needs a key, set `KNEEBENCH_API_KEY` first. Gated models (such as Llama) also need `HF_TOKEN` so kneebench can download the tokenizer.

Once released, `uv tool install kneebench` (or `pip install kneebench`) will work too. The install is large (about 800 MB) because AIPerf brings `transformers`, `pandas` and friends.

## Where to run it

kneebench only sends HTTP requests, so it runs anywhere: a laptop, a Linux box or a pod. **Where** it runs changes the numbers:

- **Close to the server** (same cluster or region) gives numbers you can act on.
- **Across the internet** adds network time to every first token, and a slow client can become the bottleneck. kneebench measures the distance on every run (TCP connect, TLS handshake, a plain `/health` round trip), separates "server time" from "what your users see", and warns when the network dominates.

kneebench never starts, stops or reconfigures your server. It only benchmarks an existing OpenAI-compatible endpoint.

## What it measures

| Phase | Question it answers |
|---|---|
| 0. Theory | What is the fastest this model can possibly run on this GPU? How many requests fit in memory? |
| 1. Single-request baseline | With the GPU idle, how close to that limit do you get? |
| 2. Concurrency sweep | How do throughput and per-user speed trade off from 1 to hundreds of users? |
| 3. Request-rate sweep | What happens when users arrive at random, including in bursts? |
| 4. Shape sweep | How do prompt length and answer length change everything? |
| 5. Three walls | Where does it saturate, miss your latency target, or run out of memory? |

Every run also checks that the test itself is trustworthy: steady state reached, client not the bottleneck, outputs really as long as requested, tails and not just medians, errors counted as failures, versions recorded.

## How to read the report

*(Arrives in M2.)* Diagnosis first, evidence second, next steps last: a two-sentence verdict, a traffic-light scorecard, one section per finding that matters for your workload, a ranked "try these next" list with copy-ready flags for your vLLM version, and a glossary for every term. Exports to PDF (readable by people and LLMs), Markdown and JSON.

## Limitations

- **Endpoint only.** No GPU telemetry; memory and queue findings come from vLLM's `/metrics`. If your gateway hides `/metrics`, those findings are marked as not visible.
- **Synthetic prompts.** Random prompts never share a prefix, so prefix caching can't help them. Real chat traffic often does better on time to first token.
- **Client location matters.** See [Where to run it](#where-to-run-it).
- **No Windows.** macOS and Linux only (WSL2 works).
- **vLLM first.** Other OpenAI-compatible servers work with reduced findings; SGLang support is on the roadmap.

## Roadmap

- [x] **M0 Skeleton:** packaging with bundled, pinned AIPerf; `kneebench doctor`; CI.
- [ ] **M1 Detect, theory and baseline:** auto-detection, physics floors, SQLite, single-request baseline, Markdown report.
- [ ] **M2 MVP:** concurrency sweep, the three walls, web UI, full report with PDF export.
- [ ] **M3 Rate sweep:** Poisson and bursty arrivals.
- [ ] **M4 Shape sweep:** prompt and output length map, multi-shape workloads.
- [ ] **M5 What-ifs and compare:** move the SLO and watch capacity move; before and after runs.
- [ ] **M6 Release:** v0.1.0 on PyPI.
- [ ] **Later:** MoE and MLA physics, GuideLLM backend, SGLang servers, optional friendlier wording layer.

## Credits

- [NVIDIA AIPerf](https://github.com/ai-dynamo/aiperf) (Apache-2.0) generates the load. kneebench bundles a pinned version.
- The analysis follows the [Inference Engineering notes](https://github.com/S1LV3RJ1NX/inference-engineering-notes), lessons 24 to 27.
- [GuideLLM](https://github.com/vllm-project/guidellm) (Apache-2.0) provides the mock server used in integration tests.

## License

MIT. See [LICENSE](LICENSE).
