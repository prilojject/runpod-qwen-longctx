# Benchmark results: v1.0.0 (measured 2026-11-24, Asia/Bangkok)

Setup: Runpod Serverless, max 1 worker, no network volume (weights on container disk),
llama.cpp b12651 (prebuilt CUDA 12.8), image `ghcr.io/prilojject/runpod-qwen-longctx`
(the 4090 runs used digest `sha256:ed2f43ab…`, the 48 GB runs `sha256:a6dcb3d6…`; the handler logic is the same,
the later digest only adds the GPU-probe fix and the load-VRAM measurement fix).
Needle in a haystack built by `bench/needle.py` / `bench/haystack.py` (generated sentences, needle
`The secret code is <CODE>.` at 10% / 50% / 90% depth). `temperature 0`, `max_tokens 32`, thinking off,
`cache_prompt: false` (every prompt prefilled from scratch, `timings.cache_n = 0` in every row).

- **TTFT** = llama.cpp `timings.prompt_ms` (prefill time on the worker; excludes queue/network).
- **Prefill / decode tok/s** = `timings.prompt_per_second` / `timings.predicted_per_second`.
- **Peak VRAM** = max `nvidia-smi memory.used` sampled every 0.5 s during the request (whole GPU).
  Unavailable ("n/a") on the Blackwell MIG slice, where nvidia-smi answers `[Insufficient Permissions]`.
- Raw per-request records: `bench/raw/*.jsonl`.

## Needle results

| GPU | Preset | Context / KV | Prompt tokens | Needle found | TTFT (s) | Prefill tok/s | Decode tok/s | Peak VRAM (GiB) |
|---|---|---|---|---|---|---|---|---|
| RTX 4090 | p1 9B Q8_0 | 262144 / f16 | ~62.9K | 3/3 | 6.8–7.5 | 8,395–9,301 | 63–72 | 16.7 |
| RTX 4090 | p1 9B Q8_0 | 262144 / f16 | ~126.8K | 3/3 | 16.7–17.4 | 7,291–7,582 | 60–62 | 16.7 |
| RTX 4090 | p1 9B Q8_0 | 262144 / **q8_0** | ~126.7K | 1/1 | 18.0 | 7,054 | 66 | 13.9 |
| RTX 4090 | p2 35B-A3B UD-IQ4_XS | 131072 / q8_0 | ~62.9K | 3/3 | 8.9–9.0 | 6,967–7,046 | 112–119 | 18.3 |
| RTX 4090 | p2 35B-A3B UD-IQ4_XS | 131072 / q8_0 | ~126.8K | 3/3 | 22.3–22.5 | 5,631–5,690 | 96–98 | 18.3 |
| RTX 4090 | p2 model, `CONTEXT=262144` | 262144 / q8_0 | 247,120 | 1/1 | 59.9 | 4,128 | 70 | 20.0 |
| RTX A40 (48 GB)\* | p3 35B-A3B UD-Q6_K | 262144 / f16 | ~126.7K | 3/3 | 47.9–48.2 | 2,629–2,649 | 62–64 | 32.5 |
| RTX A40 (48 GB)\* | p3 35B-A3B UD-Q6_K | 262144 / f16 | 247,120 | 1/1 | 128.2 | 1,928 | 49 | 32.8 |
| RTX PRO 6000 Blackwell MIG 2g.48gb\*\* | p3 35B-A3B UD-Q6_K | 262144 / f16 | ~126.7K | 3/3 | 27.9–28.7 | 4,421–4,548 | 85–89 | n/a |
| RTX PRO 6000 Blackwell MIG 2g.48gb\*\* | p3 35B-A3B UD-Q6_K | 262144 / f16 | 247,120 | 1/1 | 72.2 | 3,423 | 67 | n/a |

\* The 48 GB pool (`AMPERE_48,ADA_48_PRO`) assigned an A40, the nearest equivalent to the planned L40S / A6000.
\*\* Runpod placed the first 48 GB worker on a MIG slice (47.4 GiB visible) before we excluded it. This confirms the
sm_120 kernels work. The Hub listing still excludes MIG slices (`gpuIds`).

## Load / memory (from llama-server logs, MiB)

| GPU | Preset | ctx / KV | model buffer | KV buffer | recurrent state (RS) | compute buffer | Peak VRAM GiB |
|---|---|---|---|---|---|---|---|
| RTX 4090 | smoke 9B Q4_K_M | 16384 / f16 | 4,861 | 512 | 50 | 120 | 5.9 |
| RTX 4090 | p1 9B Q8_0 | 262144 / f16 | (8.9 GiB file) | 8,192 | 50 | 360 | 16.7 |
| RTX 4090 | p1 9B Q8_0 | 262144 / q8_0 | | 4,352 | 50 | 1,344 | 13.9 |
| RTX 4090 | p2 35B-A3B IQ4_XS | 131072 / q8_0 | 16,383 | 1,360 | 63 | 436 | 18.3 |
| RTX 4090 | p2 model | 262144 / q8_0 | 16,383 | 2,720 | 63 | 820 | 20.0 |
| RTX 4090 | p2 model | 262144 / **f16** | 16,383 | 5,120 | 63 | 360 | 21.9 (fits, no fallback) |
| RTX A40 | p3 35B-A3B Q6_K | 262144 / f16 | 27,425 | 5,120 | 63 | 360 | 32.5–32.8 |
| RTX A40 | p4 27B UD-Q5_K_XL (load + 2K needle only) | 262144 / f16 | 18,266 | 16,384 | 150 | 398 | 34.7 |

KV per token matches the spec formula (9B: 32 KiB f16, 35B-A3B: 20 KiB f16, 27B: 64 KiB f16).
Context checkpoints (`--ctx-checkpoints`, default 32/slot) are kept in host RAM by llama-server. VRAM stayed flat
between requests.

## Auto-fit (OOM fallback): one forced test

`PRESET=p4-48gb-27b-256k` (27B, 18.7 GiB weights) on an RTX 4090:

| Attempt | ctx | KV | Result |
|---|---|---|---|
| 1 | 262144 | f16 | OOM |
| 2 | 262144 | q8_0 | OOM |
| 3 | 131072 | q8_0 | **loaded**: `effective_context=131072`, peak 23.44 GiB |

All three attempts took 12.2 s in total. A 126.8K-token needle then ran at that fallback configuration without a request-time OOM:
found, TTFT 64.0 s, prefill 1,982 tok/s, decode 29.5 tok/s. `worker.fallback_steps` and a warning reported the
step-down.

## Prompt cache on a hybrid model (2 turns, p2, RTX 4090, ~62.9K-token document)

| Turn | cached tokens (`cache_n`) | prefilled tokens | prefill ms |
|---|---|---|---|
| 1: new document + question | 0 | 62,870 | 9,080 |
| 2: same history + answer + new question (append) | 62,877 | 21 | 148 |
| 3: one sentence edited at ~25% depth | 8 | 62,864 | 8,723 |

Appending reuses the whole prefix. Editing the middle of the history re-prefills almost the whole prompt (recurrent state
cannot be rolled back).

## Other checks

- Hub smoke (`preset: smoke`, 1,832-token needle, RTX 4090): found. TTFT 0.33 s. Cold start (image pull + 5.3 GiB
  download + load) 148 s wall.
- Runpod OpenAI route through `https://api.runpod.ai/v2/<id>/openai/v1`: `/chat/completions` non-stream ✅, `stream: true`
  (SSE chunks, last chunk carries `worker`, then `[DONE]`) ✅, `/models` ✅.
- Download speed (container disk, HF xet): 9B Q8_0 8.9 GiB in 74.5 s incl. sha256; 27B Q5_K_XL 18.7 GiB in 49.8 s;
  35B Q6_K 27.3 GiB in 46.2 s (all including sha256 verification).
- 256K payload: a 247K-token prompt (≈0.9 MB JSON) worked through async `/run`.

## Not measured

- P4 needle at 64K/128K on 48 GB (only load + a 2K needle at 256K/f16), P5 (35B-A3B Q8_0), L40S / RTX 6000 Ada,
  RTX 3090 / L4 / A5000, B200/B300 (excluded in `gpuIds`).

## Cost

Total GPU spend for all runs above, including one failed 48 GB attempt (see `48gb_p3_failed_gpu-probe-bug.jsonl`,
fixed in the shipped image): **≈ $0.7** (Runpod balance delta), well under the $5 budget.
