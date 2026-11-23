# Qwen Long Context 256K — Single GPU (Runpod Serverless)

[![Runpod Hub](https://api.runpod.io/badge/prilojject/runpod-qwen-longctx)](https://console.runpod.io/hub/prilojject/runpod-qwen-longctx)

OpenAI-compatible serverless endpoint that runs **hybrid Qwen models with 128K–256K context on one 24 GB or 48 GB GPU**.
llama.cpp + pinned GGUF weights + tested presets + automatic context fitting.

- **Models:** Qwen3.5-9B · Qwen3.6-35B-A3B (MoE) · Qwen3.6-27B
- **Context:** up to 262,144 tokens
- **GPUs:** RTX 4090 / 3090 / L4 / A5000 (24 GB), L40S / L40 / RTX 6000 Ada / A6000 / A40 (48 GB)
- **API:** `/openai/v1/chat/completions`, `/openai/v1/completions`, `/openai/v1/models`, streaming
- **No HF token needed** (Apache-2.0 weights, downloaded once to your network volume)

## Why long context fits on a smaller GPU

In a regular transformer, every layer keeps a memory of every token it has read (the KV cache). At 256K tokens that memory often grows larger than the model itself.

The new Qwen models are *hybrid*. Only **1 of every 4 layers** uses full attention and needs that per-token memory. The other 3 use **Gated DeltaNet**, a linear-attention layer that compresses what it has read into a small fixed-size state, about 2 MB per layer, the same at 1K tokens as at 256K. Qwen3.6-35B-A3B is also a Mixture-of-Experts model, so only about 3B of its 35B parameters work on each token.

Put together:
- **~4× less context memory** than an all-attention model with the same layer layout. For Qwen3.6-35B-A3B, the whole 256K context needs about **5 GiB** of KV cache at full 16-bit precision.
- **4-bit to 6-bit GGUF weights** (llama.cpp) make the 35B model fit in 16–27 GiB.
- **Optional 8-bit KV cache** on 24 GB cards. The linear-attention state always stays in full 32-bit precision; only the attention layers' cache is compressed.

Long context gets **cheaper**, not free: reading a very long prompt still takes time (see Limitations).

### Memory math (Qwen3.6-35B-A3B)

| | All-attention model (hypothetical) | Hybrid Qwen3.6-35B-A3B |
|---|---|---|
| Layers with per-token KV cache | 40 | 10 |
| KV cache @ 256K, 16-bit | ~20 GiB | **5 GiB** (measured: 5,120 MiB) |
| Linear-attention state | — | ~63 MB, constant (measured: 62.8 MiB) |

Formula: `2 × attention_layers × kv_heads × head_dim × bytes × tokens` (kv_heads = 2, head_dim = 256 from `config.json`).

## What it's good for

- **Long documents:** contracts, reports, books, transcripts. Ask questions over the whole text without chunking.
- **RAG with big context:** put dozens of retrieved documents in one prompt instead of the top 3.
- **Agents with long memory:** keep long tool-call histories and conversations in context.
- **Codebase review:** paste a whole repository or a large diff and ask for a review or a summary.

## Presets

| Preset | GPU | Model (GGUF) | Context | KV cache | Measured peak VRAM |
|---|---|---|---|---|---|
| **24GB · 9B · 256K** (default on 24 GB) | RTX 4090 / 3090 / L4 / A5000 | Qwen3.5-9B Q8_0 | 256K | 16-bit | 16.7 GiB (RTX 4090) |
| **24GB · 35B-A3B · 128K** | RTX 4090 / 3090 / L4 / A5000 | Qwen3.6-35B-A3B UD-IQ4_XS | 128K | 8-bit | 18.3 GiB (RTX 4090) |
| **48GB · 35B-A3B · 256K** (default on 48 GB) | L40S / L40 / RTX 6000 Ada / A6000 / A40 | Qwen3.6-35B-A3B UD-Q6_K | 256K | 16-bit | 32.8 GiB (A40) |
| **48GB · 27B dense · 256K** | L40S / L40 / RTX 6000 Ada / A6000 / A40 | Qwen3.6-27B UD-Q5_K_XL | 256K | 16-bit | 34.7 GiB (A40, load test only) |

Peak VRAM = whole-GPU `nvidia-smi` maximum during the longest measured request (see `bench/results.md`).
Tip: on a 4090 the 35B-A3B model also loaded with `CONTEXT=262144` (8-bit KV, 20.0 GiB) and found a needle in a
247K-token prompt. It is not the default preset because the margin on 24 GB cards is smaller.

| Env `PRESET` | Card |
|---|---|
| `auto` (default) | < 40 GiB VRAM → `p1-24gb-9b-256k`, otherwise `p3-48gb-35b-256k` |
| `p1-24gb-9b-256k` | Qwen3.5-9B Q8_0, 256K, KV f16 |
| `p2-24gb-35b-128k` | Qwen3.6-35B-A3B UD-IQ4_XS, 128K, KV q8_0 |
| `p3-48gb-35b-256k` | Qwen3.6-35B-A3B UD-Q6_K, 256K, KV f16 |
| `p4-48gb-27b-256k` | Qwen3.6-27B UD-Q5_K_XL, 256K, KV f16 (load-tested; needle not benchmarked yet) |
| `p5-48gb-35b-q8-256k` | Qwen3.6-35B-A3B Q8_0, 256K, KV q8_0 (max weight quality, not benchmarked yet) |
| `smoke` | Qwen3.5-9B Q4_K_M, 16K (quick test) |
| `custom` | your own GGUF via `MODEL_REPO` / `MODEL_FILE` / `MODEL_REVISION` |

## Benchmark (needle-in-a-haystack, TTFT, speed)

Measured by us on 2026-11-24, llama.cpp b12651, temperature 0, thinking off, prompt cache off. Needle at 10% / 50% / 90% depth.
TTFT = server-side prefill time (`timings.prompt_ms`). Full data and method: [`bench/results.md`](bench/results.md).

| GPU | Preset | Prompt | Needle found | TTFT | Prefill tok/s | Decode tok/s | Peak VRAM |
|---|---|---|---|---|---|---|---|
| RTX 4090 | p1 | ~64K | 3/3 | 6.8–7.5 s | 8,395–9,301 | 63–72 | 16.7 GiB |
| RTX 4090 | p1 | ~128K | 3/3 | 16.7–17.4 s | 7,291–7,582 | 60–62 | 16.7 GiB |
| RTX 4090 | p2 | ~64K | 3/3 | 8.9–9.0 s | 6,967–7,046 | 112–119 | 18.3 GiB |
| RTX 4090 | p2 | ~128K | 3/3 | 22.3–22.5 s | 5,631–5,690 | 96–98 | 18.3 GiB |
| A40 (48 GB) | p3 | ~128K | 3/3 | 47.9–48.2 s | 2,629–2,649 | 62–64 | 32.5 GiB |
| A40 (48 GB) | p3 | ~247K | 1/1 | 128.2 s | 1,928 | 49 | 32.8 GiB |

Also measured: p1 at 128K with 8-bit KV (found, 13.9 GiB); p3 on an RTX PRO 6000 Blackwell MIG 48 GB slice (128K: 3/3, TTFT ~28 s; 247K: found, TTFT 72 s).
L40S / A6000 / 3090 / L4 were not available in this run. Your numbers will vary with GPU model and prompt.

## Quick start

1. Deploy from the Runpod Hub (or create a Serverless endpoint from `ghcr.io/prilojject/runpod-qwen-longctx:v1.0.0`).
2. **Attach a network volume** (≥ 40 GB). Weights are cached there, so later cold starts skip the 5–34 GiB download.
3. Pick a 24 GB or 48 GB GPU and a `PRESET`. Set **Execution timeout ≥ 600 s**.
4. Call it with openai-python:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
    api_key="<RUNPOD_API_KEY>",
)

long_text = open("contract.txt").read()   # up to the preset's context

resp = client.chat.completions.create(
    model="qwen-longctx",
    messages=[
        {"role": "system", "content": "Answer only from the document."},
        {"role": "user", "content": long_text + "\n\nList every termination clause with its section number."},
    ],
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

Quick start (curl):

```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/chat/completions \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-longctx",
    "messages": [{"role": "user", "content": "Summarize this repository: ..."}],
    "max_tokens": 512
  }'
```

Native Runpod input (`/run`, `/runsync`) also works:

```json
{"input": {"messages": [{"role": "user", "content": "..."}], "max_tokens": 512}}
```

The handler is a streaming generator (like the official vLLM worker), so a native `/runsync` / `/status` result is a one-element
list: `output[0]` is the OpenAI-compatible JSON. Through `/openai/v1/...` you get the plain OpenAI object. Every response carries a `worker` block (real output from our test run):

```json
"worker": {"preset": "smoke", "effective_context": 16384, "kv_cache_type": "f16", "fallback_steps": [],
           "gpu": "NVIDIA GeForce RTX 4090", "vram_total_gib": 23.99, "elapsed_seconds": 0.17,
           "memory_mib": {"CUDA0 model": 4861.28, "CUDA0 KV": 512, "CUDA0 RS": 50.25, "CUDA0 compute": 120.02},
           "warnings": ["no network volume: ..."]}
```

Per-job overrides (restart llama-server when they change): `preset` (any named preset), `context`, `kv_cache_type`, `min_context`.
`custom` models are env-only. Tokens / API keys are rejected in job input.
For prompts above ~128K tokens prefer async `/run` + `/status` (a 247K-token prompt is ~0.9 MB of JSON and worked).

## Configuration

| Env | Default | Description |
|---|---|---|
| `PRESET` | `auto` | Model + context + KV bundle (table above) |
| `CONTEXT` | preset | Target context in tokens, or `auto` (largest that fits, llama.cpp `--fit`). Capped at 262,144 |
| `MIN_CONTEXT` | `16384` | Auto-fit never goes below this |
| `KV_CACHE_TYPE` | preset | `f16` / `q8_0` / `q4_0` for the attention layers (linear-attention state is always FP32) |
| `PARALLEL` | `1` | llama.cpp slots; the context is shared between slots |
| `ENABLE_THINKING` | `false` | Qwen thinking by default; per-request `chat_template_kwargs.enable_thinking` wins |
| `MAX_TOKENS_DEFAULT` | `2048` | Used when a request has no `max_tokens` |
| `MODEL_REPO` / `MODEL_FILE` / `MODEL_REVISION` | — | Only for `PRESET=custom` (revision required; sha256 check is off for custom files) |
| `LOAD_TIMEOUT_S` | `900` | Max download + load time on a cold start |
| `HF_TOKEN` | — | Not needed for the built-in presets (use a Runpod secret for private repos) |

Sampling defaults when the client sets none (Qwen3.6 card): non-thinking `temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5`;
thinking `1.0 / 0.95 / 20 / 0 / 1.5`.

## Auto-fit (what happens on out-of-memory)

1. 16-bit KV → 8-bit KV at the same context.
2. Halve the context, repeat down to `MIN_CONTEXT` (16K).
3. Still no fit → a clear `OOM_MIN_CONTEXT` error. No restart loops, max 6 attempts.

`worker.fallback_steps` shows exactly what was tried. Measured example: the 27B 256K preset on an RTX 4090 went
256K/f16 → OOM, 256K/q8_0 → OOM, 128K/q8_0 → loaded (12 s for all three attempts) and then answered a 127K-token needle correctly.
A prompt longer than the effective context fails with `CONTEXT_EXCEEDED` and the token counts; nothing is cut silently.

## Limitations

- **Long prompts take time to read.** The first token on a 128K–256K prompt can take tens of seconds or more on a single GPU. Exact times per preset are in the README benchmark. Set your endpoint's execution timeout high enough (we suggest ≥ 600 s), and use async `/run` for very long prompts.
- **Quality at the far end of the context.** The 24 GB presets passed a needle-in-a-haystack test at 64K and 128K (3 depths each), the 48 GB 35B-A3B preset at 128K and 247K; the 27B preset is only load-tested so far. Finding one fact is easier than reasoning across 200K tokens, though, and 8-bit KV cache (used on 24 GB) can lose a little precision. Use the 48 GB presets when quality matters most.
- **One long request at a time by default.** The whole context goes to a single request (`PARALLEL=1`). Raising `PARALLEL` splits the context between requests. For high-throughput serving of many short requests on 80 GB GPUs, the official vLLM worker is the better fit.
- **Text only in v1.** The vision part of Qwen3.5/3.6 is not loaded.
- Thinking mode is **off by default** for faster, cheaper answers. Turn it on per request with `"chat_template_kwargs": {"enable_thinking": true}` or with `ENABLE_THINKING=true`.
- **Prompt cache on hybrid models:** appending to a conversation reuses the whole prefix (measured: 62,877 tokens cached, 148 ms),
  but editing the middle of a long history re-reads almost the whole prompt (the linear-attention state cannot be rolled back).
- Without a network volume, weights are re-downloaded on every cold start (measured 46–75 s for 9–27 GiB incl. sha256).
- B200/B300 are excluded (only PTX JIT for sm_100/103, not tested). sm_86, sm_89 and sm_120 kernels ship in the binary.

## FAQ

**Is this faster than vLLM?** Not for many parallel requests. This worker is about fitting long context on a cheaper GPU with presets that already work. For high-concurrency serving on 80 GB GPUs, use the official vLLM worker.
**Can I use 1M context?** No. Qwen's native window is 262,144 tokens; longer windows need RoPE scaling and are out of scope for v1.
**Images / video?** Not in v1 (text only).
**Tool calling / JSON?** Passed through to llama.cpp's OpenAI endpoint (`--jinja` chat template). Treat it as best-effort.

## How it works

`handler.py` starts the official `llama-server` (llama.cpp **b12651**, prebuilt CUDA 12.8 archive, sha256-verified at build time)
on `127.0.0.1:8080` with `--no-webui`, and proxies jobs to it. The GGUF file is downloaded by pinned revision SHA, verified against
its sha256 once (a marker file avoids re-hashing), and cached under `/runpod-volume/qwen-longctx`. Nothing is baked into the image.
The build fails if the archive's `libggml-cuda.so` lacks sm_86 / sm_89 / sm_120 cubins (`cuobjdump --list-elf`).

## Licenses

Worker code: MIT. llama.cpp: MIT (ggml-org), see `LICENSE.llama.cpp`. Weights: Qwen3.5-9B, Qwen3.6-35B-A3B, Qwen3.6-27B by Qwen / Alibaba Cloud, Apache-2.0; GGUF by Unsloth, Apache-2.0. Pinned revisions are in `src/presets.py`.
