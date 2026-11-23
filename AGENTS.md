# Agent notes — prilojject/runpod-qwen-longctx

Headless Runpod Serverless worker: official llama.cpp `llama-server` on `127.0.0.1:8080` (`--no-webui`), proxied by `handler.py`.

- Engine pin: llama.cpp **b12651** (commit f5e6915c), prebuilt `llama-b12651-bin-ubuntu-cuda-12.8-x64.tar.gz`, sha256 `cb099350…ef367ea`. Never build from source, never use floating `:server-cuda` / `:latest`.
- Kernel check in the Dockerfile asserts only what the archive really contains: ELF sm_86, sm_89, sm_120a (PTX 50–90). Do not add asserts for archs that are not in the archive.
- Base: `nvidia/cuda:12.8.2-runtime-ubuntu24.04` (digest-pinned). No torch. `allowedCudaVersions` 12.8+ only.
- Weights: one `.gguf` per preset, pinned repo revision + sha256 (`src/presets.py`). Never `snapshot_download`, never mmproj, never bake into the image. Cache: `/runpod-volume/qwen-longctx`.
- OOM ladder: f16 → q8_0 → ctx/2 … ≥ MIN_CONTEXT, max 6 attempts, then `OOM_MIN_CONTEXT`. No infinite retries; every response carries `worker.effective_context`.
- OpenAI route (`openai_route` + `openai_input`) is required; the handler is an async generator with `return_aggregate_stream`.
- nvidia-smi may return `[Insufficient Permissions]` (MIG); `src/gpu.py` falls back to the CUDA driver API. Keep it defensive.
- Image builds: GitHub Actions only (`build-image.yml`, manual dispatch) → GHCR. `.runpod/Dockerfile` is a thin `FROM ghcr…@digest`.
- Benchmarks: `bench/needle.py`, results in `bench/results.md`. Only real measured numbers go into README / listing.
- Releases of this repo: `targetCommitish=main`. Secrets never in job input, repo or logs.
