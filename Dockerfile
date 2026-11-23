# Runpod Serverless worker: hybrid Qwen long context (128K-256K) on llama.cpp.
# llama-server binds 127.0.0.1:8080 only; its web UI is disabled (--no-webui).
#
# Engine: official prebuilt llama.cpp release b12651 (commit f5e6915c), CUDA 12.8
# build, verified by sha256. Not built from source, no floating :server-cuda tag.
# Weights are NOT baked in: one pinned .gguf is downloaded at runtime and cached
# under /runpod-volume/qwen-longctx when a network volume is attached.
# Releases of this repo must use targetCommitish=main.

# ---- stage 1: fetch + verify llama.cpp, check CUDA kernels ------------------
FROM ubuntu:24.04 AS llama
ARG LLAMA_TAG=b12651
ARG LLAMA_SHA256=cb099350c6a025e67eefaa13f2b67b2dcce53e38a0da22476e070d166ef367ea
# cuobjdump from NVIDIA's CUDA 12.8.1 redistributables (only used for this check).
ARG CUOBJDUMP_URL=https://developer.download.nvidia.com/compute/cuda/redist/cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-12.8.90-archive.tar.xz
ARG CUOBJDUMP_SHA256=dbf41355a0f7a8aa491c1839a6c6ff8df02cdb6c6eefd66360543616c1b4980f
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp/build
RUN curl -fsSL -o llama.tgz "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz" \
    && echo "${LLAMA_SHA256}  llama.tgz" | sha256sum -c - \
    && mkdir -p /opt/llama.cpp \
    && tar -xzf llama.tgz -C /opt/llama.cpp --strip-components=1 \
    && rm llama.tgz \
    && test -x /opt/llama.cpp/llama-server && test -f /opt/llama.cpp/libggml-cuda.so && test -f /opt/llama.cpp/LICENSE
# FramePack / LLaMA-Factory lesson: assert only kernels the archive really ships.
# Inspected b12651: ELF cubins sm_86, sm_89, sm_120a (+ PTX 50/61/70/75/80/90).
RUN curl -fsSL -o cuobjdump.txz "${CUOBJDUMP_URL}" \
    && echo "${CUOBJDUMP_SHA256}  cuobjdump.txz" | sha256sum -c - \
    && tar -xJf cuobjdump.txz \
    && CUOBJDUMP="$(find . -name cuobjdump -type f | head -n1)" \
    && "$CUOBJDUMP" --list-elf /opt/llama.cpp/libggml-cuda.so | sed -E 's/.*\.(sm_[0-9a-z]+)\..*/\1/' | sort -u > /opt/llama.cpp/KERNELS.txt \
    && echo "ELF kernels:" && cat /opt/llama.cpp/KERNELS.txt \
    && for arch in sm_86 sm_89 sm_120; do grep -q "^${arch}" /opt/llama.cpp/KERNELS.txt || { echo "missing ${arch} kernels"; exit 1; }; done \
    && rm -rf /tmp/build/*

# ---- stage 2: runtime ---------------------------------------------------------
FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04@sha256:7e62235c5d6f22a57ec8ba51f1583634fdbbf4b08a15b42dc1a60f086145c210

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    LLAMA_CPP_DIR=/opt/llama.cpp \
    PRESET=auto \
    MIN_CONTEXT=16384 \
    PARALLEL=1 \
    ENABLE_THINKING=false \
    MAX_TOKENS_DEFAULT=2048 \
    LOAD_TIMEOUT_S=900

# libgomp/libssl/zlib/zstd are runtime deps of the prebuilt binaries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl python3 python3-venv libgomp1 libssl3t64 zlib1g libzstd1 \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama /opt/llama.cpp /opt/llama.cpp
# Every shared lib must resolve except libcuda.so.1 (injected by the host driver).
RUN missing="$(for f in /opt/llama.cpp/llama-server /opt/llama.cpp/*.so; do ldd "$f" | grep 'not found'; done | grep -v 'libcuda.so.1' | sort -u)"; \
    if [ -n "$missing" ]; then echo "unresolved libs: $missing"; exit 1; fi; \
    /opt/llama.cpp/llama-server --version 2>&1 | tee /opt/llama.cpp/VERSION.txt; \
    grep -q "b12651\|12651" /opt/llama.cpp/VERSION.txt || { echo "unexpected llama.cpp version"; exit 1; }; \
    /opt/llama.cpp/llama-server --help 2>&1 | grep -q -- "--no-webui" || { echo "--no-webui flag missing"; exit 1; }

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && pip freeze > /app/requirements.lock \
    && python -c "import runpod, huggingface_hub, requests, sys; print('runpod', runpod.__version__, '| huggingface_hub', huggingface_hub.__version__, '| python', sys.version.split()[0])"

COPY handler.py /app/handler.py
COPY src /app/src
COPY LICENSE NOTICE LICENSE.llama.cpp /app/
RUN python handler.py --test_input '{"input": {"dry_run": true, "preset": "smoke"}}' > /dev/null

# No secrets baked in. Port 8080 is loopback-only and not exposed.
CMD ["python", "-u", "/app/handler.py"]
