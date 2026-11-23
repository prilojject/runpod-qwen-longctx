# Runpod Hub always builds from a Dockerfile. This thin file lets the Hub
# reuse the already-built public image (built by GitHub Actions from the
# repository-root Dockerfile) instead of rebuilding it.
#
# Pinned by digest, not a floating tag. This is the exact image used for the
# 48 GB GPU benchmarks in bench/results.md (llama.cpp b12651, kernels
# sm_86 / sm_89 / sm_120a verified at build time).
# Releases of this repo still use targetCommitish=main.
FROM ghcr.io/prilojject/runpod-qwen-longctx@sha256:a6dcb3d689949b2b0a465db553b616d79634c9b5fc9ff392f16069f9d4146dc3
