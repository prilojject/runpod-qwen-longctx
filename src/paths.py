"""Filesystem locations. Weights are never baked into the image."""

from __future__ import annotations

import os

VOLUME_ROOT = "/runpod-volume"
LLAMA_DIR = os.environ.get("LLAMA_CPP_DIR", "/opt/llama.cpp")
LLAMA_SERVER = os.path.join(LLAMA_DIR, "llama-server")
LLAMA_CPP_TAG = "b12651"
LLAMA_CPP_COMMIT = "f5e6915c28f10330885fcb9243a93ae98d3115ab"


def has_volume() -> bool:
    return os.path.isdir(VOLUME_ROOT) and os.access(VOLUME_ROOT, os.W_OK)


def cache_root() -> str:
    override = os.environ.get("QWEN_LONGCTX_CACHE", "").strip()
    if override:
        return override
    if has_volume():
        return os.path.join(VOLUME_ROOT, "qwen-longctx")
    return "/cache/qwen-longctx"


def models_dir() -> str:
    return os.path.join(cache_root(), "models")


def model_path(repo: str, revision: str, filename: str) -> str:
    return os.path.join(models_dir(), repo.replace("/", "__"), revision, filename)
