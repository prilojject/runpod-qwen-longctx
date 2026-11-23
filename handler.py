"""Runpod Serverless entry point: hybrid Qwen long context on llama.cpp.

llama-server runs on 127.0.0.1:8080 only (web UI disabled). Jobs come in as
- Runpod OpenAI route: {"openai_route": "/v1/chat/completions", "openai_input": {...}}
  (what https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1 sends), or
- native: {"messages": [...]} / {"prompt": "..."} + OpenAI sampling fields.

Local dry run (no GPU, no weights):
    python handler.py --test_input '{"input": {"dry_run": true}}'
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
import traceback
from typing import Any, AsyncIterator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import __version__  # noqa: E402
from src.config import (  # noqa: E402
    ConfigError,
    ServerConfig,
    apply_request_defaults,
    plan_ladder,
    resolve_request,
    resolve_server_config,
)
from src.gpu import VramPeak, gpu_info  # noqa: E402
from src.models import ModelError, ensure_model  # noqa: E402
from src.paths import LLAMA_CPP_TAG, cache_root, has_volume  # noqa: E402
from src.secret_utils import RedactFilter, redact  # noqa: E402
from src.server import LlamaServer, ServerError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _h in logging.getLogger().handlers:
    _h.addFilter(RedactFilter())
log = logging.getLogger("qwen-longctx")

_SERVER = LlamaServer(log=lambda m: log.info("%s", m))
_LOCK = threading.Lock()
_GPU: dict[str, Any] | None = None
_RESTART_FROM: dict[tuple, int] = {}  # key -> ladder index to start from after a request-time OOM
STREAM_FLUSH_S = 0.25
STREAM_FLUSH_N = 32


def _gpu() -> dict[str, Any]:
    global _GPU
    if _GPU is None or not _GPU.get("gpu"):
        _GPU = gpu_info()
    return _GPU


def _parallel() -> int:
    try:
        return max(1, min(16, int(os.environ.get("PARALLEL", "1") or 1)))
    except ValueError:
        return 1


def worker_block(scfg: ServerConfig | None, started: float, warnings: list[str], extra: dict | None = None) -> dict:
    st = _SERVER.state
    g = _gpu()
    blk: dict[str, Any] = {
        "version": __version__,
        "llama_cpp": LLAMA_CPP_TAG,
        "preset": scfg.preset if scfg else None,
        "model_repo": scfg.model_repo if scfg else None,
        "model_file": scfg.model_file if scfg else None,
        "revision": scfg.revision if scfg else None,
        "effective_context": st.effective_context if st else None,
        "kv_cache_type": st.kv_cache_type if st else (scfg.kv_cache_type if scfg else None),
        "fallback_steps": st.fallback_steps if st else [],
        "gpu": g.get("gpu"),
        "vram_total_gib": g.get("vram_total_gib"),
        "elapsed_seconds": round(time.time() - started, 2),
        "warnings": list(dict.fromkeys(warnings)),
    }
    if st:
        blk["memory_mib"] = st.memory
        blk["load_seconds"] = st.load_seconds
    if extra:
        blk.update(extra)
    return blk


def _error(code: str, message: str, scfg: ServerConfig | None, started: float, warnings: list[str],
           details: dict | None = None) -> dict:
    blk = worker_block(scfg, started, warnings, details)
    # Yielding {"error": ...} fails the Runpod job (no silent success).
    return {"error": f"{code}: {redact(message)} | worker={json.dumps(blk, default=str)[:3000]}"}


def ensure_server(scfg: ServerConfig, warnings: list[str]) -> dict[str, Any]:
    """Blocking: make sure llama-server is running with scfg. Returns load info."""
    with _LOCK:
        st = _SERVER.state
        if st is not None and st.key == scfg.key() and _SERVER.alive():
            return {"cold_start": False}
        if st is not None and _parallel() > 1 and st.key != scfg.key():
            raise ConfigError("per-request preset/context changes are disabled when PARALLEL > 1")
        deadline = time.time() + scfg.load_timeout_s
        info = ensure_model(scfg.model_repo, scfg.model_file, scfg.revision, scfg.sha256, scfg.size_bytes,
                            timeout_s=scfg.load_timeout_s, log=lambda m: log.info("%s", m))
        start_index = _RESTART_FROM.pop(scfg.key(), 0)
        with VramPeak() as vp:
            _SERVER.load(info["path"], scfg.context, scfg.kv_cache_type, scfg.min_context, scfg.parallel,
                         deadline, key=scfg.key(), start_index=start_index)
        st = _SERVER.state
        if st and st.fallback_steps:
            warnings.append(f"auto-fit: preset did not fit, running context={st.effective_context} "
                            f"kv={st.kv_cache_type}")
        return {"cold_start": True, "download_seconds": info.get("seconds", 0.0),
                "downloaded": info.get("downloaded"), "load_vram_peak_gib": vp.peak_gib}


def _mark_request_oom(scfg: ServerConfig) -> None:
    st = _SERVER.state
    if st is None or scfg.context == "auto":
        _SERVER.stop()
        return
    plan = plan_ladder(int(scfg.context), scfg.kv_cache_type, scfg.min_context)
    try:
        idx = plan.index((st.effective_context, st.kv_cache_type)) + 1
    except ValueError:
        idx = 1
    _RESTART_FROM[scfg.key()] = min(idx, len(plan) - 1)
    _SERVER.stop()


def _route(kind: str) -> str:
    return {"chat": "/v1/chat/completions", "completion": "/v1/completions", "models": "/v1/models"}[kind]


async def handler(job: dict) -> AsyncIterator[Any]:
    started = time.time()
    job_input = job.get("input") or {}
    warnings: list[str] = []
    scfg: ServerConfig | None = None
    try:
        rcfg = resolve_request(job_input)
        scfg = resolve_server_config(job_input, _gpu().get("vram_total_gib"))
    except ConfigError as exc:
        yield _error("BAD_INPUT", str(exc), scfg, started, warnings)
        return
    warnings += scfg.warnings + rcfg.warnings
    if not has_volume():
        warnings.append(f"no network volume: weights are cached in {cache_root()} and re-downloaded on every "
                        "cold start (attach a network volume to cache them under /runpod-volume/qwen-longctx)")

    if rcfg.dry_run:
        yield {"status": "ok", "dry_run": True, "config": scfg.public(),
               "ladder": plan_ladder(scfg.context, scfg.kv_cache_type, scfg.min_context)
               if scfg.context != "auto" else "auto",
               "worker": worker_block(scfg, started, warnings)}
        return
    if not _gpu().get("gpu"):
        yield _error("NO_GPU", "no CUDA GPU visible to the worker", scfg, started, warnings)
        return

    try:
        load_info = await asyncio.to_thread(ensure_server, scfg, warnings)
    except (ModelError, ServerError) as exc:
        log.error("setup failed: %s", redact(traceback.format_exc()))
        yield _error(exc.code, str(exc), scfg, started, warnings, getattr(exc, "details", None))
        return
    except ConfigError as exc:
        yield _error("BAD_INPUT", str(exc), scfg, started, warnings)
        return
    except Exception as exc:  # noqa: BLE001
        log.error("setup crashed: %s", redact(traceback.format_exc()))
        yield _error("SETUP_CRASHED", f"{type(exc).__name__}: {exc}", scfg, started, warnings)
        return

    if rcfg.kind == "models":
        st = _SERVER.state
        out = {"object": "list", "data": [{"id": "qwen-longctx", "object": "model", "owned_by": "prilojject",
                                           "meta": {"file": scfg.model_file, "n_ctx": st.effective_context if st else None}}]}
        out["worker"] = worker_block(scfg, started, warnings, load_info)
        yield out
        return

    body = apply_request_defaults(rcfg.body, rcfg.kind)
    path = _route(rcfg.kind)

    if not rcfg.stream:
        body["stream"] = False
        try:
            with VramPeak() as vp:
                resp = await asyncio.to_thread(_SERVER.request, path, body, rcfg.request_timeout_s)
        except ServerError as exc:
            if exc.code in {"OOM_DURING_REQUEST", "SERVER_DIED"}:
                _mark_request_oom(scfg)
            yield _error(exc.code, str(exc), scfg, started, warnings, exc.details)
            return
        extra = dict(load_info, request_vram_peak_gib=vp.peak_gib)
        msg = ((resp.get("choices") or [{}])[0] or {}).get("message") or {}
        if rcfg.kind == "chat" and not (msg.get("content") or "").strip() and (msg.get("reasoning_content") or "").strip():
            warnings.append("content is empty but reasoning_content is present: raise max_tokens or set "
                            "chat_template_kwargs.enable_thinking=false")
        resp["worker"] = worker_block(scfg, started, warnings, extra)
        yield resp
        return

    # ---- streaming: a thread reads SSE, we batch chunks to limit Runpod stream posts ----
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def pump() -> None:
        try:
            for payload in _SERVER.stream(path, body, rcfg.request_timeout_s):
                loop.call_soon_threadsafe(q.put_nowait, payload)
        except BaseException as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(q.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, sentinel)

    threading.Thread(target=pump, daemon=True).start()
    held: str | None = None  # hold the last chunk to attach the worker block before [DONE]
    batch: list[Any] = []
    last_flush = time.time()

    def fmt(payload: str) -> Any:
        if rcfg.openai:
            return f"data: {payload}\n\n"
        return json.loads(payload) if payload.startswith("{") else payload

    def flush() -> Any:
        nonlocal batch, last_flush
        out = "".join(batch) if rcfg.openai else list(batch)
        batch = []
        last_flush = time.time()
        return out

    while True:
        item = await q.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            exc = item
            code = getattr(exc, "code", "STREAM_ERROR")
            if code in {"OOM_DURING_REQUEST", "SERVER_DIED"}:
                _mark_request_oom(scfg)
            if batch:
                yield flush()
            yield _error(code, str(exc), scfg, started, warnings, getattr(exc, "details", None))
            return
        if item == "[DONE]":
            if held is not None:
                try:
                    d = json.loads(held)
                    d["worker"] = worker_block(scfg, started, warnings, load_info)
                    held = json.dumps(d)
                except ValueError:
                    pass
                batch.append(fmt(held))
                held = None
            batch.append(fmt("[DONE]") if rcfg.openai else "[DONE]")
            continue
        if held is not None:
            batch.append(fmt(held))
        held = item
        if len(batch) >= STREAM_FLUSH_N or time.time() - last_flush >= STREAM_FLUSH_S:
            yield flush()
    if held is not None:
        batch.append(fmt(held))
    if batch:
        yield flush()


def concurrency_modifier(_current: int) -> int:
    return _parallel()


async def _local(payload: dict) -> list:
    return [x async for x in handler(payload)]


if __name__ == "__main__":
    if "--test_input" in sys.argv:
        raw = sys.argv[sys.argv.index("--test_input") + 1]
        payload = json.load(open(raw[1:], encoding="utf-8")) if raw.startswith("@") else json.loads(raw)
        if "input" not in payload:
            payload = {"input": payload}
        payload.setdefault("id", "local-test")
        print(json.dumps(asyncio.run(_local(payload)), indent=2, default=str))
    else:
        import runpod

        runpod.serverless.start({
            "handler": handler,
            "return_aggregate_stream": True,
            "concurrency_modifier": concurrency_modifier,
        })
