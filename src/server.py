"""llama-server on 127.0.0.1:8080 with an OOM fallback ladder.

The built-in web UI is disabled (--no-webui) and the port is loopback-only.
"""

from __future__ import annotations

import collections
import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import requests

from .config import MAX_ATTEMPTS, SERVER_HOST, SERVER_PORT, plan_ladder
from .paths import LLAMA_SERVER
from .secret_utils import redact

OOM_PATTERNS = (
    "out of memory",
    "cudamalloc failed",
    "failed to allocate",
    "failed to create context",
    "unable to allocate",
    "erroroutofdevicememory",
    "cuda_error_out_of_memory",
)
_CKPT_RE = re.compile(r"created context checkpoint .*size\s*=\s*([\d.]+)\s*MiB")
_BUF_RE = re.compile(r"(\S+)\s+(model|KV|RS|compute|output|recurrent)\s+buffer size\s*=\s*([\d.]+)\s*MiB", re.I)


class ServerError(RuntimeError):
    def __init__(self, message: str, code: str = "SERVER_ERROR", details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def is_oom_text(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in OOM_PATTERNS)


def build_args(model_path: str, ctx: int | str, kv: str, parallel: int, min_context: int,
               host: str = SERVER_HOST, port: int = SERVER_PORT) -> list[str]:
    if host != "127.0.0.1":
        raise ServerError(f"refusing to bind llama-server on non-loopback host {host!r}", "CONFIG")
    args = [
        LLAMA_SERVER, "-m", model_path,
        "-ctk", kv, "-ctv", kv,
        "-fa", "on", "-ngl", "999", "-np", str(parallel),
        "--host", host, "--port", str(port),
        "--jinja", "--reasoning-format", "deepseek", "--metrics",
        "--no-webui", "-ub", "512", "-b", "2048",
        "--timeout", "3600", "--alias", "qwen-longctx",
        # -lv 4 prints buffer sizes (model / KV / RS / compute); prompts are never logged.
        "--log-verbosity", "4", "--log-colors", "off",
    ]
    if ctx == "auto":
        args += ["--fit", "on", "--fit-ctx", str(min_context), "--fit-target", "1024"]
    else:
        args += ["-c", str(int(ctx)), "--fit", "off"]
    if parallel > 1:
        args += ["--kv-unified"]
    return args


@dataclass
class LoadedState:
    model_path: str
    effective_context: int
    kv_cache_type: str
    fallback_steps: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, float] = field(default_factory=dict)
    load_seconds: float = 0.0
    key: tuple = ()


class LlamaServer:
    def __init__(self, log: Callable[[str], None] = print, host: str = SERVER_HOST, port: int = SERVER_PORT):
        self.log = log
        self.host = host
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.lines: collections.deque[str] = collections.deque(maxlen=400)
        self.memory: dict[str, float] = {}
        self.oom_seen = False
        self.state: LoadedState | None = None
        self._reader: threading.Thread | None = None

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ---- process control -------------------------------------------------
    def _read(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            self.lines.append(line)
            if is_oom_text(line):
                self.oom_seen = True
            m = _BUF_RE.search(line)
            if m:
                key = f"{m.group(1)} {m.group(2)}"
                self.memory[key] = round(self.memory.get(key, 0.0) + float(m.group(3)), 2)
                if "CPU_Mapped" in line:
                    continue  # dozens of near-zero lines per load
            c = _CKPT_RE.search(line)
            if c:
                self.memory["context_checkpoint_mib"] = float(c.group(1))
            self.log("[llama-server] " + redact(line)[:2000])

    def _spawn(self, args: list[str]) -> None:
        self.lines.clear()
        self.memory = {}
        self.oom_seen = False
        self.log("starting: " + " ".join(args[1:]))
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     errors="replace", start_new_session=True, env=dict(os.environ))
        self._reader = threading.Thread(target=self._read, args=(self.proc,), daemon=True)
        self._reader.start()

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        self.state = None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait(timeout=10)
        if self._reader:
            self._reader.join(timeout=5)

    def _wait_healthy(self, deadline: float) -> tuple[bool, str]:
        """(ok, reason). reason is 'oom' | 'exited' | 'timeout' when not ok."""
        while time.time() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                if self._reader:
                    self._reader.join(timeout=5)
                return False, "oom" if self.oom_seen else "exited"
            if self.oom_seen:
                return False, "oom"
            try:
                r = requests.get(self.base + "/health", timeout=2)
                if r.status_code == 200:
                    return True, "ok"
            except requests.RequestException:
                pass
            time.sleep(0.5)
        return False, "timeout"

    def props_n_ctx(self) -> int | None:
        try:
            r = requests.get(self.base + "/props", timeout=5)
            d = r.json()
            n = (d.get("default_generation_settings") or {}).get("n_ctx")
            return int(n) if n else None
        except Exception:
            return None

    # ---- ladder ------------------------------------------------------------
    def load(self, model_path: str, context: int | str, kv: str, min_context: int, parallel: int,
             deadline: float, key: tuple = (), start_index: int = 0) -> LoadedState:
        """Start llama-server, stepping down KV precision / context on OOM."""
        self.stop()
        t0 = time.time()
        if context == "auto":
            plan: list[tuple[int | str, str]] = [("auto", kv)]
        else:
            plan = list(plan_ladder(int(context), kv, min_context, MAX_ATTEMPTS))
        plan = plan[start_index:] or plan[-1:]
        steps: list[dict[str, Any]] = []
        for i, (ctx, k) in enumerate(plan):
            if time.time() >= deadline:
                break
            self._spawn(build_args(model_path, ctx, k, parallel, min_context, self.host, self.port))
            ok, reason = self._wait_healthy(deadline)
            if ok:
                eff = self.props_n_ctx() if ctx == "auto" else int(ctx)
                if eff is None:
                    eff = self.props_n_ctx() or 0
                self.state = LoadedState(model_path, int(eff), k, steps, dict(self.memory),
                                         round(time.time() - t0, 1), key)
                return self.state
            tail = "\n".join(list(self.lines)[-15:])
            self.stop()
            steps.append({"context": ctx, "kv_cache_type": k, "result": reason})
            if reason == "timeout":
                raise ServerError(f"llama-server did not become healthy before LOAD_TIMEOUT_S "
                                  f"(context={ctx}, kv={k})", "LOAD_TIMEOUT", {"fallback_steps": steps})
            if reason != "oom":
                raise ServerError("llama-server exited during load (not an out-of-memory error): "
                                  + redact(tail)[-1200:], "LOAD_FAILED", {"fallback_steps": steps})
            self.log(f"OOM at context={ctx} kv={k}; stepping down ({i + 1}/{len(plan)})")
        if steps and time.time() >= deadline:
            raise ServerError("ran out of LOAD_TIMEOUT_S while stepping down after OOM", "LOAD_TIMEOUT",
                              {"fallback_steps": steps})
        raise ServerError(
            f"model does not fit on this GPU even at MIN_CONTEXT={min_context} with KV {plan[-1][1]} "
            f"after {len(steps)} attempts; use a bigger GPU or a smaller preset",
            "OOM_MIN_CONTEXT", {"fallback_steps": steps})

    # ---- requests ----------------------------------------------------------
    def request(self, path: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        url = self.base + path
        try:
            if body is None:
                r = requests.get(url, timeout=(5, timeout))
            else:
                r = requests.post(url, json=body, timeout=(5, timeout))
        except requests.Timeout as exc:
            raise ServerError(f"llama-server did not answer within {int(timeout)} s", "REQUEST_TIMEOUT") from exc
        except requests.RequestException as exc:
            raise self._conn_error(exc) from exc
        return self._parse(r)

    def stream(self, path: str, body: dict[str, Any], timeout: float) -> Iterator[str]:
        """Yield raw SSE 'data: ...' payload strings (without the prefix)."""
        url = self.base + path
        try:
            r = requests.post(url, json=body, timeout=(5, timeout), stream=True)
        except requests.Timeout as exc:
            raise ServerError(f"llama-server did not answer within {int(timeout)} s", "REQUEST_TIMEOUT") from exc
        except requests.RequestException as exc:
            raise self._conn_error(exc) from exc
        if r.status_code != 200:
            self._parse(r)
        deadline = time.time() + timeout
        try:
            for raw in r.iter_lines(decode_unicode=True):
                if time.time() > deadline:
                    raise ServerError(f"stream exceeded {int(timeout)} s", "REQUEST_TIMEOUT")
                if not raw or not raw.startswith("data:"):
                    if raw and raw.startswith("error:"):
                        raise self._error_from_payload(raw[6:].strip(), 500)
                    continue
                payload = raw[5:].strip()
                if payload.startswith("{") and '"error"' in payload[:20]:
                    raise self._error_from_payload(payload, 500)
                yield payload
                if payload == "[DONE]":
                    return
        except requests.RequestException as exc:
            raise self._conn_error(exc) from exc
        finally:
            r.close()

    def _conn_error(self, exc: Exception) -> ServerError:
        if not self.alive():
            return ServerError("llama-server process died during the request"
                               + (" (out of memory)" if self.oom_seen else ""),
                               "OOM_DURING_REQUEST" if self.oom_seen else "SERVER_DIED")
        return ServerError(f"llama-server connection error: {type(exc).__name__}", "SERVER_ERROR")

    def _error_from_payload(self, text: str, status: int) -> ServerError:
        try:
            d = json.loads(text)
        except ValueError:
            d = {"error": {"message": text[:600]}}
        err = d.get("error") if isinstance(d, dict) else None
        err = err if isinstance(err, dict) else {"message": str(err or d)[:600]}
        msg = str(err.get("message") or "")
        etype = str(err.get("type") or "")
        if etype == "exceed_context_size_error" or "exceeds the available context size" in msg:
            return ServerError(
                f"prompt is longer than the effective context: {err.get('n_prompt_tokens')} prompt tokens, "
                f"context {err.get('n_ctx')}", "CONTEXT_EXCEEDED",
                {"n_prompt_tokens": err.get("n_prompt_tokens"), "n_ctx": err.get("n_ctx")})
        if is_oom_text(msg) or self.oom_seen:
            return ServerError("out of memory during the request: " + msg[:300], "OOM_DURING_REQUEST")
        return ServerError(f"llama-server HTTP {status}: {msg[:600]}", "BAD_REQUEST" if status < 500 else "SERVER_ERROR")

    def _parse(self, r: requests.Response) -> dict[str, Any]:
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError as exc:
                raise ServerError("llama-server returned non-JSON", "SERVER_ERROR") from exc
        raise self._error_from_payload(r.text, r.status_code)

    def tokenize_count(self, text: str, timeout: float = 120) -> int:
        d = self.request("/tokenize", {"content": text}, timeout)
        return len(d.get("tokens") or [])
