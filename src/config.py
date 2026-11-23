"""Resolve worker config. Priority: job input > env > preset defaults."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from .presets import KV_TYPES, MAX_POSITION_EMBEDDINGS, PRESETS, PRESET_NAMES, auto_preset
from .secret_utils import find_forbidden_input_keys

DEFAULT_MIN_CONTEXT = 16384
DEFAULT_LOAD_TIMEOUT_S = 900
DEFAULT_REQUEST_TIMEOUT_S = 1800
DEFAULT_MAX_TOKENS = 2048
MAX_ATTEMPTS = 6
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080

OPENAI_ROUTES = ("/v1/chat/completions", "/v1/completions", "/v1/models")

# Server-level knobs a job may override (they restart llama-server).
INPUT_SERVER_KEYS = ("preset", "context", "kv_cache_type", "min_context")


class ConfigError(ValueError):
    """Bad job input or env."""


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name, "")
    val = val.strip() if isinstance(val, str) else ""
    if val.startswith("{{"):  # unresolved Runpod secret/template reference
        return default
    return val or default


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on", "y"}:
        return True
    if s in {"0", "false", "no", "off", "n"}:
        return False
    raise ConfigError(f"not a boolean: {value!r}")


def parse_int(value: Any, name: str, default: int | None = None, lo: int | None = None, hi: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        n = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer, got {value!r}") from exc
    if lo is not None and n < lo:
        raise ConfigError(f"{name} must be >= {lo}, got {n}")
    if hi is not None and n > hi:
        raise ConfigError(f"{name} must be <= {hi}, got {n}")
    return n


def round_ctx(n: int) -> int:
    """Round a context size down to a multiple of 1024 (min 1024)."""
    return max(1024, (int(n) // 1024) * 1024)


@dataclass
class ServerConfig:
    """Everything that requires (re)starting llama-server."""

    preset: str
    model_repo: str
    model_file: str
    revision: str
    sha256: str | None
    size_bytes: int | None
    context: int | str  # int tokens or "auto"
    kv_cache_type: str
    min_context: int
    parallel: int
    load_timeout_s: int
    warnings: list[str] = field(default_factory=list)

    def key(self) -> tuple:
        return (self.model_repo, self.model_file, self.revision, self.context,
                self.kv_cache_type, self.min_context, self.parallel)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("warnings", None)
        return d


def resolve_server_config(job_input: dict[str, Any] | None, vram_total_gib: float | None) -> ServerConfig:
    job_input = job_input or {}
    warnings: list[str] = []

    preset = str(job_input.get("preset") or _env("PRESET", "auto")).strip().lower()
    if preset not in PRESET_NAMES:
        raise ConfigError(f"unknown preset {preset!r}; one of {', '.join(PRESET_NAMES)}")
    if preset == "custom" and job_input.get("preset"):
        raise ConfigError("preset=custom can only be set with the PRESET env var, not in job input")
    if preset == "auto":
        preset = auto_preset(vram_total_gib)

    if preset == "custom":
        repo = _env("MODEL_REPO")
        fname = _env("MODEL_FILE")
        rev = _env("MODEL_REVISION")
        if not repo or not fname:
            raise ConfigError("PRESET=custom needs MODEL_REPO and MODEL_FILE")
        if not rev:
            raise ConfigError("PRESET=custom needs MODEL_REVISION (a commit SHA; 'main' works but is not reproducible)")
        validate_repo_file(repo, fname)
        if rev in {"main", "master"} or len(rev) < 40:
            warnings.append(f"MODEL_REVISION={rev!r} is not a full commit SHA: not reproducible")
        sha, size = None, None
        default_ctx, default_kv = MAX_POSITION_EMBEDDINGS, "f16"
        warnings.append("custom GGUF: sha256 verification is disabled")
    else:
        p = PRESETS[preset]
        repo, fname, rev, sha, size = p.repo, p.filename, p.revision, p.sha256, p.size_bytes
        default_ctx, default_kv = p.context, p.kv_cache_type

    raw_ctx = job_input.get("context")
    if raw_ctx in (None, ""):
        raw_ctx = _env("CONTEXT", "")
    if isinstance(raw_ctx, str) and raw_ctx.strip().lower() == "auto":
        context: int | str = "auto"
    else:
        ctx = parse_int(raw_ctx, "context", default_ctx, lo=1024)
        assert ctx is not None
        if ctx > MAX_POSITION_EMBEDDINGS:
            warnings.append(f"context {ctx} is above the model's {MAX_POSITION_EMBEDDINGS} tokens; capped")
            ctx = MAX_POSITION_EMBEDDINGS
        if ctx != round_ctx(ctx):
            warnings.append(f"context {ctx} rounded down to a multiple of 1024")
        context = round_ctx(ctx)

    kv = str(job_input.get("kv_cache_type") or _env("KV_CACHE_TYPE", "") or default_kv).strip().lower()
    if kv not in KV_TYPES:
        raise ConfigError(f"kv_cache_type must be one of {', '.join(KV_TYPES)}, got {kv!r}")
    if kv == "q4_0":
        warnings.append("KV cache q4_0: less precise at long range")

    min_ctx = parse_int(job_input.get("min_context") if job_input.get("min_context") not in (None, "") else _env("MIN_CONTEXT", ""),
                        "min_context", DEFAULT_MIN_CONTEXT, lo=1024, hi=MAX_POSITION_EMBEDDINGS)
    assert min_ctx is not None
    min_ctx = round_ctx(min_ctx)
    if isinstance(context, int) and context < min_ctx:
        min_ctx = context

    parallel = parse_int(_env("PARALLEL", ""), "PARALLEL", 1, lo=1, hi=16)
    load_timeout = parse_int(_env("LOAD_TIMEOUT_S", ""), "LOAD_TIMEOUT_S", DEFAULT_LOAD_TIMEOUT_S, lo=60, hi=7200)
    assert parallel is not None and load_timeout is not None

    return ServerConfig(
        preset=preset, model_repo=repo, model_file=fname, revision=rev, sha256=sha, size_bytes=size,
        context=context, kv_cache_type=kv, min_context=min_ctx, parallel=parallel,
        load_timeout_s=load_timeout, warnings=warnings,
    )


def validate_repo_file(repo: str, fname: str) -> None:
    if repo.count("/") != 1 or any(c in repo for c in " \\:") or ".." in repo:
        raise ConfigError(f"MODEL_REPO must look like owner/name, got {repo!r}")
    if fname.startswith("/") or ".." in fname or "\\" in fname or not fname.endswith(".gguf"):
        raise ConfigError(f"MODEL_FILE must be a relative .gguf file name inside the repo, got {fname!r}")


def plan_ladder(context: int, kv: str, min_context: int, max_attempts: int = MAX_ATTEMPTS) -> list[tuple[int, str]]:
    """OOM fallback plan: f16 -> q8_0 at same ctx, then halve ctx down to min_context.

    Never more than max_attempts steps.
    """
    steps: list[tuple[int, str]] = [(context, kv)]
    if kv == "f16":
        steps.append((context, "q8_0"))
        kv = "q8_0"
    ctx = context
    while len(steps) < max_attempts:
        nxt = round_ctx(ctx // 2)
        if nxt < min_context:
            if ctx > min_context:
                steps.append((min_context, kv))
            break
        steps.append((nxt, kv))
        ctx = nxt
    return steps[:max_attempts]


@dataclass
class RequestConfig:
    kind: str  # "chat" | "completion" | "models"
    openai: bool
    body: dict[str, Any]
    stream: bool
    dry_run: bool
    request_timeout_s: int
    warnings: list[str] = field(default_factory=list)


def resolve_request(job_input: dict[str, Any] | None) -> RequestConfig:
    job_input = job_input or {}
    if not isinstance(job_input, dict):
        raise ConfigError("input must be a JSON object")
    bad = find_forbidden_input_keys(job_input)
    if bad:
        raise ConfigError(f"secrets are not accepted in job input ({', '.join(bad)}); use Runpod secrets/env")

    warnings: list[str] = []
    dry_run = parse_bool(job_input.get("dry_run"), False)
    timeout = parse_int(job_input.get("request_timeout_s"), "request_timeout_s",
                        parse_int(_env("REQUEST_TIMEOUT_S", ""), "REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S, lo=10),
                        lo=10, hi=7200)
    assert timeout is not None

    route = job_input.get("openai_route")
    if route:
        route = "/" + str(route).strip().lstrip("/")
        if not route.startswith("/v1/"):
            route = "/v1" + route if not route.startswith("/v1") else route
        if route not in OPENAI_ROUTES:
            raise ConfigError(f"unsupported openai_route {route!r}; one of {', '.join(OPENAI_ROUTES)}")
        body = job_input.get("openai_input") or {}
        if not isinstance(body, dict):
            raise ConfigError("openai_input must be a JSON object")
        body = dict(body)
        kind = {"/v1/chat/completions": "chat", "/v1/completions": "completion", "/v1/models": "models"}[route]
        openai = True
    else:
        openai = False
        body = {k: v for k, v in job_input.items()
                if k not in {"dry_run", "request_timeout_s", "openai_route", "openai_input", *INPUT_SERVER_KEYS}}
        if "messages" in body:
            kind = "chat"
        elif "prompt" in body:
            kind = "completion"
        elif dry_run:
            kind = "chat"
            body.setdefault("messages", [{"role": "user", "content": "ping"}])
        else:
            raise ConfigError("input needs 'messages' (chat) or 'prompt' (completion), or openai_route + openai_input")

    if kind == "chat":
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ConfigError("messages must be a non-empty list")
    elif kind == "completion":
        if not isinstance(body.get("prompt"), (str, list)):
            raise ConfigError("prompt must be a string (or list)")

    stream = parse_bool(body.get("stream"), False) if kind != "models" else False
    return RequestConfig(kind=kind, openai=openai, body=body, stream=stream, dry_run=dry_run,
                         request_timeout_s=timeout, warnings=warnings)


def apply_request_defaults(body: dict[str, Any], kind: str) -> dict[str, Any]:
    """Fill max_tokens, thinking flag and Qwen sampling defaults the client did not set."""
    from .presets import SAMPLING_NON_THINKING, SAMPLING_THINKING

    out = dict(body)
    out.setdefault("model", "qwen-longctx")
    if out.get("max_tokens") in (None, "") and out.get("max_completion_tokens") in (None, ""):
        out["max_tokens"] = parse_int(_env("MAX_TOKENS_DEFAULT", ""), "MAX_TOKENS_DEFAULT", DEFAULT_MAX_TOKENS, lo=1)
    thinking = parse_bool(_env("ENABLE_THINKING", ""), False)
    if kind == "chat":
        ctk = dict(out.get("chat_template_kwargs") or {})
        if "enable_thinking" in ctk:
            thinking = parse_bool(ctk["enable_thinking"], thinking)
        ctk["enable_thinking"] = thinking
        out["chat_template_kwargs"] = ctk
    defaults = SAMPLING_THINKING if thinking else SAMPLING_NON_THINKING
    for k, v in defaults.items():
        if out.get(k) is None:
            out[k] = v
    return out
