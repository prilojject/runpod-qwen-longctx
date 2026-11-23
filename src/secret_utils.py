"""Secret handling: read tokens from the environment, never log them."""

from __future__ import annotations

import logging
import os
import re

# Job-input keys that must never be accepted (tokens belong in Runpod secrets / env).
FORBIDDEN_INPUT_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "openai_api_key",
    "openai-api-key",
    "hf_token",
    "hugging_face_hub_token",
    "huggingface_token",
    "huggingface_hub_token",
    "access_token",
    "authorization",
    "auth",
    "admin_key",
    "admin-key",
    "secret",
    "password",
    "token",
    "runpod_api_key",
    "runpod_api",
}

_SECRET_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "RUNPOD_API_KEY",
    "RUNPOD_WEBHOOK_GET_JOB",
    "RUNPOD_WEBHOOK_POST_OUTPUT",
    "AWS_SECRET_ACCESS_KEY",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
)

_SECRET_ENV_PREFIXES = ("RUNPOD_",)


def get_hf_token() -> str | None:
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    # An unresolved Runpod secret reference means "not configured".
    if not tok or tok.startswith("{{"):
        return None
    return tok


def collect_secrets() -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for key, value in os.environ.items():
        if not value or len(value) < 8 or value.startswith("{{"):
            continue
        if key in _SECRET_ENV_NAMES or key.startswith(_SECRET_ENV_PREFIXES):
            if value not in seen:
                seen.add(value)
                found.append(value)
    return found


class RedactFilter(logging.Filter):
    """Replace any secret value that sneaks into a log record."""

    def __init__(self) -> None:
        super().__init__()
        self._secrets = collect_secrets()

    def filter(self, record: logging.LogRecord) -> bool:
        if self._secrets:
            msg = record.getMessage()
            hit = False
            for s in self._secrets:
                if s in msg:
                    msg = msg.replace(s, "***")
                    hit = True
            if hit:
                record.msg, record.args = msg, ()
        return True


def redact(text: str) -> str:
    for s in collect_secrets():
        text = text.replace(s, "***")
    return text


_KEY_RE = re.compile(r"(?i)(api[_-]?key|authorization|token|secret|password)\s*[:=]\s*([^\s,;]+)")


def redact_patterns(text: str) -> str:
    """Redact both known secret values and obvious key=value patterns."""
    text = redact(text)
    return _KEY_RE.sub(lambda m: f"{m.group(1)}=***", text)


def find_forbidden_input_keys(payload: object, prefix: str = "") -> list[str]:
    """Return dotted paths of forbidden secret-like keys in a job input."""
    hits: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).strip().lower() in FORBIDDEN_INPUT_KEYS:
                hits.append(path)
            hits.extend(find_forbidden_input_keys(value, path))
    elif isinstance(payload, list):
        for i, value in enumerate(payload):
            hits.extend(find_forbidden_input_keys(value, f"{prefix}[{i}]"))
    return hits
