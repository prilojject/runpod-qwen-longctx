"""Download exactly one .gguf by revision and verify sha256 once (marker file).

The download + hash runs in a child process so a stuck transfer can be
killed when the load deadline passes (no hung workers).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Callable

from .paths import cache_root, model_path
from .secret_utils import redact

MARKER_SUFFIX = ".sha256-ok"


class ModelError(RuntimeError):
    def __init__(self, message: str, code: str = "MODEL_ERROR"):
        super().__init__(message)
        self.code = code


def sha256_file(path: str, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def marker_path(path: str) -> str:
    return path + MARKER_SUFFIX


def marker_ok(path: str, sha256: str | None, size: int | None) -> bool:
    """True if a previous run already verified this exact file."""
    mp = marker_path(path)
    if not (os.path.isfile(path) and os.path.isfile(mp)):
        return False
    try:
        with open(mp, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return False
    st = os.stat(path)
    if data.get("size") != st.st_size:
        return False
    if size is not None and st.st_size != size:
        return False
    if sha256 is not None and data.get("sha256") != sha256:
        return False
    return True


def write_marker(path: str, sha256: str | None) -> None:
    with open(marker_path(path), "w", encoding="utf-8") as fh:
        json.dump({"sha256": sha256, "size": os.stat(path).st_size, "verified_at": int(time.time())}, fh)


def _download_and_verify(repo: str, filename: str, revision: str, sha256: str | None, size: int | None) -> dict:
    """Child-process body. Returns {"path": ..., "downloaded": bool, "hashed": bool}."""
    os.environ.setdefault("HF_HOME", os.path.join(cache_root(), "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    dest = model_path(repo, revision, filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if marker_ok(dest, sha256, size):
        return {"path": dest, "downloaded": False, "hashed": False}

    downloaded = False
    for attempt in (1, 2):
        if not os.path.isfile(dest) or (size is not None and os.path.getsize(dest) != size):
            need = (size or 0) * 1.03
            free = shutil.disk_usage(os.path.dirname(dest)).free
            if need and free < need:
                raise ModelError(
                    f"not enough disk for {filename}: need ~{need / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB "
                    "(raise container disk or attach a bigger network volume)", "DISK_FULL")
            from huggingface_hub import hf_hub_download

            token = (os.environ.get("HF_TOKEN") or "").strip() or None
            if token and token.startswith("{{"):
                token = None
            hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                            local_dir=os.path.dirname(dest), token=token)
            downloaded = True
        if sha256 is None:
            write_marker(dest, None)
            return {"path": dest, "downloaded": downloaded, "hashed": False}
        got = sha256_file(dest)
        if got == sha256:
            write_marker(dest, sha256)
            return {"path": dest, "downloaded": downloaded, "hashed": True}
        os.remove(dest)
        if attempt == 2:
            raise ModelError(f"sha256 mismatch for {filename}: expected {sha256}, got {got}", "SHA256_MISMATCH")
    raise ModelError("unreachable")


def ensure_model(repo: str, filename: str, revision: str, sha256: str | None, size: int | None,
                 timeout_s: float, log: Callable[[str], None] = print) -> dict:
    """Make sure the file is on disk and verified. Kills the child on timeout."""
    dest = model_path(repo, revision, filename)
    if marker_ok(dest, sha256, size):
        return {"path": dest, "downloaded": False, "hashed": False}
    log(f"downloading {repo}@{revision[:12]} {filename} into {os.path.dirname(dest)}")
    payload = json.dumps({"repo": repo, "filename": filename, "revision": revision, "sha256": sha256, "size": size})
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()
    try:
        env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1")
        proc = subprocess.run([sys.executable, "-m", "src.models", payload], cwd=root, stdout=subprocess.PIPE,
                              stderr=None, text=True, timeout=max(30, timeout_s), env=env)
    except subprocess.TimeoutExpired as exc:
        raise ModelError(f"download/verify of {filename} did not finish within {int(timeout_s)} s "
                         "(raise LOAD_TIMEOUT_S or use a network volume)", "DOWNLOAD_TIMEOUT") from exc
    out = (proc.stdout or "").strip().splitlines()
    last = out[-1] if out else ""
    try:
        res = json.loads(last)
    except ValueError:
        res = {"error": "download failed (see worker logs)", "code": "DOWNLOAD_FAILED"}
    if proc.returncode != 0 or "error" in res:
        raise ModelError(str(res.get("error"))[:800], str(res.get("code") or "DOWNLOAD_FAILED"))
    res["seconds"] = round(time.time() - t0, 1)
    log(f"model ready: {filename} downloaded={res['downloaded']} hashed={res['hashed']} in {res['seconds']} s")
    return res


if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    try:
        result = _download_and_verify(args["repo"], args["filename"], args["revision"], args["sha256"], args["size"])
        print(json.dumps(result))
    except ModelError as exc:
        print(json.dumps({"error": redact(str(exc)), "code": exc.code}))
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        msg = redact(f"{type(exc).__name__}: {exc}")
        print(json.dumps({"error": msg[:800], "code": "DOWNLOAD_FAILED"}))
        sys.exit(1)
