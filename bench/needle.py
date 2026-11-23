"""Needle-in-a-haystack benchmark against a deployed Runpod endpoint.

    RUNPOD_API_KEY=... python bench/needle.py --endpoint <ID> --plan p1 --out bench/raw/p1.jsonl

Each case is one async /run job (temperature 0, max_tokens 32, thinking off,
cache_prompt false so every prompt is prefilled from scratch). The script polls
/status, cancels a job that exceeds --job-timeout, and records TTFT (llama.cpp
timings.prompt_ms), prefill/decode tok/s, VRAM peak and the worker block.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haystack import make_code, messages  # noqa: E402

DEPTHS = (0.1, 0.5, 0.9)
PLANS: dict[str, list[dict]] = {
    "smoke": [{"preset": "smoke", "tokens": 2000, "depth": 0.5}],
    "p1": [{"preset": "p1-24gb-9b-256k", "tokens": t, "depth": d} for t in (63000, 126000) for d in DEPTHS]
    + [{"preset": "p1-24gb-9b-256k", "tokens": 126000, "depth": 0.5, "kv_cache_type": "q8_0"}],
    "p2": [{"preset": "p2-24gb-35b-128k", "tokens": t, "depth": d} for t in (63000, 126000) for d in DEPTHS],
    "p2-256k": [{"preset": "p2-24gb-35b-128k", "tokens": 2000, "depth": 0.5, "context": 262144}],
    "p2-fallback": [{"preset": "p2-24gb-35b-128k", "tokens": 2000, "depth": 0.5, "context": 262144,
                     "kv_cache_type": "f16"}],
    "p2-256k-240k": [{"preset": "p2-24gb-35b-128k", "tokens": 240000, "depth": 0.5, "context": 262144}],
    "p4-on-24gb": [{"preset": "p4-48gb-27b-256k", "tokens": 2000, "depth": 0.5}],
    "p4-on-24gb-128k": [{"preset": "p4-48gb-27b-256k", "tokens": 126000, "depth": 0.5}],
    "p4-load": [{"preset": "p4-48gb-27b-256k", "tokens": 2000, "depth": 0.5}],
    "p3": [{"preset": "p3-48gb-35b-256k", "tokens": 126000, "depth": d} for d in DEPTHS],
    "p3-240k": [{"preset": "p3-48gb-35b-256k", "tokens": 240000, "depth": 0.5}],
}


def call(endpoint: str, method: str, path: str, body: dict | None = None, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        f"https://api.runpod.ai/v2/{endpoint}/{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]},
        method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode(errors="replace")[:500]}
    except Exception as e:  # noqa: BLE001
        return {"exc": f"{type(e).__name__}: {e}"}


def run_case(endpoint: str, case: dict, seed: int, job_timeout: int) -> dict:
    code = make_code(seed)
    inp = {"preset": case["preset"], "messages": messages(case["tokens"], case["depth"], code, seed=seed),
           "max_tokens": 32, "temperature": 0, "cache_prompt": False,
           "chat_template_kwargs": {"enable_thinking": False}}
    for k in ("context", "kv_cache_type"):
        if k in case:
            inp[k] = case[k]
    t0 = time.time()
    sub = call(endpoint, "POST", "run", {"input": inp})
    jid = sub.get("id")
    rec = {"case": case, "code": code, "submitted": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    if not jid:
        rec["error"] = f"submit failed: {sub}"
        return rec
    st: dict = {}
    while True:
        time.sleep(5)
        st = call(endpoint, "GET", f"status/{jid}")
        status = st.get("status")
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        if time.time() - t0 > job_timeout:
            call(endpoint, "POST", f"cancel/{jid}")
            rec["error"] = f"cancelled by watchdog after {job_timeout}s (last status {status})"
            return rec
    rec["status"] = st.get("status")
    rec["delay_ms"] = st.get("delayTime")
    rec["execution_ms"] = st.get("executionTime")
    rec["wall_s"] = round(time.time() - t0, 1)
    out = st.get("output")
    if isinstance(out, list) and out:
        out = out[0]
    if st.get("status") != "COMPLETED" or not isinstance(out, dict) or "choices" not in out:
        rec["error"] = str(st.get("error") or out)[:3000]
        return rec
    msg = (out["choices"][0].get("message") or {})
    content = (msg.get("content") or "").strip()
    t = out.get("timings") or {}
    w = out.get("worker") or {}
    rec.update({
        "answer": content[:200],
        "found": code in content,
        "prompt_tokens": (out.get("usage") or {}).get("prompt_tokens"),
        "cache_n": t.get("cache_n"),
        "ttft_s": round(t["prompt_ms"] / 1000, 2) if t.get("prompt_ms") else None,
        "prefill_tps": round(t.get("prompt_per_second") or 0, 1),
        "decode_tps": round(t.get("predicted_per_second") or 0, 1),
        "gpu": w.get("gpu"),
        "effective_context": w.get("effective_context"),
        "kv_cache_type": w.get("kv_cache_type"),
        "fallback_steps": w.get("fallback_steps"),
        "request_vram_peak_gib": w.get("request_vram_peak_gib"),
        "load_vram_peak_gib": w.get("load_vram_peak_gib"),
        "memory_mib": w.get("memory_mib"),
        "cold_start": w.get("cold_start"),
        "download_seconds": w.get("download_seconds"),
        "load_seconds": w.get("load_seconds"),
        "warnings": w.get("warnings"),
    })
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--plan", required=True, choices=sorted(PLANS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--job-timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ok = True
    for i, case in enumerate(PLANS[args.plan]):
        rec = run_case(args.endpoint, case, args.seed + i, args.job_timeout)
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        brief = {k: rec.get(k) for k in ("status", "found", "answer", "prompt_tokens", "ttft_s", "prefill_tps",
                                         "decode_tps", "request_vram_peak_gib", "effective_context",
                                         "kv_cache_type", "gpu", "wall_s", "error")}
        print(time.strftime("%H:%M:%S"), args.plan, i, json.dumps(case), json.dumps(brief)[:1500], flush=True)
        if rec.get("error"):
            ok = False
            if "watchdog" in rec["error"] or "submit failed" in rec["error"]:
                break
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
