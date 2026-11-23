"""Two-turn prompt-cache check on a hybrid model (context checkpoints).

Turn 1: ~64K document + question. Turn 2: same history + answer + new question
(append-only). Turn 3: edit the middle of the document (forces a re-prefill).
Records timings.cache_n / prompt_n / prompt_ms per turn.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haystack import haystack, make_code  # noqa: E402
from needle import call  # noqa: E402


def job(endpoint: str, inp: dict, timeout: int = 900) -> dict:
    sub = call(endpoint, "POST", "run", {"input": inp})
    jid = sub["id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(3)
        st = call(endpoint, "GET", f"status/{jid}")
        if st.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            out = st.get("output")
            return out[0] if isinstance(out, list) and out else {"error": st.get("error") or out}
    call(endpoint, "POST", f"cancel/{jid}")
    return {"error": "watchdog timeout"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--preset", default="p2-24gb-35b-128k")
    ap.add_argument("--tokens", type=int, default=63000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    code = make_code(4242)
    doc = haystack(a.tokens, 0.5, code, seed=4242)
    base = {"preset": a.preset, "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    m1 = [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": doc + "\n\nWhat is the secret code?"}]
    rows = []
    o1 = job(a.endpoint, dict(base, messages=m1))
    ans = ((o1.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    m2 = m1 + [{"role": "assistant", "content": ans}, {"role": "user", "content": "Who appears on day 3?"}]
    o2 = job(a.endpoint, dict(base, messages=m2))
    edited = doc.replace(" on day 1000.", " on day 1000 (edited).", 1)
    m3 = [m1[0], {"role": "user", "content": edited + "\n\nWhat is the secret code?"}]
    o3 = job(a.endpoint, dict(base, messages=m3))
    for name, o in (("turn1_new", o1), ("turn2_append", o2), ("turn3_edit_middle", o3)):
        t = o.get("timings") or {}
        rows.append({"turn": name, "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
                     "prompt_ms": t.get("prompt_ms"), "gpu": (o.get("worker") or {}).get("gpu"),
                     "error": o.get("error")})
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
