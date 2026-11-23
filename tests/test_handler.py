import asyncio

import handler


def run(inp):
    async def go():
        return [x async for x in handler.handler({"id": "t", "input": inp})]
    return asyncio.run(go())


def test_dry_run():
    out = run({"dry_run": True, "preset": "smoke"})
    assert out[0]["status"] == "ok"
    assert out[0]["worker"]["preset"] == "smoke"
    assert "effective_context" in out[0]["worker"]


def test_bad_input_fails_job():
    out = run({"hf_token": "x", "messages": [{"role": "user", "content": "x"}]})
    assert "error" in out[0] and out[0]["error"].startswith("BAD_INPUT")


def test_no_gpu_fails_cleanly(monkeypatch):
    monkeypatch.setattr(handler, "_GPU", {"gpu": None, "vram_total_gib": None})
    monkeypatch.setattr(handler, "gpu_info", lambda: {"gpu": None, "vram_total_gib": None})
    out = run({"messages": [{"role": "user", "content": "x"}]})
    assert out[0]["error"].startswith("NO_GPU")
