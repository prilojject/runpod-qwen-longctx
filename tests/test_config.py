import pytest

from src.config import (ConfigError, apply_request_defaults, plan_ladder, resolve_request,
                        resolve_server_config, round_ctx)
from src.presets import MAX_POSITION_EMBEDDINGS, PRESETS, auto_preset


def test_auto_preset_by_vram():
    assert auto_preset(23.6) == "p1-24gb-9b-256k"
    assert auto_preset(None) == "p1-24gb-9b-256k"
    assert auto_preset(44.4) == "p3-48gb-35b-256k"


def test_presets_pinned():
    for p in PRESETS.values():
        assert len(p.revision) == 40 and len(p.sha256) == 64 and p.filename.endswith(".gguf")
        assert "mmproj" not in p.filename
    assert PRESETS["smoke"].context == 16384


def test_priority_input_over_env_over_preset(monkeypatch):
    c = resolve_server_config({}, 24.0)
    assert (c.preset, c.context, c.kv_cache_type) == ("p1-24gb-9b-256k", 262144, "f16")
    monkeypatch.setenv("PRESET", "p2-24gb-35b-128k")
    monkeypatch.setenv("CONTEXT", "65536")
    c = resolve_server_config({}, 24.0)
    assert (c.preset, c.context, c.kv_cache_type) == ("p2-24gb-35b-128k", 65536, "q8_0")
    c = resolve_server_config({"preset": "smoke", "context": 8192, "kv_cache_type": "q8_0"}, 24.0)
    assert (c.preset, c.context, c.kv_cache_type, c.min_context) == ("smoke", 8192, "q8_0", 8192)


def test_context_capped_at_max_position_embeddings():
    c = resolve_server_config({"context": 1048576}, 24.0)
    assert c.context == MAX_POSITION_EMBEDDINGS
    assert any("capped" in w for w in c.warnings)


def test_context_auto_and_rounding():
    assert resolve_server_config({"context": "auto"}, 24.0).context == "auto"
    assert resolve_server_config({"context": 100000}, 24.0).context == 99328
    assert round_ctx(1500) == 1024


def test_custom_only_via_env(monkeypatch):
    with pytest.raises(ConfigError):
        resolve_server_config({"preset": "custom"}, 24.0)
    monkeypatch.setenv("PRESET", "custom")
    with pytest.raises(ConfigError):
        resolve_server_config({}, 24.0)
    monkeypatch.setenv("MODEL_REPO", "org/repo")
    monkeypatch.setenv("MODEL_FILE", "../../etc/passwd")
    monkeypatch.setenv("MODEL_REVISION", "main")
    with pytest.raises(ConfigError):
        resolve_server_config({}, 24.0)
    monkeypatch.setenv("MODEL_FILE", "model.gguf")
    c = resolve_server_config({}, 24.0)
    assert c.sha256 is None and any("not a full commit SHA" in w for w in c.warnings)


def test_bad_values():
    with pytest.raises(ConfigError):
        resolve_server_config({"preset": "nope"}, 24.0)
    with pytest.raises(ConfigError):
        resolve_server_config({"kv_cache_type": "q2"}, 24.0)


def test_ladder_f16_256k():
    assert plan_ladder(262144, "f16", 16384) == [
        (262144, "f16"), (262144, "q8_0"), (131072, "q8_0"), (65536, "q8_0"), (32768, "q8_0"), (16384, "q8_0")]


def test_ladder_q8_128k_and_limits():
    assert plan_ladder(131072, "q8_0", 16384) == [(131072, "q8_0"), (65536, "q8_0"), (32768, "q8_0"), (16384, "q8_0")]
    assert len(plan_ladder(262144, "f16", 1024)) == 6
    assert plan_ladder(16384, "q8_0", 16384) == [(16384, "q8_0")]
    assert plan_ladder(99328, "q8_0", 16384)[-1] == (16384, "q8_0")


def test_request_openai_route():
    r = resolve_request({"openai_route": "/v1/chat/completions",
                         "openai_input": {"messages": [{"role": "user", "content": "hi"}], "stream": True}})
    assert r.openai and r.kind == "chat" and r.stream
    r = resolve_request({"openai_route": "/v1/models"})
    assert r.kind == "models" and not r.stream
    with pytest.raises(ConfigError):
        resolve_request({"openai_route": "/v1/embeddings", "openai_input": {}})


def test_request_native_strips_server_keys():
    r = resolve_request({"messages": [{"role": "user", "content": "x"}], "preset": "smoke", "context": 4096})
    assert not r.openai and r.kind == "chat"
    assert "preset" not in r.body and "context" not in r.body
    assert resolve_request({"prompt": "x"}).kind == "completion"
    with pytest.raises(ConfigError):
        resolve_request({"foo": 1})


def test_tokens_rejected_in_input():
    for bad in ({"messages": [{"role": "user", "content": "x"}], "hf_token": "abc"},
                {"openai_route": "/v1/chat/completions", "openai_input": {"messages": [], "api_key": "k"}}):
        with pytest.raises(ConfigError):
            resolve_request(bad)


def test_thinking_defaults(monkeypatch):
    b = apply_request_defaults({"messages": []}, "chat")
    assert b["chat_template_kwargs"]["enable_thinking"] is False
    assert b["temperature"] == 0.7 and b["top_p"] == 0.8 and b["max_tokens"] == 2048
    monkeypatch.setenv("ENABLE_THINKING", "true")
    b = apply_request_defaults({"messages": []}, "chat")
    assert b["chat_template_kwargs"]["enable_thinking"] is True and b["temperature"] == 1.0
    b = apply_request_defaults({"messages": [], "temperature": 0,
                                "chat_template_kwargs": {"enable_thinking": False}}, "chat")
    assert b["chat_template_kwargs"]["enable_thinking"] is False and b["temperature"] == 0
