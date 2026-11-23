import time

import pytest

from src import server as srv
from src.server import LlamaServer, ServerError, build_args, is_oom_text


def test_loopback_only_and_no_webui():
    args = build_args("/m.gguf", 262144, "f16", 1, 16384)
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "8080"
    assert "--no-webui" in args and "--fit" in args and args[args.index("--fit") + 1] == "off"
    assert args[args.index("-c") + 1] == "262144"
    with pytest.raises(ServerError):
        build_args("/m.gguf", 4096, "f16", 1, 4096, host="0.0.0.0")


def test_auto_uses_fit():
    args = build_args("/m.gguf", "auto", "f16", 1, 16384)
    assert "-c" not in args and args[args.index("--fit") + 1] == "on"
    assert args[args.index("--fit-ctx") + 1] == "16384"


def test_oom_patterns():
    assert is_oom_text("ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory")
    assert is_oom_text("llama_init_from_model: failed to create context")
    assert not is_oom_text("model loaded")


class FakeServer(LlamaServer):
    """Simulates llama-server: OOM above a fitting (ctx, kv) budget."""

    def __init__(self, fits):
        super().__init__(log=lambda m: None)
        self.fits = fits
        self.tried = []
        self._ok = False

    def _spawn(self, args):
        ctx = int(args[args.index("-c") + 1])
        kv = args[args.index("-ctk") + 1]
        self.tried.append((ctx, kv))
        self._ok = self.fits(ctx, kv)
        self.oom_seen = not self._ok

    def _wait_healthy(self, deadline):
        return (True, "ok") if self._ok else (False, "oom")

    def stop(self):
        self.state = None


def test_ladder_steps_down_until_fit():
    s = FakeServer(lambda ctx, kv: kv == "q8_0" and ctx <= 65536)
    st = s.load("/m.gguf", 262144, "f16", 16384, 1, time.time() + 60)
    assert s.tried == [(262144, "f16"), (262144, "q8_0"), (131072, "q8_0"), (65536, "q8_0")]
    assert st.effective_context == 65536 and st.kv_cache_type == "q8_0"
    assert [x["result"] for x in st.fallback_steps] == ["oom", "oom", "oom"]


def test_ladder_gives_up_with_clear_error():
    s = FakeServer(lambda ctx, kv: False)
    with pytest.raises(ServerError) as ei:
        s.load("/m.gguf", 262144, "f16", 16384, 1, time.time() + 60)
    assert ei.value.code == "OOM_MIN_CONTEXT"
    assert len(s.tried) == 6 and s.tried[-1] == (16384, "q8_0")


def test_non_oom_exit_does_not_retry():
    s = FakeServer(lambda ctx, kv: False)
    s._wait_healthy = lambda deadline: (False, "exited")
    with pytest.raises(ServerError) as ei:
        s.load("/m.gguf", 131072, "q8_0", 16384, 1, time.time() + 60)
    assert ei.value.code == "LOAD_FAILED" and len(s.tried) == 1


def test_context_exceeded_mapping():
    s = LlamaServer(log=lambda m: None)
    e = s._error_from_payload('{"error":{"code":400,"message":"the request exceeds the available context size, '
                              'try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9012,'
                              '"n_ctx":8192}}', 400)
    assert e.code == "CONTEXT_EXCEEDED" and e.details == {"n_prompt_tokens": 9012, "n_ctx": 8192}


def test_buffer_regex():
    m = srv._BUF_RE.search("load_tensors:        CUDA0 model buffer size =  8890.12 MiB")
    assert m and m.group(1) == "CUDA0" and m.group(2) == "model"
    m = srv._BUF_RE.search("llama_memory_recurrent:      CUDA0 RS buffer size =    50.25 MiB")
    assert m and m.group(2) == "RS"
