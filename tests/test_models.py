import json
import os

from src import models


def test_marker_roundtrip(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"hello")
    sha = models.sha256_file(str(f))
    assert not models.marker_ok(str(f), sha, 5)
    models.write_marker(str(f), sha)
    assert models.marker_ok(str(f), sha, 5)
    assert not models.marker_ok(str(f), "0" * 64, 5)
    assert not models.marker_ok(str(f), sha, 6)
    f.write_bytes(b"hello!")  # size change invalidates the marker
    assert not models.marker_ok(str(f), sha, None)


def test_ensure_model_skips_when_marker_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_LONGCTX_CACHE", str(tmp_path))
    from src.paths import model_path

    p = model_path("o/r", "a" * 40, "x.gguf")
    os.makedirs(os.path.dirname(p))
    open(p, "wb").write(b"abc")
    sha = models.sha256_file(p)
    models.write_marker(p, sha)
    res = models.ensure_model("o/r", "x.gguf", "a" * 40, sha, 3, timeout_s=5)
    assert res == {"path": p, "downloaded": False, "hashed": False}
    assert json.load(open(p + models.MARKER_SUFFIX))["sha256"] == sha
