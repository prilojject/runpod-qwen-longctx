import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_KEYS = ("PRESET", "CONTEXT", "KV_CACHE_TYPE", "MIN_CONTEXT", "PARALLEL", "ENABLE_THINKING", "MAX_TOKENS_DEFAULT",
         "MODEL_REPO", "MODEL_FILE", "MODEL_REVISION", "LOAD_TIMEOUT_S", "HF_TOKEN")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    yield
