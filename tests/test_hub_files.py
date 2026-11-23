import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    with open(os.path.join(ROOT, ".runpod", name), encoding="utf-8") as fh:
        return json.load(fh)


def test_tests_json_smoke_needle():
    t = load("tests.json")
    assert len(t["tests"]) == 1
    test = t["tests"][0]
    assert test["input"]["preset"] == "smoke"
    raw = json.dumps(test)
    assert "MANGO-7421" in raw
    assert t["config"]["gpuTypeId"] == "NVIDIA GeForce RTX 4090"
    assert all(v >= "12.8" for v in t["config"]["allowedCudaVersions"]) and "12.7" not in t["config"]["allowedCudaVersions"]


def test_hub_json():
    h = load("hub.json")
    c = h["config"]
    assert h["type"] == "serverless" and c["gpuCount"] == 1 and c["containerDiskInGb"] == 60
    assert "12.8" in c["allowedCudaVersions"] and "12.7" not in c["allowedCudaVersions"]
    assert "-NVIDIA B200" in c["gpuIds"] and "-NVIDIA B300 SXM6 AC" in c["gpuIds"]
    keys = {e["key"] for e in c["env"]}
    assert {"PRESET", "CONTEXT", "MIN_CONTEXT", "KV_CACHE_TYPE", "ENABLE_THINKING"} <= keys
    presets = {p["defaults"]["PRESET"] for p in c["presets"]}
    assert presets == {"p1-24gb-9b-256k", "p2-24gb-35b-128k", "p3-48gb-35b-256k", "p4-48gb-27b-256k", "smoke"}
    hf = [e for e in c["env"] if e["key"] == "HF_TOKEN"][0]
    assert hf["input"]["default"] == ""
