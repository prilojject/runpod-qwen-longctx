"""Pinned presets: GGUF repo @ revision SHA + file sha256 (LFS oid).

All weights: Apache-2.0, gated=false (checked via the HF API on 2026-11-24).
Model weights by Qwen (Alibaba Cloud); GGUF quantization by Unsloth.
Base models: Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a,
Qwen/Qwen3.6-35B-A3B @ 345d467b117d1fd5eda9b879a5988de9ae5f47a7,
Qwen/Qwen3.6-27B @ 1cda14dd71a3f47115850eb790f416f165da1cb5.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_POSITION_EMBEDDINGS = 262144

REPO_9B = "unsloth/Qwen3.5-9B-GGUF"
REV_9B = "3885219b6810b007914f3a7950a8d1b469d598a5"
REPO_35B = "unsloth/Qwen3.6-35B-A3B-GGUF"
REV_35B = "a483e9e6cbd595906af30beda3187c2663a1118c"
REPO_27B = "unsloth/Qwen3.6-27B-GGUF"
REV_27B = "82d411acf4a06cfb8d9b073a5211bf410bfc29bf"


@dataclass(frozen=True)
class Preset:
    name: str
    repo: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str
    context: int
    kv_cache_type: str
    family: str  # "qwen3.5" | "qwen3.6"
    note: str = ""


PRESETS: dict[str, Preset] = {
    "smoke": Preset(
        "smoke", REPO_9B, REV_9B, "Qwen3.5-9B-Q4_K_M.gguf", 5680522464,
        "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
        16384, "f16", "qwen3.5", "Quick test only",
    ),
    "p1-24gb-9b-256k": Preset(
        "p1-24gb-9b-256k", REPO_9B, REV_9B, "Qwen3.5-9B-Q8_0.gguf", 9527502048,
        "809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4",
        262144, "f16", "qwen3.5",
    ),
    "p2-24gb-35b-128k": Preset(
        "p2-24gb-35b-128k", REPO_35B, REV_35B, "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", 17730509792,
        "649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3",
        131072, "q8_0", "qwen3.6",
    ),
    "p3-48gb-35b-256k": Preset(
        "p3-48gb-35b-256k", REPO_35B, REV_35B, "Qwen3.6-35B-A3B-UD-Q6_K.gguf", 29308320736,
        "4fe53b148b46f9b88830e2a3055c5b15c3a4d1e3ddc9a1384a108d8b9d59f043",
        262144, "f16", "qwen3.6",
    ),
    "p4-48gb-27b-256k": Preset(
        "p4-48gb-27b-256k", REPO_27B, REV_27B, "Qwen3.6-27B-UD-Q5_K_XL.gguf", 20038256864,
        "ac310abf2895aa397121bad6c0be89466af41f0f1606a21c1131b110eeb19d0e",
        262144, "f16", "qwen3.6", "not benchmarked yet",
    ),
    "p5-48gb-35b-q8-256k": Preset(
        "p5-48gb-35b-q8-256k", REPO_35B, REV_35B, "Qwen3.6-35B-A3B-Q8_0.gguf", 36903140320,
        "d1a395809f65a43a13ad119eb4e7acdef1ac6d68120f39902c8ab96e72794a59",
        262144, "q8_0", "qwen3.6", "not benchmarked yet",
    ),
}

PRESET_NAMES = ("auto", *PRESETS.keys(), "custom")
KV_TYPES = ("f16", "q8_0", "q4_0")

# Qwen3.6 model card sampling defaults.
SAMPLING_NON_THINKING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
SAMPLING_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}

AUTO_48GB_THRESHOLD_GIB = 40.0


def auto_preset(vram_total_gib: float | None) -> str:
    """< 40 GiB -> P1 (works on any 24 GB pool), otherwise P3."""
    if vram_total_gib is not None and vram_total_gib >= AUTO_48GB_THRESHOLD_GIB:
        return "p3-48gb-35b-256k"
    return "p1-24gb-9b-256k"
