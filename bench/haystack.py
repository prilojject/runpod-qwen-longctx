"""Deterministic needle-in-a-haystack text (generated, no third-party text)."""

from __future__ import annotations

import random

NAMES = ["Ada", "Bruno", "Chen", "Dara", "Emil", "Farah", "Goran", "Hana", "Ivo", "Jun", "Kira", "Lena", "Milo",
         "Nora", "Oskar", "Pia", "Quinn", "Rosa", "Sami", "Tara", "Uma", "Viktor", "Wren", "Yuki", "Zane"]
VERBS = ["painted", "repaired", "measured", "counted", "moved", "cleaned", "inspected", "sketched", "labeled",
         "weighed", "photographed", "sorted", "polished", "described", "carried"]
ADJS = ["blue", "quiet", "heavy", "narrow", "golden", "wooden", "dusty", "bright", "ancient", "round", "small",
        "green", "silver", "tall", "warm"]
NOUNS = ["lantern", "bridge", "barrel", "window", "ladder", "compass", "kettle", "fence", "boat", "clock",
         "basket", "statue", "wagon", "drum", "mirror"]
PLACES = ["harbor", "orchard", "market", "library", "mill", "station", "garden", "workshop", "square", "valley",
          "tower", "bakery", "pier", "forest", "museum"]

# Measured with the Qwen3.5 tokenizer (llama.cpp /tokenize): ~16.3 tokens per sentence.
TOKENS_PER_SENTENCE = 16.3


def sentence(rng: random.Random, i: int) -> str:
    return (f"{rng.choice(NAMES)} {rng.choice(VERBS)} the {rng.choice(ADJS)} {rng.choice(NOUNS)} "
            f"near the {rng.choice(PLACES)} on day {i}.")


def make_code(seed: int) -> str:
    rng = random.Random(seed * 7919 + 13)
    word = rng.choice(["MANGO", "KIWI", "LEMON", "PLUM", "OLIVE", "CEDAR", "AMBER", "COBALT", "MAPLE", "ONYX"])
    return f"{word}-{rng.randint(1000, 9999)}"


def haystack(n_tokens: int, depth: float, code: str, seed: int = 0) -> str:
    """Filler of ~n_tokens with 'The secret code is CODE.' at depth (0..1)."""
    rng = random.Random(seed)
    n = max(4, int(n_tokens / TOKENS_PER_SENTENCE))
    at = min(n - 1, max(1, int(n * depth)))
    out = []
    for i in range(n):
        if i == at:
            out.append(f"The secret code is {code}.")
        out.append(sentence(rng, i + 1))
    return " ".join(out)


def messages(n_tokens: int, depth: float, code: str, seed: int = 0) -> list[dict]:
    return [
        {"role": "system", "content": "Answer with the code only."},
        {"role": "user", "content": haystack(n_tokens, depth, code, seed) + "\n\nWhat is the secret code?"},
    ]
