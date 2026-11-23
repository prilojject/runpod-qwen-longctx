"""GPU info via nvidia-smi (no torch in this image) and a VRAM peak sampler."""

from __future__ import annotations

import subprocess
import threading
from typing import Any


def _smi(query: str) -> list[str]:
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True, timeout=10)
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


def gpu_info() -> dict[str, Any]:
    try:
        rows = _smi("name,memory.total,driver_version,compute_cap")
    except Exception:
        return {"gpu": None, "vram_total_gib": None}
    name, mem, driver, cap = [p.strip() for p in rows[0].split(",")][:4]
    return {
        "gpu": name,
        "vram_total_gib": round(float(mem) / 1024, 2),
        "driver": driver,
        "compute_capability": cap,
        "gpu_count": len(rows),
    }


def vram_used_mib() -> float | None:
    try:
        return float(_smi("memory.used")[0])
    except Exception:
        return None


class VramPeak:
    """Poll nvidia-smi in a daemon thread; report peak memory.used in GiB."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_mib: float | None = None
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self) -> "VramPeak":
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            v = vram_used_mib()
            if v is not None and (self.peak_mib is None or v > self.peak_mib):
                self.peak_mib = v
            self._stop.wait(self.interval)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=15)
        v = vram_used_mib()
        if v is not None and (self.peak_mib is None or v > self.peak_mib):
            self.peak_mib = v

    @property
    def peak_gib(self) -> float | None:
        return None if self.peak_mib is None else round(self.peak_mib / 1024, 2)
