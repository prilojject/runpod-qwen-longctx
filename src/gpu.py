"""GPU info via nvidia-smi (no torch in this image) with a CUDA-driver fallback,
plus a VRAM peak sampler.

Some Runpod hosts answer '[Insufficient Permissions]' / '[N/A]' for nvidia-smi
fields, so every field is parsed defensively and name / total memory fall back
to the CUDA driver API (cuDeviceGetName / cuDeviceTotalMem, no context created).
"""

from __future__ import annotations

import ctypes
import subprocess
import threading
from typing import Any


def _smi(query: str) -> list[str]:
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True, timeout=10)
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


def _num(text: str | None) -> float | None:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _clean(text: str | None) -> str | None:
    t = (text or "").strip()
    return None if not t or t.startswith("[") else t


def _cuda_driver_info() -> dict[str, Any]:
    """Name / total memory / compute capability of device 0 via libcuda (no context)."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        if cuda.cuInit(0) != 0:
            return {}
        dev = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            return {}
        name = ctypes.create_string_buffer(256)
        cuda.cuDeviceGetName(name, 256, dev)
        total = ctypes.c_size_t()
        cuda.cuDeviceTotalMem_v2(ctypes.byref(total), dev)
        major, minor = ctypes.c_int(), ctypes.c_int()
        cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)  # COMPUTE_CAPABILITY_MAJOR
        cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)  # COMPUTE_CAPABILITY_MINOR
        count = ctypes.c_int()
        cuda.cuDeviceGetCount(ctypes.byref(count))
        return {
            "gpu": name.value.decode(errors="replace") or None,
            "vram_total_gib": round(total.value / 2**30, 2) if total.value else None,
            "compute_capability": f"{major.value}.{minor.value}" if major.value else None,
            "gpu_count": count.value or None,
        }
    except Exception:
        return {}


def parse_smi_row(row: str) -> dict[str, Any]:
    parts = [p.strip() for p in row.split(",")] + [""] * 4
    name, mem, driver, cap = parts[:4]
    mib = _num(mem)
    return {
        "gpu": _clean(name),
        "vram_total_gib": round(mib / 1024, 2) if mib else None,
        "driver": _clean(driver),
        "compute_capability": _clean(cap),
    }


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"gpu": None, "vram_total_gib": None}
    try:
        rows = _smi("name,memory.total,driver_version,compute_cap")
        if rows:
            info.update(parse_smi_row(rows[0]))
            info["gpu_count"] = len(rows)
    except Exception:
        pass
    if not info.get("gpu") or not info.get("vram_total_gib"):
        drv = _cuda_driver_info()
        for k, v in drv.items():
            if v and not info.get(k):
                info[k] = v
    return info


def vram_used_mib() -> float | None:
    try:
        return _num(_smi("memory.used")[0])
    except Exception:
        return None


class VramPeak:
    """Poll nvidia-smi in a daemon thread; report peak memory.used in GiB (None if unreadable)."""

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
            self._sample()
            self._stop.wait(self.interval)

    def _sample(self) -> None:
        v = vram_used_mib()
        if v is not None and (self.peak_mib is None or v > self.peak_mib):
            self.peak_mib = v

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=15)
        self._sample()

    @property
    def peak_gib(self) -> float | None:
        return None if self.peak_mib is None else round(self.peak_mib / 1024, 2)
