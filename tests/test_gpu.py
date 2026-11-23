from src import gpu


def test_insufficient_permissions_row_does_not_crash():
    d = gpu.parse_smi_row("NVIDIA L40S, [Insufficient Permissions], 570.86, 8.9")
    assert d["gpu"] == "NVIDIA L40S" and d["vram_total_gib"] is None and d["compute_capability"] == "8.9"
    d = gpu.parse_smi_row("[Insufficient Permissions], [N/A], [N/A], [N/A]")
    assert d["gpu"] is None and d["vram_total_gib"] is None


def test_gpu_info_falls_back_to_driver(monkeypatch):
    monkeypatch.setattr(gpu, "_smi", lambda q: ["NVIDIA RTX A6000, [Insufficient Permissions], 550.1, 8.6"])
    monkeypatch.setattr(gpu, "_cuda_driver_info", lambda: {"gpu": "NVIDIA RTX A6000", "vram_total_gib": 47.53})
    info = gpu.gpu_info()
    assert info["gpu"] == "NVIDIA RTX A6000" and info["vram_total_gib"] == 47.53


def test_vram_used_unreadable(monkeypatch):
    monkeypatch.setattr(gpu, "_smi", lambda q: ["[Insufficient Permissions]"])
    assert gpu.vram_used_mib() is None
