"""ToF frame-pack validation and replay objects."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

ORDER = ("L", "FL", "F", "FR", "R")


def validate_pack(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    required = {"ranges_mm", "status", "t", "seq", "scene", "distance_mm"}
    missing = required - set(z.files)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    ranges = np.asarray(z["ranges_mm"], dtype=np.float64)
    status = np.asarray(z["status"])
    t = np.asarray(z["t"], dtype=np.float64).ravel()
    seq = np.asarray(z["seq"], dtype=np.int64).ravel()
    if ranges.ndim != 2 or ranges.shape[1] != len(ORDER) or len(t) != len(ranges):
        raise ValueError(f"{path}: ranges shape {ranges.shape}, t {t.shape}")
    if status.shape != ranges.shape or len(seq) != len(t):
        raise ValueError(f"{path}: status/seq shape mismatch")
    if len(t) < 2 or np.any(np.diff(t) <= 0) or np.any(np.diff(seq) <= 0):
        raise ValueError(f"{path}: t/seq must increase")
    return {
        "path": str(path), "ranges_mm": ranges, "status": status.astype(str),
        "t": t, "seq": seq, "scene": str(np.asarray(z["scene"]).ravel()[0]),
        "distance_mm": int(np.asarray(z["distance_mm"]).ravel()[0]),
        "frames": len(t),
    }


def make_frame(pack: dict, i: int) -> SimpleNamespace:
    return SimpleNamespace(
        ranges={n: (None if not np.isfinite(pack["ranges_mm"][i, j]) else float(pack["ranges_mm"][i, j]))
                for j, n in enumerate(ORDER)},
        status={n: str(pack["status"][i, j]) for j, n in enumerate(ORDER)},
        seq=int(pack["seq"][i]),
        t=float(pack["t"][i]),
        healthy=True,
    )


def load_pack(root: str | Path, scene: str, distance_mm: int) -> dict:
    p = Path(root) / f"{scene}_{int(distance_mm)}mm.npz"
    pack = validate_pack(p)
    if pack["scene"] != scene or int(pack["distance_mm"]) != int(distance_mm):
        raise ValueError(f"{p}: metadata mismatch")
    return pack
