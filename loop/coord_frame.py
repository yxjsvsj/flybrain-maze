"""P2d-26d：**坐标框架核验**工具（纯逻辑，可离线测试）。

- 解析 OBJ 网格顶点（`v x y z`，忽略 vn/f/注释）；
- 给出 bbox 与"突缺点是否落在网格框内"的**逐轴**判定；
- **明确：bbox 包含只是检查之一，范围重叠 ≠ 同框**；函数返回 `conclusive=False`
  除非调用方另附来源/变换证据。
"""
from __future__ import annotations

import numpy as np


def obj_vertices(lines) -> np.ndarray:
    """从 OBJ 文本行里取顶点，返回 (N,3) float64。"""
    vs = []
    for ln in lines:
        if isinstance(ln, bytes):
            ln = ln.decode("utf-8", "replace")
        if ln.startswith("v "):
            p = ln.split()
            vs.append((float(p[1]), float(p[2]), float(p[3])))
    return np.asarray(vs, np.float64).reshape(-1, 3)


def bbox(pts: np.ndarray):
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    return {"min": pts.min(0).tolist(), "max": pts.max(0).tolist(),
            "extent": (pts.max(0) - pts.min(0)).tolist()}


def axis_containment(mesh_bbox, pts, margin: float = 0.0) -> dict:
    """逐轴检查 pts 是否落在 mesh bbox 内（可给 margin）。"""
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    lo = np.asarray(mesh_bbox["min"]) - margin
    hi = np.asarray(mesh_bbox["max"]) + margin
    inside = ((pts >= lo) & (pts <= hi)).all(1)
    per_axis = {}
    for i, ax in enumerate("xyz"):
        per_axis[ax] = bool(((pts[:, i] >= lo[i]) & (pts[:, i] <= hi[i])).all())
    return {"per_axis": per_axis, "frac_inside": float(inside.mean()) if len(pts) else 0.0,
            "all_inside": bool(inside.all()) if len(pts) else False}


def frame_verdict(origin_evidence: bool, transform_evidence: bool, containment: dict) -> dict:
    """结论**不只看是否落在框内**。只有同时具备 (a) 显式来源/坐标系说明 与
    (b) 无未知变换（单位/轴序/缩放/原点一致）时，才 `conclusive=True`；
    bbox 包含仅作佐证。"""
    conclusive = bool(origin_evidence and transform_evidence)
    return {"conclusive": conclusive,
            "note": ("同框已证实" if conclusive else
                     "暂判：坐标量级/包含性一致，但单位/轴序/缩放/原点未全部证实 → 不得据此宣布同框"),
            "containment": containment}
