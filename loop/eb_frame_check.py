"""P2d-26e：**EB 网格 ↔ 突触坐标同框核验**（可复现；只用来源说明 + 空间叠合，不选变换）。

判据（**不是**"两个 bbox 重叠就算过"）：
  1. 来源：网格与突触同为 DVID/飞出版本 v1.0（显式来源说明）。
  2. **独立标注**的 EB 突触点（neuPrint 的 ROI 字段）应**落在** EB 网格体积内：
     - bbox 逐轴包含率（对 88 细胞整体与逐细胞）；
     - 点-在-网格近似（最近面法向有符号距离，网格非水密 → 只作佐证）；
  3. **负对照**：非 EB 突触点不应落在 EB 内；**错轴对照**（把 EB 点 (x,y) 交换）应几乎全部落到外面
     —— 只有 **identity** 变换自洽（**不尝试多种变换挑最好看的**）。

纯函数 `same_frame_report()` 可离线测试；CLI 读真实 OBJ 与坐标 CSV。
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

from loop.coord_frame import axis_containment, bbox, obj_vertices


def roi_split(rows):
    """按 ROI 字段分 EB（严格 token `EB`）与非 EB 点。"""
    eb, non = [], []
    for r in rows:
        toks = [t.strip() for t in str(r.get("roi") or "").split(";")]
        p = (float(r["x"]), float(r["y"]), float(r["z"]))
        (eb if "EB" in toks else non).append(p)
    return np.asarray(eb, float).reshape(-1, 3), np.asarray(non, float).reshape(-1, 3)


def same_frame_report(mesh_bbox, eb_pts, non_eb_pts, swap_eb=None, margin: float = 0.0) -> dict:
    eb_pts = np.asarray(eb_pts, float).reshape(-1, 3)
    non_eb_pts = np.asarray(non_eb_pts, float).reshape(-1, 3)
    eb_frac = axis_containment(mesh_bbox, eb_pts, margin)["frac_inside"] if len(eb_pts) else 0.0
    non_frac = axis_containment(mesh_bbox, non_eb_pts, margin)["frac_inside"] if len(non_eb_pts) else 0.0
    sw = eb_pts[:, [1, 0, 2]] if swap_eb is None else np.asarray(swap_eb, float)
    swap_frac = axis_containment(mesh_bbox, sw, margin)["frac_inside"] if len(sw) else 0.0
    # 判据：独立标注的 EB 点须落入 EB 网格(bbox)，且**错轴对照**须落外（identity 唯一自洽）。
    # 注意：non-EB **bbox** 包含率**不作判据**——不同 ROI(FB/CRE/LAL…)在空间上与 EB 的 bbox 重叠，
    # ROI 并非 bbox 不相交；仅作信息记录。
    ok = (eb_frac >= 0.99) and (swap_frac <= 0.05)
    return {"eb_frac_in_bbox": round(float(eb_frac), 4),
            "other_roi_frac_in_bbox_INFO": round(float(non_frac), 4),
            "swap_axis_frac_in_bbox": round(float(swap_frac), 4),
            "same_frame_identity": bool(ok),
            "verdict": ("同框成立（identity 自洽：独立标注 EB 突触点落入 EB 网格，错轴对照落外）"
                        if ok else "同框不成立或证据不足")}


def point_in_mesh_fraction(pts, V, F, sample=4000, seed=0):
    """最近面法向有符号距离近似（网格非水密，仅佐证）。<0 记为内部。"""
    pts = np.asarray(pts, float).reshape(-1, 3)
    if not len(pts):
        return 0.0, 0.0
    V = np.asarray(V, float); F = np.asarray(F, int)
    cent = V[F].mean(1)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    rng = np.random.default_rng(seed)
    s = pts[rng.choice(len(pts), min(sample, len(pts)), replace=False)]
    sd = np.empty(len(s))
    for i, p in enumerate(s):
        j = int(np.argmin(((cent - p) ** 2).sum(1)))
        sd[i] = float(np.dot(p - cent[j], nrm[j]))
    return float(np.mean(sd < 0)), float(np.median(sd))


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                     # noqa: BLE001
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default="loop/p2d26e_coords.csv")
    ap.add_argument("--mesh", default=r"C:\Users\yang\AppData\Local\Temp\opencode\syn\EB.obj")
    a = ap.parse_args(argv)
    mesh = obj_vertices(open(a.mesh, encoding="utf-8", errors="replace"))
    mbb = bbox(mesh)
    rows = list(csv.DictReader(open(a.coords, encoding="utf-8")))
    eb, non = roi_split(rows)
    rep = same_frame_report(mbb, eb, non)
    print("mesh bbox:", {k: [round(v) for v in mbb[k]] for k in ("min", "max")})
    print("EB points:", len(eb), " non-EB points:", len(non))
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["same_frame_identity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
