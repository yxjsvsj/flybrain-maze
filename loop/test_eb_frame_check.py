"""P2d-26e 回归：eb_frame_check 的同框判据（bbox 包含 + 负对照 + 错轴对照；非"重叠即过"）。"""
from __future__ import annotations

import sys

import numpy as np

from loop.eb_frame_check import roi_split, same_frame_report

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== eb_frame_check ===========")
    # 非立方且在 x 上偏移的框：真正落在其内的点群交换 (x,y) 后应落外 → identity 可判别
    mesh_bbox = {"min": [40.0, 10.0, 0.0], "max": [100.0, 30.0, 50.0], "extent": [60.0, 20.0, 50.0]}
    rng = np.random.default_rng(0)
    eb = rng.uniform([45, 12, 5], [55, 20, 45], size=(500, 3))          # inside
    non = rng.uniform([200, 200, 200], [300, 300, 300], size=(500, 3))  # far outside
    rep = same_frame_report(mesh_bbox, eb, non)
    check("EB points inside bbox -> frac 1.0", rep["eb_frac_in_bbox"] == 1.0)
    check("identity verdict True (swap falls out)", rep["same_frame_identity"] is True)
    check("swap actually falls outside", rep["swap_axis_frac_in_bbox"] <= 0.05)

    # 错轴：把点群本身换成 (x,y) 交换后的错误坐标 -> 落入率骤降 -> identity 不成立
    swapped_pts = eb[:, [1, 0, 2]]
    rep2 = same_frame_report(mesh_bbox, swapped_pts, non)
    check("axis-swapped points -> containment drops", rep2["eb_frac_in_bbox"] < 0.99)
    check("axis-swapped points -> not identity-consistent", rep2["same_frame_identity"] is False)

    # 仅重叠、但不包含（点散布在框外一半）-> 不通过
    half = np.vstack([eb, non])
    rep3 = same_frame_report(mesh_bbox, half, non)
    check("partial containment (<0.99) -> fail", rep3["same_frame_identity"] is False)

    # roi_split：严格 token 'EB'（'EBr1' 不算 EB）
    rows = [{"roi": "EB;FB", "x": 1, "y": 1, "z": 1},
            {"roi": "EBr1", "x": 2, "y": 2, "z": 2},
            {"roi": "PB", "x": 3, "y": 3, "z": 3}]
    e, n = roi_split(rows)
    check("roi_split counts strict EB only", len(e) == 1 and len(n) == 2)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
