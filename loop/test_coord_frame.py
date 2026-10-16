"""P2d-26d 回归：coord_frame 的 OBJ 解析 / bbox / 逐轴包含 / 结论不只看重叠。"""
from __future__ import annotations

import sys

import numpy as np

from loop.coord_frame import axis_containment, bbox, frame_verdict, obj_vertices

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== coord_frame ===========")
    lines = ["# OBJ file", "v 0 0 0", "vn 1 0 0", "v 2 4 6", "f 1 2 3", "v 1 1 1"]
    v = obj_vertices(lines)
    check("obj_vertices parses v only", v.shape == (3, 3))
    check("obj_vertices values", np.allclose(v[1], [2, 4, 6]))
    bb = bbox(v)
    check("bbox min/max", bb["min"] == [0.0, 0.0, 0.0] and bb["max"] == [2.0, 4.0, 6.0])
    check("bbox extent", bb["extent"] == [2.0, 4.0, 6.0])

    inside = np.array([[1, 1, 1], [2, 4, 6]], float)
    outside = np.array([[1, 1, 7]], float)
    a = axis_containment(bb, inside)
    check("all_inside True", a["all_inside"] and abs(a["frac_inside"] - 1.0) < 1e-9)
    b = axis_containment(bb, outside)
    check("outside flagged on z axis", b["per_axis"]["z"] is False and b["per_axis"]["x"] is True)

    # 结论：仅"落在框内"不足以宣布同框
    v1 = frame_verdict(False, False, a)
    check("containment alone NOT conclusive", v1["conclusive"] is False)
    v2 = frame_verdict(True, False, a)
    check("missing transform evidence -> not conclusive", v2["conclusive"] is False)
    v3 = frame_verdict(True, True, a)
    check("origin+transform evidence -> conclusive", v3["conclusive"] is True)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
