"""P5-1 回归：p5_looming 的纯逻辑（刺激预算匹配 / 条件结构 / 群组拆分 / 冻结常量）。无模拟器。"""
from __future__ import annotations

import sys

import numpy as np

from loop.p5_looming import (C0_L, C0_R, CONDS, K_SIG, SEEDS, _frames, _groups, load_data)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== p5_looming ===========")
    az = np.linspace(-1, 1, 6006)
    F = {c: _frames(az, c) for c in CONDS}
    check("none condition is zero drive", not F["none"].any())
    check("luminance is spatially uniform when active",
          np.allclose(F["luminance"][50, :], F["luminance"][50, 0]) and F["luminance"][50, 0] > 0)
    # 预算匹配：非 none 条件的总线能量相等
    tot = {c: float(F[c].sum()) for c in CONDS if c != "none"}
    check("budget matched across stim conditions",
          max(tot.values()) - min(tot.values()) < 1e-3 * max(tot.values()))
    # 扩张的中心在左/右
    k = 60
    check("left_loom centered left", abs(az[int(np.argmax(F["left_loom"][k]))] - C0_L) < 0.15)
    check("right_loom centered right", abs(az[int(np.argmax(F["right_loom"][k]))] - C0_R) < 0.15)
    # 扩张 vs 收缩：带宽随时间变宽/变窄（用二阶矩）
    def width(cond):
        c0 = C0_L if cond.startswith("left") else C0_R
        v1 = np.average((az - c0) ** 2, weights=F[cond][27])   # STIM 早期
        v2 = np.average((az - c0) ** 2, weights=F[cond][72])   # STIM 晚期
        return v1, v2
    wl = width("left_loom"); wc = width("left_contract")
    check("loom widens over time", wl[1] > wl[0])
    check("contract narrows over time", wc[1] < wc[0])

    # 冻结契约
    check("frozen seeds/conds/K_SIG", SEEDS == [11, 12, 13, 14, 15]
          and CONDS == ("none", "left_loom", "right_loom", "left_contract", "luminance") and K_SIG == 3.0)
    D = load_data(); g = _groups(D)
    check("LC4 L+R == LC4 total", len(g["LC4_L"]) + len(g["LC4_R"]) == int((D["cell_type"] == "LC4").sum()))
    check("LPLC2 L&R non-empty", len(g["LPLC2_L"]) > 0 and len(g["LPLC2_R"]) > 0)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
