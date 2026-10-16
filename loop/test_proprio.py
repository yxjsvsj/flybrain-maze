"""P2d-24a 回归：注入对象/零反馈/时段窗口/种子记录（无需脑）+ 下游可达（合成图）。"""
from __future__ import annotations

import sys

import numpy as np

from loop.proprio_audit import downstream
from loop.proprio_gate2 import (HISTORICAL, NEW_SEEDS, PLANS, YSIGN, prop_inj, seg_mean, segments)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== proprio (P2d-24a) ===========")
    # 下游可达（合成图）
    csc = ([0, 1, 2, 3, 3], [1, 2, 3])
    check("downstream depth2", downstream(csc, [0], depth=2)[0] == {0, 1, 2})

    vL, vR = np.array([10, 11]), np.array([20, 21])
    # 零反馈 / z 段：无额外注入
    inj, iv = prop_inj("z", True, vL, vR)
    check("z state -> no injection", inj == [] and iv == 0.0)
    inj, iv = prop_inj("+", False, vL, vR)
    check("feedback=False -> no injection", inj == [] and iv == 0.0)
    # + 段：双侧正/负偏置，注入对象为 vL/vR
    inj, iv = prop_inj("+", True, vL, vR)
    check("+ injects BOTH pools with +K on L and -K on R",
          len(inj) == 2 and np.array_equal(inj[0][0], vL) and inj[0][1] > 0
          and np.array_equal(inj[1][0], vR) and inj[1][1] < 0 and iv > 0)
    # - 段：符号相反
    inj2, _ = prop_inj("-", True, vL, vR)
    check("- flips the sign vs +", inj2[0][1] < 0 and inj2[1][1] > 0)

    # 分段连续覆盖；末段是 z（撤除后窗口）
    for k, plan in PLANS.items():
        segs = segments(plan, 0.02)
        cont = segs[0][1] == 0 and all(segs[i][2] == segs[i + 1][1] for i in range(len(segs) - 1))
        check(f"plan {k}: segments contiguous", cont)
        check(f"plan {k}: last segment is z (post-removal)", segs[-1][0] == "z")

    # 统计窗口确实落在段末（撤除后）
    x = np.concatenate([np.zeros(100), np.ones(100)])
    check("seg_mean uses the LAST 0.5s of the segment", seg_mean(x, 0, 200, 0.02) == 1.0)

    # 种子记录
    check("declared NEW_SEEDS are not in HISTORICAL", all(s not in HISTORICAL for s in NEW_SEEDS))
    check("1601/2309 recorded as historical", 1601 in HISTORICAL and 2309 in HISTORICAL)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
