"""P2d-29 回归：p2d29_candidate 纯逻辑（causal_lp / ns_turn / 冻结门槛裁定）。无模拟器。"""
from __future__ import annotations

import sys

import numpy as np

from loop.p2d29_candidate import (DIR_KEEP_MIN, GAIN, REDUCE_MIN, SIGN, TAU_F, causal_lp, ns_turn)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== p2d29_candidate ===========")
    check("frozen params TAU_F=1.0, gain=8, sign=-1", (TAU_F, GAIN, SIGN) == (1.0, 8.0, -1.0))

    # causal_lp 收敛到常量 -> 等于常量（稳态无偏）
    x = np.ones(2000) * 0.5
    f = causal_lp(x, dt=0.02, tau=TAU_F)
    check("LP converges to constant input", abs(f[-1] - 0.5) < 1e-3)
    f0 = causal_lp(np.array([0.5]), 0.02, TAU_F, x0=0.1)[0]
    check("LP is causal (single-step update)", abs(f0 - (0.1 + 0.02 * (0.5 - 0.1))) < 1e-12)

    # 更小的 tau 收敛更快
    step = np.zeros(300); step[50:] = 1.0
    f_fast = causal_lp(step, 0.02, 0.1); f_slow = causal_lp(step, 0.02, 2.0)
    check("smaller tau tracks faster", f_fast[60] > f_slow[60])

    # ns_turn: 正 Dmb -> 负 turn（sign=-1）；并裁剪到 ±1
    tn = ns_turn(np.ones(500) * 0.5)
    check("positive Dmb -> negative turn", tn[-1] < 0)
    check("output clipped to [-1,1]", tn.max() <= 1.0 and tn.min() >= -1.0)
    check("large input saturates at -1", abs(ns_turn(np.ones(500) * 5.0)[-1] + 1.0) < 1e-9)

    # 关键：常值 DC 不会被去除（说明 NS 不消 DC 偏置）——这是 Part D 失败的机制
    dc = ns_turn(np.ones(500) * 0.2)
    check("NS does NOT remove DC bias (|turn|->gain*dc)", abs(abs(dc[-1]) - min(1.0, GAIN * 0.2)) < 1e-6)

    # 门槛数值
    check("gate constants", REDUCE_MIN == 0.30 and DIR_KEEP_MIN == 0.60)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
