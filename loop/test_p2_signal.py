"""P2d-21 回归：计划调度 / 窗口统计 / 背景统计（无需脑）。"""
from __future__ import annotations

import sys

import numpy as np

from loop.p2_signal import FAR_Q, PLANS, WIN_S, roll_mean, schedule, stats

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== p2_signal ===========")
    dt = 0.02
    sB = schedule(PLANS["B"], dt)
    check("schedule B: 60 A + 100 FL + 240 A = 400 steps",
          len(sB) == 400 and sB[:60] == ["A"] * 60 and sB[60:160] == ["FL"] * 100
          and sB[160:] == ["A"] * 240)
    sD = schedule(PLANS["D"], dt)
    check("schedule D contains both FR and FL", "FR" in sD and "FL" in sD)

    rm = roll_mean(np.ones(200), 10)
    check("roll_mean keeps length", len(rm) == 200)
    check("roll_mean steady value ~1", abs(rm[-1] - 1.0) < 1e-9)

    st = stats(np.zeros(100), dt)
    check("stats of zeros: mean/std 0", st["mean"] == 0.0 and st["std"] == 0.0)
    st2 = stats(np.random.default_rng(0).normal(0.1, 0.05, 2000), dt)
    check("stats mean ~0.1", abs(st2["mean"] - 0.1) < 0.02)
    check("stats q has 5 entries", len(st2["q"]) == 5)
    check("stats block_means has 1/5/10/20/40s",
          set(st2["block_means"]) == {"1s", "5s", "10s", "20s", "40s"})
    check("WIN_S/FAR_Q declared", WIN_S == 0.5 and FAR_Q == 99.0)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
