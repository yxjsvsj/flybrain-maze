"""P2d-18 回归：反事实的分叉规则 / 帧重建与撤除 / 指标（无需脑）。"""
from __future__ import annotations

import sys

import numpy as np

from loop.p2_counterfactual import frame_at, metrics, pick_fork

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def fake_rec(n=60):
    t = np.arange(n) * 0.02
    r = {"t": t, "seq": np.repeat(np.arange(n // 2 + 1), 2)[:n].astype(float)}
    for s in ("L", "FL", "F", "FR", "R"):
        r["tof" + s] = np.full(n, 500.0)
        r["st" + s] = np.full(n, 2.0)          # VALID
    return r


def main() -> int:
    print("=========== p2_counterfactual ===========")
    # 分叉规则：|turn|>=0.5 的最长段起点
    turn = np.zeros(100)
    turn[20:60] = 0.95                          # 最长段 20..59
    turn[70:80] = 1.0                           # 较短
    check("pick_fork = start of longest strong-turn run", pick_fork(turn, np.arange(100) * 0.02) == 20)

    rec = fake_rec()
    f_orig = frame_at(rec, 5, "orig")
    check("frame_at orig keeps VALID ranges", f_orig.status["F"] == "VALID" and f_orig.ranges["F"] == 500.0)
    f_none = frame_at(rec, 5, "none")
    check("mode none removes all", all(f_none.status[n] == "NO_TARGET" for n in ("L", "FL", "F", "FR", "R")))
    f_noF = frame_at(rec, 5, "noF")
    check("mode noF removes only F", f_noF.status["F"] == "NO_TARGET" and f_noF.status["FL"] == "VALID")
    f_noLR = frame_at(rec, 5, "noLR")
    check("mode noLR removes FL/FR but keeps F",
          f_noLR.status["FL"] == "NO_TARGET" and f_noLR.status["FR"] == "NO_TARGET"
          and f_noLR.status["F"] == "VALID")

    # 指标：分叉后 turn 落到半值
    Ds = np.zeros(200)
    turns = np.zeros(200)
    turns[:100] = 0.9                            # pre 段
    turns[100:120] = 0.9
    turns[120:] = 0.1                            # 分叉后 0.4s 降下来
    m = metrics(Ds, turns, 100, 0.0, 0.02)
    check("metrics pre_turn_abs ~ 0.9", abs(m["pre_turn_abs"] - 0.9) < 1e-9)
    check("metrics t_fall_half detected", abs(m["t_fall_half"] - 0.4) < 0.05)
    m2 = metrics(Ds, np.full(200, 0.9), 100, 0.0, 0.02)
    check("metrics no fall -> t_fall_half == -1", m2["t_fall_half"] == -1.0)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
