"""P2d-13b 回归：DNa02 最小转向读出（基线校正 + 平滑 + 限幅）与切换汇总。纯 numpy，无需脑。"""
from __future__ import annotations

import sys

import numpy as np

from loop.dn_steering import CLAMP, DNReadout, summarize_test

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def settle(rd, raw, n=300):
    t = 0.0
    for _ in range(n):
        t = rd(raw)
    return t


def main() -> int:
    print("=========== DNa02 minimal steering readout ===========")
    dt = 0.02

    # 基线校正：raw==b0 -> turn -> 0
    rd = DNReadout(dt, b0=0.5)
    check("baseline: raw==b0 -> turn->0", abs(settle(rd, 0.5)) < 1e-3)

    # 左右判别：同样相对基线的偏离，一个正一个负
    rd = DNReadout(dt, b0=0.5)
    left = settle(rd, 1.5)          # 相对基线 +1.0
    rd = DNReadout(dt, b0=0.5)
    right = settle(rd, -0.5)        # 相对基线 -1.0
    check("left stimulus -> turn>0", left > 0.9)
    check("right stimulus -> turn<0", right < -0.9)
    check("opposite stimuli -> opposite signs", np.sign(left) == -np.sign(right))

    # 限幅
    rd = DNReadout(dt, b0=0.0)
    check("clamp at +CLAMP", abs(settle(rd, 100.0) - CLAMP) < 1e-6)
    rd = DNReadout(dt, b0=0.0)
    check("clamp at -CLAMP", abs(settle(rd, -100.0) + CLAMP) < 1e-6)

    # 平滑：单步不是立即到目标（抑制 bang-bang）
    rd = DNReadout(dt, b0=0.0, tau_s=0.5)
    t1 = rd(1.0)
    check("smoothing: single step << target (no bang-bang)", 0.0 < t1 < 0.2)

    # use_baseline=False 时不做基线扣除
    rd = DNReadout(dt, b0=0.5, use_baseline=False)
    check("use_baseline=False ignores b0", abs(settle(rd, 0.5) - 0.5) < 1e-3)

    # summarize_test：符号 / 延迟 / 恢复
    n = 100
    segs = [{"seg": "A", "turn": np.zeros(n)},
            {"seg": "B", "turn": np.full(n, 0.8)},
            {"seg": "A", "turn": np.zeros(n)}]
    s = summarize_test(segs, dt)
    check("summarize: sign(baseline->left stim)=+", s["sign"] > 0)
    check("summarize: recovery A1 ~ 0", abs(s["a1"]) < 1e-9)
    check("summarize: positive latency found", s["lat"] >= 0.0)
    check("summarize: no latency when below thr",
          summarize_test([{"seg": "A", "turn": np.zeros(n)},
                          {"seg": "B", "turn": np.full(n, 0.2)},
                          {"seg": "A", "turn": np.zeros(n)}], dt)["lat"] < 0)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
