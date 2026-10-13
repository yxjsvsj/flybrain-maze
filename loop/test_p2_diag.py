"""P2d-17 回归：p2_diag.analyze 的四类原因判定 + 有符号/绝对累计转角（无需脑）。"""
from __future__ import annotations

import sys

import numpy as np

from loop.p2_diag import analyze, spans

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def mk(n=100, om=None):
    t = np.arange(n) * 0.02
    turn = np.zeros(n)
    turn[10:50] = 1.0                      # assertive 步 10..49
    front = np.zeros(n)
    front[60:70] = 1.0
    if om is None:
        om = np.zeros(n)
        om[10:50] = 1.0
        om[60:70] = 1.0
    nearR = np.zeros(n)
    nearR[10:30] = 1.0                     # 相关侧(右,因 turn>0)在 10..29 有目标 -> B
    return {"t": t, "turn": turn, "front": front, "om": om, "nearL": np.zeros(n), "nearR": nearR,
            "Dmb": np.full(n, 0.2), "reach": False, "cev": 0, "bestD": 3}


def main() -> int:
    print("=========== p2_diag.analyze ===========")
    check("spans finds a contiguous run", spans(np.arange(10) * 0.1, np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool))
          == [(0.1, 0.2)])

    a = analyze(mk())
    check("B (assertive & relevant-near) ~ 20%", abs(a["B_frac"] - 0.20) < 0.02)
    check("A (assertive & no target) ~ 20%", abs(a["A_frac"] - 0.20) < 0.02)
    check("C saturation (|turn|>=1) ~ 40%", abs(a["C_frac"] - 0.40) < 0.02)
    check("D front-stop & rotating ~ 10%", abs(a["D_frac"] - 0.10) < 0.02)
    m = mk()
    check("signed cumulative turn matches sum(omega)*dt",
          abs(a["cum_signed_deg"] - np.degrees(np.sum(m["om"]) * 0.02)) < 1e-6)
    check("abs == signed when no sign cancellation", abs(a["cum_abs_deg"] - a["cum_signed_deg"]) < 1e-6)

    # 正负抵消：signed ~ 0 但 abs 大
    om2 = np.zeros(100)
    om2[10:30] = 1.0
    om2[30:50] = -1.0
    a2 = analyze(mk(om=om2))
    check("cancellation: signed ~ 0", abs(a2["cum_signed_deg"]) < 1e-6)
    check("cancellation: abs > 0", a2["cum_abs_deg"] > 20)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
