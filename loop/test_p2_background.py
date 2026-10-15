"""P2d-20 回归：flybrain 动力学审计 + 窗口统计（无需脑）。"""
from __future__ import annotations

import sys

import numpy as np

from loop.p2_background import audit_brain, win_stats

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


class _B:
    dt, tau, gain, tonic = 0.02, 0.100, 3.0, 0.14
    noise_hz, noise_amp, decay = 1.2, 0.22, 0.8
    refractory_steps, sensory_input = 0, False


def main() -> int:
    print("=========== p2_background (audit) ===========")
    a = audit_brain(_B())
    check("audit exposes gain (3.0)", a["gain"] == 3.0)
    check("audit exposes tonic (0.14)", abs(a["tonic"] - 0.14) < 1e-9)
    check("audit exposes noise (hz/amp)", a["noise_hz"] == 1.2 and a["noise_amp"] == 0.22)
    check("audit states inject does not remove tonic/noise/recurrent",
          "tonic" in a["order"] and "noise" in a["order"] and "synaptic" in a["order"])

    dt = 0.02
    x = np.arange(1000, dtype=float)
    # fork=100 (=2.0s)；窗口 0-2s -> 步 100..200 的均值
    check("win_stats window 0-2s mean", abs(win_stats(x, 100, dt, 0.0, 2.0) - np.mean(x[100:200])) < 1e-9)
    check("win_stats window 2-5s mean", abs(win_stats(x, 100, dt, 2.0, 5.0) - np.mean(x[200:350])) < 1e-9)
    check("win_stats window 5-10s mean", abs(win_stats(x, 100, dt, 5.0, 10.0) - np.mean(x[350:600])) < 1e-9)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
