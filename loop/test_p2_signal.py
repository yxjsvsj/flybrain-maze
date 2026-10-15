"""P2d-21a 回归：检测器评价口径（事件级 FAR、因果方向、分段、miss 不算正确）。无需脑。"""
from __future__ import annotations

import sys

import numpy as np

from loop.p2_signal import (FAR_Q, PLANS, THR_FROZEN, WIN_S, detect_segment, event_far, over_runs,
                            roll_mean, schedule, segments)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== p2_signal (detector eval) ===========")
    dt = 0.02
    check("THR frozen to 0.1246 (not re-optimized)", THR_FROZEN == 0.1246 and WIN_S == 0.5
          and FAR_Q == 99.0)

    # 连续超阈算一次事件
    mask = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1], bool)
    check("over_runs: consecutive True = ONE run", over_runs(mask) == [(1, 4), (5, 6), (7, 9)])

    ev = event_far(np.array([0.0, 1, 1, 0, 1]), 0.5, dt, 40.0)
    check("event_far counts 2 independent events", ev["n_events"] == 2)
    check("event_far step fraction 3/5", abs(ev["step_frac"] - 3 / 5) < 1e-9)
    check("event_far durations", ev["durs"] == [0.04, 0.02])

    # 检出时方向错，但整段均值方向对 —— 方向必须用**检测时刻**判
    wabs = np.array([0.0, 0, 2.0, 2.0, 0, 0])         # 在 2,3 超阈
    wsq = np.array([0.0, 0, -1.0, -1.0, 1.0, 2.0])    # 检测时刻 signed=-1，但段均值>0
    oc, lat, d = detect_segment(wabs, wsq, 1.0, 2, 6, dt, want=+1)
    check("detection-time direction used (not segment mean): outcome<=wrong, dir=-1",
          oc == "wrong" and d == -1)
    check("segment-mean direction would have been + (mean wsq>0)",
          float(np.mean(wsq[2:6])) > 0)

    # 漏检必须记为 miss，不能算正确方向
    oc2, lat2, d2 = detect_segment(np.zeros(6), np.zeros(6), 1.0, 2, 6, dt, want=+1)
    check("no over-threshold -> miss (never auto-correct)", oc2 == "miss" and lat2 == -1.0 and d2 == 0)

    # 方向不确定
    oc3, _, _ = detect_segment(wabs, np.array([0.0, 0, 0.0, 0.0, 0, 0]), 1.0, 2, 6, dt, want=+1)
    check("zero signed window -> uncertain", oc3 == "uncertain")

    # 分段：交替 D 计划必须分成 FR 与 FL 两段（不混为一窗）
    names = [s[0] for s in segments(PLANS["D"], dt)]
    check("D plan segmented A/FR/A/FL/A", names == ["A", "FR", "A", "FL", "A"])
    segsB = segments(PLANS["B"], dt)
    check("B plan has one FL stimulus segment",
          [n for n, _, _ in segsB if n != "A"] == ["FL"])

    check("schedule length from segments", len(schedule(PLANS["B"], dt)) == segments(PLANS["B"], dt)[-1][2])
    check("roll_mean steady ~1", abs(roll_mean(np.ones(100), 10)[-1] - 1.0) < 1e-9)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
