"""P2d-22 回归：归一化读出的检测边界（pre 用自身历史、carryover 单列、miss 不算正确）。无需脑。"""
from __future__ import annotations

import sys

import numpy as np

from loop.p2_normalize import detect_pre

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== p2_normalize ===========")
    dt = 0.02
    # 段内新起事件 -> 新检出
    wabs = np.array([0., 0, 0, 0, 2, 2, 0, 0])
    wsq = np.array([0., 0, 0, 0, 1, 1, 0, 0])
    oc, lat, d, pf, cy = detect_pre(wabs, wsq, 1.0, 4, 8, dt, +1, pre=[0, 1, 2])
    check("new in-segment event -> correct, dir +1", oc == "correct" and d == +1 and cy is False)
    check("latency measured from segment start", abs(lat - 0.0) < 1e-9)

    # 建段前已持续超阈 -> carryover（不算新检出）
    wabs2 = np.array([0., 0, 2, 2, 2, 2, 2, 2])
    oc2, lat2, d2, pf2, cy2 = detect_pre(wabs2, wsq, 1.0, 4, 8, dt, +1, pre=[0, 1, 2])
    check("pre-existing sustained run -> carryover (not a new detection)", oc2 == "carryover" and cy2 is True)

    # 漏检
    oc3, _, _, pf3, cy3 = detect_pre(np.zeros(8), np.zeros(8), 1.0, 4, 8, dt, +1, pre=[0, 1, 2])
    check("no event -> miss", oc3 == "miss" and cy3 is False)

    # 刺激前误报必须用**该试次自身**的历史
    wabs4 = np.array([0., 3, 0, 0, 0, 0, 0, 0])       # 超阈只在刺激前(索引1)
    _, _, _, pf4, _ = detect_pre(wabs4, np.zeros(8), 1.0, 4, 8, dt, +1, pre=[0, 1, 2, 3])
    check("pre-stim false alarm detected from the trial's OWN history", pf4 is True)
    _, _, _, pf5, _ = detect_pre(wabs4, np.zeros(8), 1.0, 4, 8, dt, +1, pre=[4, 5])
    check("pre window without the event -> no pre-FA", pf5 is False)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
