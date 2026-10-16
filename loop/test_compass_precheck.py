"""P2d-26 回归：预检指标（局部性/峰宽/峰数/有效相位/漂移）在合成反例上的行为。无需脑。"""
from __future__ import annotations

import sys

import numpy as np

from loop.compass_precheck import (PEAK_TO_MEAN_MIN, effective_phase, locality, n_peaks,
                                   peak_width, phase_drift)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== compass_precheck metrics ===========")
    n = 16
    single = np.zeros(n)
    single[4] = 1.0
    single[3] = single[5] = 0.6
    check("single bump -> locality high", locality(single) > 2.0)
    check("single bump -> n_peaks == 1", n_peaks(single) == 1)
    ph = effective_phase(single)
    check("single bump -> valid phase near idx4", np.isfinite(ph) and abs(ph - 2 * np.pi * 4 / n) < 0.5)

    uniform = np.ones(n)
    check("uniform -> locality ~1 (<min)", locality(uniform) < PEAK_TO_MEAN_MIN)
    check("uniform -> phase NaN (invalid)", not np.isfinite(effective_phase(uniform)))

    silent = np.zeros(n)
    check("all-silent -> phase NaN", not np.isfinite(effective_phase(silent)))
    check("all-silent -> n_peaks 0", n_peaks(silent) == 0)

    two = np.zeros(n)
    two[2] = two[10] = 1.0
    check("two bumps -> n_peaks == 2", n_peaks(two) == 2)

    # 峰宽：宽度 3 的块
    blk = np.zeros(n)
    blk[3:6] = 1.0
    check("3-wide block -> peak_width == 3", peak_width(blk) == 3.0)

    check("phase_drift wraps correctly",
          abs(phase_drift([0.0, 0.1, 0.2]) - 0.2) < 1e-9
          and abs(phase_drift([-3.0, 0.0, 3.0]) - (6.0 - 2 * np.pi)) < 1e-6)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
