"""P4-1 回归：p4_sensory_motor 的纯逻辑（行 L1 归一化 + 归档口径）与冻结常量。"""
from __future__ import annotations

import sys

import numpy as np
from scipy import sparse

from loop.p4_sensory_motor import (GO_EFFECT, GO_STABLE, GO_VPN, SEEDS, SOLO_CHANGE,
                                   archive_p31a, row_l1_normalize)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== p4_sensory_motor ===========")
    # 行 L1 归一化：每行 |w| 之和 = 1；零行不变
    W = sparse.csr_matrix(np.array([[2.0, -2.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, -1.0]]))
    Wn = row_l1_normalize(W).toarray()
    check("row L1 == 1", np.allclose(np.abs(Wn).sum(1), [1.0, 0.0, 1.0]))
    check("zero row preserved", np.allclose(Wn[1], [0.0, 0.0, 0.0]))
    check("signs preserved", Wn[0, 1] < 0 and Wn[2, 0] > 0)

    # 冻结契约
    check("single frozen change named", "normalization" in SOLO_CHANGE.lower())
    check("frozen seeds", SEEDS == [11, 12, 13, 14, 15])
    check("frozen gates", (GO_STABLE, GO_EFFECT, GO_VPN) == (4, 0.5, 0.5))

    # 归档口径：control-subtracted，5 seed
    arc = archive_p31a()
    check("archive has 5 seeds", len(arc["per_seed"]) == SEEDS.__len__())
    check("archive median finite", np.isfinite(arc["median_abs_left_dnL_minus_dnR"]))
    check("archive uses control-subtraction note", "control" in arc["note"])

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
