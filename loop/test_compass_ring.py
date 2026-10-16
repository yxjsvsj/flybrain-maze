"""P2d-26 回归：环坐标重建的纯逻辑（环性判定：环 vs 线 vs 两簇）。无需脑。"""
from __future__ import annotations

import sys

import numpy as np

from loop.compass_ring import ring_metrics

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== compass_ring logic ===========")
    # 真环：点在对角相邻 + 首尾闭环
    n = 12
    S = np.zeros((n, n))
    for i in range(n):
        S[i, (i + 1) % n] = 1.0
        S[(i + 1) % n, i] = 1.0
    m = ring_metrics(S, np.arange(n))
    check("true ring -> is_ring True (wrap high)", m["is_ring"] and m["wrap"] > 0.9)
    check("true ring -> lift high", m["lift"] > 2.0)

    # 线序：相邻高、无首尾闭环
    L = np.zeros((n, n))
    for i in range(n - 1):
        L[i, i + 1] = L[i + 1, i] = 1.0
    ml = ring_metrics(L, np.arange(n))
    check("line -> is_ring False (wrap 0)", (not ml["is_ring"]) and ml["wrap"] == 0.0)

    # 两簇：块内高、块间 0
    B = np.zeros((n, n))
    B[:n // 2, :n // 2] = 1.0
    B[n // 2:, n // 2:] = 1.0
    np.fill_diagonal(B, 0.0)
    mb = ring_metrics(B, np.arange(n))
    check("two-cluster -> is_ring False", not mb["is_ring"])

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
