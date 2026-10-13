"""P2d-15c 回归：固定 L50/R50 群体索引 + 人工映射符号（无需脑）。"""
from __future__ import annotations

import sys

import numpy as np

from loop.population_turn import (GAIN, K_GROUP, L_IDX, R_IDX, SIGN, PopulationTurn, INDEX_CHECKSUM,
                                 checksum)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def make_stub(Dval, *, b0=0.2, gain=GAIN, sign=SIGN):
    p = PopulationTurn.__new__(PopulationTurn)      # 跳过 Trace（无需脑）
    p.nL, p.b0, p.gain, p.sign, p.dt = K_GROUP, b0, gain, sign, 0.02
    p.D = lambda fired: Dval
    return p


def main() -> int:
    print("=========== population_turn (frozen) ===========")
    check("L has K_GROUP indices", len(L_IDX) == K_GROUP)
    check("R has K_GROUP indices", len(R_IDX) == K_GROUP)
    check("L/R disjoint", len(set(L_IDX) & set(R_IDX)) == 0)
    check("all indices < 1314", max(L_IDX + R_IDX) < 1314 and min(L_IDX + R_IDX) >= 0)
    check("checksum matches frozen constant", checksum(L_IDX, R_IDX) == INDEX_CHECKSUM)

    # 人工映射：D>0(左障碍)->负；D<0(右障碍)->正；D==b0->0；限幅 ±1
    check("D==b0 -> turn 0", abs(make_stub(0.2)(None)) < 1e-9)
    check("D>b0 (left obstacle) -> negative turn", make_stub(0.3)(None) < 0)
    check("D<b0 (right obstacle) -> positive turn", make_stub(0.1)(None) > 0)
    check("saturates at -1", abs(make_stub(10.0)(None) + 1.0) < 1e-9)
    check("saturates at +1", abs(make_stub(-10.0)(None) - 1.0) < 1e-9)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
