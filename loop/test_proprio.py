"""P2d-24 回归：通路审计的下游可达计算（合成图，无需脑）。"""
from __future__ import annotations

import sys

from loop.proprio_audit import downstream

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== proprio_audit.downstream ===========")
    # 0->1, 1->2, 2->3, 3->(none)
    csc = ([0, 1, 2, 3, 3], [1, 2, 3])
    ds, trunc = downstream(csc, [0], depth=1)
    check("depth1 from 0 reaches {0,1}", ds == {0, 1} and trunc is False)
    ds2, _ = downstream(csc, [0], depth=2)
    check("depth2 from 0 reaches {0,1,2}", ds2 == {0, 1, 2})
    ds0, _ = downstream(csc, [3], depth=2)
    check("isolated node reaches only itself", ds0 == {3})
    # 分支：0->1,0->2 ; 1->3  (CSC: col0->[1,2], col1->[3])
    csc2 = ([0, 2, 3, 3, 3], [1, 2, 3])
    dsb, _ = downstream(csc2, [0], depth=2)
    check("branching reaches all descendants", dsb == {0, 1, 2, 3})

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
