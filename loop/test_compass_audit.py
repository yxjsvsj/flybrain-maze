"""P2d-25 回归：compass 审计的纯逻辑（1/2 跳可达；合成图，无需脑）。"""
from __future__ import annotations

import sys

from loop.compass_audit import PATTERNS, one_hop, two_hop

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== compass_audit logic ===========")
    csc = ([0, 1, 2, 3, 3], [1, 2, 3])          # 0->1, 1->2, 2->3
    check("one_hop from 0 == {1}", one_hop(csc, [0]) == {1})
    check("two_hop from 0 == {1,2}", two_hop(csc, [0]) == {1, 2})
    check("one_hop isolated node == {}", one_hop(csc, [3]) == set())
    # 双向环：0<->1
    csc2 = ([0, 1, 2], [1, 0])                   # 0->1, 1->0
    check("bidirectional loop detected (0<->1)",
          1 in one_hop(csc2, [0]) and 0 in one_hop(csc2, [1]))
    check("PATTERNS include EPG/PEN/ER/ExR", all(p in PATTERNS for p in ("EPG", "PEN", "ER", "ExR")))

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
