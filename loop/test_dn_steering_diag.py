"""P2d-13c 回归：失败分类器 classify() 的可审计性（纯 numpy，无需脑）。"""
from __future__ import annotations

import sys

from loop.dn_steering_diag import classify

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def rec(t, dmin_F, turn, need, front=False):
    return {"t": t, "dmin_F": dmin_F, "turn": turn, "need_sign": need, "front": front}


def main() -> int:
    print("=========== dn_steering_diag classify ===========")
    dt = 0.02
    fail = {"reach": False, "bestD": 5, "stop%": 0.0, "stuck": False}

    check("reached -> PASS", classify({"reach": True, "bestD": 0, "stop%": 0, "stuck": False},
                                      [], dt)[0] == "PASS")

    no_dec = [rec(i * dt, 5.0, 0.0, -1) for i in range(100)]
    check("no decision region -> E", classify(fail, no_dec, dt)[0] == "E")

    # A: 需要 -1，实际稳定 +0.5 -> 方向错
    a_recs = [rec(i * dt, 0.8, 0.5, -1) for i in range(200)]
    check("wrong direction -> A", classify(fail, a_recs, dt)[0] == "A")

    # C: 方向对（-1）但幅度小、无 front-stop
    c_recs = [rec(i * dt, 0.8, -0.12, -1) for i in range(200)]
    check("correct sign but weak -> C", classify(fail, c_recs, dt)[0] == "C")

    # D: 饱和 + 反复换向
    d_recs = [rec(i * dt, 0.8, (1.0 if i % 2 else -1.0), -1) for i in range(200)]
    check("saturated oscillation -> D", classify(fail, d_recs, dt)[0] == "D")

    # B: 方向对但弱且先发生 front_stop
    b_recs = [rec(i * dt, 0.8, -0.1, -1, front=(i > 5)) for i in range(200)]
    check("front-stop before strong turn -> B", classify(fail, b_recs, dt)[0] == "B")

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
