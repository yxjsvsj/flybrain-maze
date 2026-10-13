"""P2d-16 回归：演示脚本的索引审计与渲染（渲染用合成数据，无需脑）。"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

from loop.local_avoidance_bench import SCENARIOS
from loop.p2_demo import render
from loop.population_turn import (_ck, L_BRAIN_IDX, L_CHECKSUM, L_IDX, R_BRAIN_IDX, R_CHECKSUM,
                                 R_IDX, LB_CHECKSUM, RB_CHECKSUM, K_GROUP)
from world.maze import Maze

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def fake_demo(n=60):
    t = np.arange(n) * 0.02
    rec = {"t": t, "x": np.linspace(1.5, 6, n), "y": np.linspace(4, 2, n), "th": np.zeros(n),
           "D": np.sin(t * 3), "turn": 0.5 * np.sin(t * 3), "v": 0.5 * np.ones(n),
           "om": 0.1 * np.cos(t * 3), "front": np.zeros(n), "gd": np.zeros(n)}
    return {"rec": rec, "reach": True, "t_goal": 1.2, "cev": 0, "cevs": 0,
            "maze": Maze(SCENARIOS["must_left"])}


def main() -> int:
    print("=========== p2_demo ===========")
    # 分组索引审计：L/R 位置索引与全脑 ID 各自校验（不合并）
    check("L position checksum matches frozen", _ck(L_IDX) == L_CHECKSUM)
    check("R position checksum matches frozen", _ck(R_IDX) == R_CHECKSUM)
    check("L brain-ID checksum matches frozen", _ck(L_BRAIN_IDX) == LB_CHECKSUM)
    check("R brain-ID checksum matches frozen", _ck(R_BRAIN_IDX) == RB_CHECKSUM)
    check("L/R sizes == K_GROUP", len(L_IDX) == K_GROUP and len(R_IDX) == K_GROUP)
    check("L/R brain IDs disjoint", len(set(L_BRAIN_IDX) & set(R_BRAIN_IDX)) == 0)

    out = os.path.join(tempfile.gettempdir(), "p2demo_test.png")
    render(Maze(SCENARIOS["must_left"]), [("POP", fake_demo(), "tab:blue")],
           "synthetic test", out)
    check("render writes a non-empty PNG", os.path.exists(out) and os.path.getsize(out) > 1000)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
