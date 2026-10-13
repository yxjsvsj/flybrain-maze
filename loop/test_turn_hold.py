"""P2d-14 回归：方向坐标审计 + 短时转向保持（纯 numpy，无需脑）。"""
from __future__ import annotations

import sys

import numpy as np

from config import Config
from loop.local_avoidance_bench import SCENARIOS
from loop.turn_hold import HOLD_THRESH, HoldReadout, required_omega_sign
from world.car import DiffDriveCar
from world.maze import Maze

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== turn_hold + direction audit ===========")
    dt = 0.02

    # 1) 方向坐标审计
    check("must_left requires omega<0 (opening at -y)", required_omega_sign(SCENARIOS["must_left"]) < 0)
    check("must_right requires omega>0 (opening at +y)", required_omega_sign(SCENARIOS["must_right"]) > 0)

    # 2) 动力学：正 omega -> theta 增大 -> y 增大
    maze = Maze(["#########", "#.......#", "#.......#", "#.......#",
                 "#.......#", "#.......#", "#.......#", "#.......#", "#########"])
    car = DiffDriveCar(3.5, 5.5, 0.0, Config().car)
    for _ in range(50):
        car.set_command(0.5, 0.5)
        car.step(maze, dt)
    check("+omega increases theta", car.theta > 0)
    check("+omega (from theta~0) increases y", car.y > 5.5)

    # 3) 保持：无证据不发明方向
    hr = HoldReadout(dt, b0=0.0)
    check("weak input -> no invention (returns ~x, not +/-1)", abs(hr(0.0)) < 1e-6)

    # 4) 保持：神经给出强证据 -> 锁定该方向
    hr = HoldReadout(dt, b0=0.0)
    t = 0.0
    for _ in range(50):
        t = hr(1.0)
    check("strong +neural -> latched + and returns +level", t > 0.9 and hr.hold_dir > 0)
    check("activation counted once", hr.activations == 1)

    # 5) 保持：证据消失后维持一段时间再释放
    held = [hr(0.0) for _ in range(3)]
    check("hold persists right after evidence drops", all(v > 0.9 for v in held))
    released = [hr(0.0) for _ in range(100)]
    check("hold releases after evidence fades (returns weak x)", abs(released[-1]) < HOLD_THRESH)

    # 6) 保持：反向强证据 -> 重锁反向
    for _ in range(50):
        hr(-1.0)
    check("opposite strong evidence re-latches -", hr.hold_dir < 0 and hr.activations == 2)

    # 7) 基线：raw==b0 -> 无证据 -> 不放方向
    hr = HoldReadout(dt, b0=0.5)
    for _ in range(50):
        hr(0.5)
    check("raw==b0 -> no latch", hr.activations == 0 and hr.hold_left == 0)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
