"""P2d-14 回归：方向坐标审计 + 短时转向保持（纯 numpy，无需脑）。

覆盖：持续强输入不应升级为满舵；保持用神经实际方向+幅度且计时有界；短暂脉冲；
信号消失后到期释放；反向神经证据重锁；无证据不发明方向。
"""
from __future__ import annotations

import sys

import numpy as np

from config import Config
from loop.local_avoidance_bench import SCENARIOS
from loop.turn_hold import HOLD_MAG_CAP, HOLD_S, HOLD_THRESH, HoldReadout, required_omega_sign
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

    # 3) 弱输入不发明方向
    hr = HoldReadout(dt, b0=0.0)
    check("weak input -> no invention", abs(hr(0.0)) < 1e-6)

    # 4) 持续强输入：由神经直接控制，**不升级为满舵**
    hr = HoldReadout(dt, b0=0.0)
    vals = [hr(0.5) for _ in range(200)]        # 神经 EMA 稳态 ~0.5（>=thresh 但 <1）
    check("sustained strong input -> returns neural value (~0.5), not full-lock 1.0",
          abs(vals[-1] - 0.5) < 0.02 and vals[-1] < 0.9)
    check("hold timer bounded while sustained (<= hold_steps)", hr.hold_left <= hr.hold_steps)

    # 5) 短暂脉冲：保持用神经实际方向+幅度，不放大
    hr = HoldReadout(dt, b0=0.0)
    for _ in range(40):
        hr(0.5)                                  # 脉冲，神经 EMA ~0.4（>=thresh）
    after = [hr(0.0) for _ in range(12)]         # 证据衰减到阈值下后，保持接管
    check("brief pulse -> hold keeps pulse sign", after[-1] > 0)
    check("hold magnitude bounded by neural input, not escalated to cap",
          abs(after[-1]) <= 0.5 + 1e-9 and abs(after[-1]) < HOLD_MAG_CAP
          and abs(after[-1] - hr.hold_mag) < 1e-9)

    # 6) 信号消失 -> 到期释放回弱 x
    rel = [hr(0.0) for _ in range(int(2 * (HOLD_S / dt)) + 40)]
    check("expires and releases to weak x", abs(rel[-1]) < HOLD_THRESH)

    # 7) 反向神经证据 -> 重锁反向
    hr = HoldReadout(dt, b0=0.0)
    for _ in range(60):
        hr(1.0)
    for _ in range(60):
        hr(-1.0)
    check("opposite neural evidence re-latches -", hr.hold_dir < 0)

    # 8) 基线：raw==b0 -> 无证据 -> 不保持
    hr = HoldReadout(dt, b0=0.5)
    for _ in range(60):
        hr(0.5)
    check("raw==b0 -> no hold", hr.activations == 0 and hr.hold_left == 0)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
