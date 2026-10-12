"""P2d-12a 回归：local_avoidance_bench 验证台的可靠性。

确保测试台能可靠区分：
  - 直行可通过 vs 直行必撞；
  - 必须左转 vs 必须右转（开口在相反侧）；
  - 圆盘可行路径存在；
  - collision_events（撞墙次数）与 collision_steps（接触脑步数）是两个不同的量。
"""
from __future__ import annotations

import sys

from loop.local_avoidance_bench import (CAR_R, SCENARIOS, _disk_free, geometry_check,
                                        straight_pass, wall_clearance)
from world.car import DiffDriveCar
from world.maze import Maze

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== local_avoidance_bench reliability ===========")

    # 1) 开阔直行走廊：直行应能到达终点（对照，证明检查器会返回 True）
    straight_ok = Maze(["########", "#S....G#", "#......#", "########"])
    _sx, _sy, _sth = straight_ok.start
    sp, _sl = straight_pass(straight_ok, _sx, _sy, _sth, CAR_R)
    check("open straight corridor -> straight_pass=True", sp is True)

    # 2) 三个真实场景：直行必撞 + 圆盘可行路径存在
    for name in ("must_left", "must_right", "s_shape"):
        g = geometry_check(SCENARIOS[name])
        check(f"{name}: straight_pass=False (直行必撞)", g["straight_pass"] is False)
        check(f"{name}: path_exists=True (圆盘有可行路径)", g["path_exists"] is True)

    # 3) 必须左转 vs 必须右转：开口在相反侧（goal 相对起点的 y 方向相反）
    gl = geometry_check(SCENARIOS["must_left"])
    gr = geometry_check(SCENARIOS["must_right"])
    check("must_left goal is above start (goal_dy<0)", gl["goal_dy"] < 0)
    check("must_right goal is below start (goal_dy>0)", gr["goal_dy"] > 0)
    check("must_left / must_right require opposite turns", (gl["goal_dy"] < 0) != (gr["goal_dy"] < 0))

    # 4) 圆盘几何：格中心可放置，贴墙处不行
    m = straight_ok
    check("disk_free at cell center", _disk_free(m, 1.5, 1.5, CAR_R) is True)
    check("disk not free inside a wall", _disk_free(m, 0.5, 0.5, CAR_R) is False)
    check("wall_clearance at center > 0", wall_clearance(m, 1.5, 1.5, CAR_R) > 0)

    # 5) collision_events 与 collision_steps 是不同的量：
    #    正对墙持续顶（磨墙）-> steps 很大，events 只记"从无接触变为有接触"的次数。
    #    每步重新下令前冲（真实控制器会一直想往前走）。
    wall = Maze(["####", "#S##", "#..#", "####"])
    car = DiffDriveCar(1.5, 1.5, 0.0, __import__("config").Config().car)
    for _ in range(200):
        car.set_command(car.cfg.max_speed, 0.0)
        car.step(wall, 0.02)
    check("head-on grind: collision_events == 1", car.collision_events == 1)
    check("head-on grind: collision_steps >> events", car.collisions > car.collision_events)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
