"""P2d-6 任务一回归：turn_budget 的物理轮/航向必须与生产 DifferentialOdometry 一致，
覆盖 swap_lr、正/负转向、直行。旧实现（把 UDP left/right 当物理左右）会在此失败。

运行：  python -m loop.test_turn_budget
"""
from __future__ import annotations

import math
import sys

from hardware.odometry import (CPR, LEFT_SIGN, RIGHT_SIGN, TRACK, WHEEL_DIAM,
                               DifferentialOdometry)
from loop.physical_stall import WindowedStallDetector
from loop.turn_budget import physical_wheels

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def prod(dE1, dE2, swap):
    od = DifferentialOdometry(cpr=CPR, wheel_diam=WHEEL_DIAM, track=TRACK,
                              left_sign=LEFT_SIGN, right_sign=RIGHT_SIGN, swap_lr=swap)
    od.update(0, 0)                       # init
    return od.update(dE1, dE2)


CASES = [("straight_fwd", -1000, 1000), ("right_turn", -1000, 1400),
         ("left_turn", -1400, 1000), ("backward", 1000, -1000)]


def main() -> int:
    print("=========== turn_budget vs production DifferentialOdometry ===========")
    for name, dE1, dE2 in CASES:
        for swap in (True, False):
            dl, dr, ds, dth = physical_wheels(dE1, dE2, left_sign=LEFT_SIGN,
                                              right_sign=RIGHT_SIGN, swap_lr=swap,
                                              wheel_diam=WHEEL_DIAM, track=TRACK, cpr=CPR)
            pds, pdth = prod(dE1, dE2, swap)
            check(f"{name} swap={swap}: ds matches production", abs(ds - pds) < 1e-12)
            check(f"{name} swap={swap}: dtheta matches production", abs(dth - pdth) < 1e-12)

    # 语义：swap_lr=True（生产）下，右转 -> dtheta<0，左转 -> dtheta>0，直行 -> 0
    _, _, _, dth_straight = physical_wheels(-1000, 1000, swap_lr=True,
                                            wheel_diam=WHEEL_DIAM, track=TRACK, cpr=CPR)
    _, _, _, dth_right = physical_wheels(-1000, 1400, swap_lr=True,
                                         wheel_diam=WHEEL_DIAM, track=TRACK, cpr=CPR)
    _, _, _, dth_left = physical_wheels(-1400, 1000, swap_lr=True,
                                        wheel_diam=WHEEL_DIAM, track=TRACK, cpr=CPR)
    check(f"straight -> dtheta~0 ({math.degrees(dth_straight):+.3f}deg)",
          abs(math.degrees(dth_straight)) < 1e-6)
    check(f"right turn -> dtheta<0 ({math.degrees(dth_right):+.2f}deg)", dth_right < 0)
    check(f"left turn -> dtheta>0 ({math.degrees(dth_left):+.2f}deg)", dth_left > 0)

    # swap 会翻转同一 raw 差分的航向符号
    _, _, _, a = physical_wheels(-1000, 1400, swap_lr=True,
                                 wheel_diam=WHEEL_DIAM, track=TRACK, cpr=CPR)
    _, _, _, b = physical_wheels(-1000, 1400, swap_lr=False,
                                 wheel_diam=WHEEL_DIAM, track=TRACK, cpr=CPR)
    check("swap_lr flips turn sign", a * b < 0)

    # 生产实测对拍：bg0.10 raw E1=-1607 E2=+1811 -> dtheta≈-8.1deg（带符号）
    _, _, _, dth = physical_wheels(-1607, 1811, swap_lr=True,
                                   wheel_diam=0.067, track=0.194, cpr=1560.0)
    check(f"bg0.10 signed dtheta≈-8.1 ({math.degrees(dth):+.2f}deg)",
          abs(math.degrees(dth) - (-8.10)) < 0.1)

    # windowed stall 轮差 yaw 回退路径（无 theta）符号必须与 odom 一致
    def fallback_turned(swap):
        det = WindowedStallDetector(meters_per_cell=0.40, swap_lr=swap)
        det.observe(0.0, 0, 0, seq=0)
        det.observe(0.02, -1000, 1400, seq=1)   # 物理左更远 -> 右转
        return det.last_turned_rad
    check(f"detector fallback right-turn turned<0 ({fallback_turned(True):+.4f})",
          fallback_turned(True) < 0)
    check(f"detector fallback no-swap flips sign ({fallback_turned(False):+.4f})",
          fallback_turned(False) > 0)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
