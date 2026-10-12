"""P2d-4 准备：runner 统计里程的轮径一致性（对齐生产 0.067）。

问题：`hardware.odometry` 模块默认 `WHEEL_DIAM = 0.070`（名义），而生产 Pi 里程计跑
标定的 `--wheel-diam 0.067`。runner 的 `OdomRealCar` 之前用模块默认 → `car.distance`
有 ~4.5% 偏置。本测试锁定：runner 用 0.067，且与名义值有可辨差异。

运行：  python -m loop.test_odom_mpc
"""
from __future__ import annotations

import math
import sys

from config import CarConfig
from hardware.odometry import CPR, LEFT_SIGN, RIGHT_SIGN, WHEEL_DIAM
from hardware.odometry_client import OdomSample
from loop.odom_shadow_run import ODOM_WHEEL_DIAM, OdomRealCar

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


class FakeClient:
    def __init__(self, samples):
        self._s = list(samples)
        self._i = 0

    def latest(self):
        s = self._s[min(self._i, len(self._s) - 1)]
        self._i += 1
        return s, 0.01

    def age(self):
        return 0.01

    def stop(self):
        pass

    def stats(self):
        return {}


def _mk_car(samples, mpc=0.40):
    return OdomRealCar(2.5, 2.5, 0.0, CarConfig(), FakeClient(samples), mpc)  # type: ignore[arg-type]


def main() -> int:
    print("=========== runner odom wheel-diameter consistency ===========")
    check(f"ODOM_WHEEL_DIAM == 0.067 (got {ODOM_WHEEL_DIAM})", abs(ODOM_WHEEL_DIAM - 0.067) < 1e-12)
    check(f"module WHEEL_DIAM still nominal 0.070 (got {WHEEL_DIAM})", abs(WHEEL_DIAM - 0.070) < 1e-12)
    check("runner calibrated != module nominal", abs(ODOM_WHEEL_DIAM - WHEEL_DIAM) > 1e-9)

    MPC = 0.40
    car = _mk_car([], MPC)
    expect = math.pi * 0.067 / CPR
    check(f"m_per_count == pi*0.067/CPR ({car.m_per_count:.9g})",
          abs(car.m_per_count - expect) < 1e-15)

    # synthetic forward: LEFT_SIGN=-1 (left counts decrease), RIGHT_SIGN=+1 (right increases)
    n = 1000
    s0 = OdomSample(session_id="s", seq=0, t=0.0, left=0, right=0, x=0.0, y=0.0, theta=0.0)
    s1 = OdomSample(session_id="s", seq=1, t=0.02, left=-n, right=+n, x=0.0, y=0.0, theta=0.0)
    c = _mk_car([s0, s1], MPC)
    c.step(None, 0.02)          # first step: capture _pl/_pr, distance stays 0
    c.step(None, 0.02)          # delta n counts per wheel
    counts = 0.5 * (LEFT_SIGN * (-n) + RIGHT_SIGN * (n))     # = n
    dist_true = counts * math.pi * 0.067 / CPR
    got = c.distance * MPC
    check(f"distance uses 0.067 (got {got:.6f} m, true {dist_true:.6f} m)",
          abs(got - dist_true) < 1e-9)
    dist_nom = counts * math.pi * 0.070 / CPR
    pct = (dist_nom / dist_true - 1.0) * 100
    check(f"nominal 0.070 over-reports ~4.5% ({pct:.2f}%)", dist_nom > dist_true and 3.5 < pct < 5.5)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
