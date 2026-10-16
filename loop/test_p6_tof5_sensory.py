"""P6-1 回归：p6_tof5_sensory 的纯逻辑（五路方位权重 / 聚合 / 场景帧 / 冻结常量）。无模拟器。"""
from __future__ import annotations

import sys

import numpy as np

from loop.p6_tof5_sensory import (ANGLES, APPROACH, FAR_M, K_SIG, NEAR_M, ORDER, SEEDS,
                                  _scenario_frames, aggregate, weights)
from config import EncoderConfig, CarConfig

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== p6_tof5_sensory ===========")
    cfg = EncoderConfig(); car = CarConfig()
    wL, wR = weights()
    check("weights sum to 1 per sensor", np.allclose(wL + wR, 1.0))
    check("all 5 sensors affect both sides (wL,wR>0)", (wL > 0).all() and (wR > 0).all())
    check("L(+70) weights left more", wL[ORDER.index("L")] > wR[ORDER.index("L")])
    check("R(-70) weights right more", wR[ORDER.index("R")] > wL[ORDER.index("R")])
    check("azimuth order preserved (wL descending with angle)",
          np.all(np.diff(wL) <= 1e-9))

    # 近障聚合：L 侧 near -> loom_L > loom_R
    far = np.full(5, FAR_M)
    agg0 = aggregate(far, 0.1, far * 0, cfg, car.max_speed)
    nearL = far.copy(); nearL[ORDER.index("L")] = NEAR_M
    aggL = aggregate(nearL, 0.1, far * 0, cfg, car.max_speed)
    check("L near -> loom_L > loom_R", aggL["loom_L"] > aggL["loom_R"])
    nearR = far.copy(); nearR[ORDER.index("R")] = NEAR_M
    aggR = aggregate(nearR, 0.1, far * 0, cfg, car.max_speed)
    check("R near -> loom_R > loom_L", aggR["loom_R"] > aggR["loom_L"])
    check("near raises loom above far baseline", aggL["loom_L"] > agg0["loom_L"] + 0.5)

    # 逼近产生正 growth
    d1 = far.copy(); d2 = far.copy(); d2[ORDER.index("F")] = 0.5
    a1 = aggregate(d1, 0.1, far * 0, cfg, car.max_speed)
    a2 = aggregate(d2, 0.1, a1["size"], cfg, car.max_speed)
    check("approach produces growth", a2["growth"].max() > 0)

    # 场景帧
    F = _scenario_frames("FL", "near")
    check("near FL sets FL channel near", abs(F[50, ORDER.index("FL")] - NEAR_M) < 1e-6)
    check("none scenario all far", np.allclose(_scenario_frames("none", "none"), FAR_M))
    Fa = _scenario_frames("L", "approach")
    check("approach starts far ends near", Fa[25, ORDER.index("L")] > Fa[74, ORDER.index("L")])

    check("frozen consts", SEEDS == [11, 12, 13, 14, 15] and K_SIG == 3.0 and set(ORDER) == set(ANGLES))

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
