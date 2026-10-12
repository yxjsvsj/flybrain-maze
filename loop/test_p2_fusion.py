"""P2d-12 任务二回归：P2Fusion 语义一致性（与冻结 readout 组合同式）。"""
from __future__ import annotations

import sys

import numpy as np

from config import Config
from loop.p2_fusion import P2Fusion

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def _fuse():
    cfg = Config()
    cfg.decoder.mode = "readout"
    cfg.decoder.brain_gain = 0.15
    cfg.decoder.memory_gain = 1.0
    return P2Fusion(cfg.decoder, cfg.car), cfg


def _smooth_steady(turns, a, dt_target_fn):
    v = 0.0
    for _ in range(400):
        v = (1 - a) * v + a * dt_target_fn()
    return v


def main() -> int:
    print("=========== P2Fusion semantics ===========")
    f, cfg = _fuse()
    a = cfg.decoder.smooth

    # 1) 零增益：bg=mg=0 -> turn=0 -> omega 稳态 0
    for _ in range(400):
        f.step(brain_turn=0.9, pursuit_turn=0.5, bg=0.0, mg=0.0)
    check("zero gain -> omega ~0", abs(f.omega) < 1e-9)

    # 2) 平滑/裁剪一致性：常量 brain_turn=0.5, bg=0.15 -> turn=0.075 -> 稳态 omega
    f.reset()
    for _ in range(400):
        f.step(brain_turn=0.5, pursuit_turn=0.0, bg=0.15, mg=0.0)
    expect = cfg.car.max_omega * 0.15 * 0.5
    check(f"plain steady omega == max_omega*bg*bt ({f.omega:.5f} vs {expect:.5f})",
          abs(f.omega - expect) < 1e-6)

    # 3) bias 校正：bt 常量 c，bias=c -> turn=0 -> omega 稳态 0（去常驻）
    f.reset()
    for _ in range(400):
        f.step(brain_turn=-0.267, pursuit_turn=0.0, mode="bias", bias=-0.267, bg=0.15, mg=0.0)
    check("bias-corrected constant == bias -> omega ~0", abs(f.omega) < 1e-6)

    # 4) bias 校正：常值 bt=0.5, bias=-0.267 -> turn=0.15*(0.767) -> 稳态
    f.reset()
    for _ in range(400):
        f.step(brain_turn=0.5, pursuit_turn=0.0, mode="bias", bias=-0.267, bg=0.15, mg=0.0)
    exp4 = cfg.car.max_omega * float(np.clip(0.15 * (0.5 + 0.267), -1, 1))
    check(f"bias steady == max_omega*clip(bg*(bt-bias)) ({f.omega:.5f} vs {exp4:.5f})",
          abs(f.omega - exp4) < 1e-6)

    # 5) SHAM：时变 bt 被替换为常数 bias -> 稳态只由 bias 决定
    f.reset()
    for k in range(400):
        f.step(brain_turn=(0.9 if k % 2 else -0.9), pursuit_turn=0.0, mode="sham",
               bias=-0.267, bg=0.15, mg=0.0)
    exp5 = cfg.car.max_omega * float(np.clip(0.15 * (-0.267), -1, 1))
    check(f"sham steady uses constant bias ({f.omega:.5f} vs {exp5:.5f})",
          abs(f.omega - exp5) < 1e-6)

    # 6) 条件门控：神经与 pursuit 反向 -> bg 降为 0.25x
    f.reset()
    _, om_opp = f.step(brain_turn=+0.5, pursuit_turn=-0.5, mode="cond", bg=0.15, mg=1.0)
    turn_opp = float(np.clip(0.15 * 0.25 * 0.5 + 1.0 * (-0.5), -1, 1))
    check("cond gate reduces neural when opposing", abs(om_opp - cfg.car.max_omega * a * turn_opp) < 1e-6)

    # 7) 前向安全：dmin_F <= stop_dist -> v=0
    f.reset()
    v7, _ = f.step(brain_turn=0.0, pursuit_turn=0.0, dmin_F=cfg.decoder.front_stop_dist - 0.01)
    check("front_stop -> v=0", abs(v7) < 1e-9)

    # 8) 首次步与冻结平滑公式一致
    f.reset()
    v8, om8 = f.step(brain_turn=0.4, pursuit_turn=0.2, bg=0.15, mg=1.0)
    turn8 = float(np.clip(0.15 * 0.4 + 1.0 * 0.2, -1, 1))
    check("first step == smooth*a*omega", abs(om8 - cfg.car.max_omega * a * turn8) < 1e-12)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
