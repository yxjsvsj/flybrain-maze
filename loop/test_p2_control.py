"""P2d-13 任务一回归：P2Control 统一包络 + turn_source（无需脑，用桩前端）。"""
from __future__ import annotations

import sys

import numpy as np

from config import Config, apply_preset
from loop.p2_control import P2Control, turn_dn_single, turn_swap, turn_zero

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


class _StubDec:
    def __init__(self, cfg):
        self.car_cfg = cfg.car
        self.dt = 0.02
        self.last_brain_turn = 0.0
        self.rates = np.zeros(5)
        self.index = {k: i for i, k in enumerate(("turn_L", "turn_R", "fwd", "back", "escape"))}

    def update(self, fired, info=None, stalled=False):
        return 0.0, 0.0


class _Ctx:
    def __init__(self, bt):
        self.last_brain_turn = bt


def main() -> int:
    print("=========== P2Control fairness envelope ===========")
    cfg = Config()
    apply_preset(cfg, "real")

    # 1) turn_source 纯行为
    c = _Ctx(0.5)
    check("turn_dn_single returns last_brain_turn", turn_dn_single(c) == 0.5)
    check("turn_swap negates", turn_swap(c) == -0.5)
    check("turn_zero is 0", turn_zero(c) == 0.0)

    # 2) 统一基础前进速度 + 只换转向来源
    def build(src, fwd_frac=None):
        return P2Control(cfg.decoder, _StubDec(cfg), src, name="t", fwd_frac=fwd_frac)
    a = build(lambda c: 0.0)
    b = build(lambda c: 0.3)
    check("same base fwd_frac across turn sources", a.fwd_frac == b.fwd_frac)
    check("base fwd_frac == cfg.decoder.readout_fwd_frac",
          a.fwd_frac == cfg.decoder.readout_fwd_frac)
    for _ in range(40):
        va, oa = a.update(np.array([], int), {"dmin_F": 5.0})
    check("no-turn: v -> max_speed*fwd_frac", abs(va - cfg.car.max_speed * a.fwd_frac) < 1e-6)
    check("no-turn: omega -> 0", abs(oa) < 1e-9)

    # 3) 固定转向：omega = max_omega*turn；v 因急转降速而下降
    for _ in range(40):
        vt, ot = b.update(np.array([], int), {"dmin_F": 5.0})
    check("turn 0.3: omega -> max_omega*0.3", abs(ot - cfg.car.max_omega * 0.3) < 1e-6)
    check("turn 0.3: v < base (turn_slowdown)", vt < cfg.car.max_speed * b.fwd_frac)

    # 4) 前方安全停车（统一）：dmin_F <= front_stop -> v=0
    zb = build(lambda c: 0.0)
    zb.update(np.array([], int), {"dmin_F": 0.1})
    check("front-stop: v=0", zb.v == 0.0)
    check("front-stop: front_blocked set", zb.front_blocked is True)
    check("front-stop: front_blocks counted", zb.front_blocks == 1)

    # 5) 前方减速（统一）：stop < dmin_F < slow -> 0 < v < base
    sb = build(lambda c: 0.0)
    for _ in range(40):
        vs, _ = sb.update(np.array([], int), {"dmin_F": 0.6})
    base = cfg.car.max_speed * sb.fwd_frac
    check("front-slow: 0 < v < base", 0.0 < vs < base)

    # 6) 转向饱和计数
    ab = build(lambda c: 1.0)
    for _ in range(10):
        ab.update(np.array([], int), {"dmin_F": 5.0})
    check("saturation counted at |turn|>=1", ab.sat_steps == 10 and ab.saturation_ratio() == 1.0)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
