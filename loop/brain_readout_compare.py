"""P2d-5 任务二：离线复现真实脑 readout 的方向偏置（无电机，无 Pi）。

同 seed、同初始脑状态（brain.reset(64)），对 4 种静态感觉情形各跑 N 步，
记录原始 brain_turn / turn_cmd / omega 的均值、范围、方向。

情形（真实 5 路 ToF 角度 L+70/FL+30/F0/FR-30/R-70；静态 => growth≈0，靠 size 项）：
  A 五路 NO_TARGET（无遮挡）        -> 是否有持续右转偏置？
  B 右前方静态墙 (FR=300mm)
  C 左前方静态墙 (FL=300mm)
  D 左右对称静态通道 (FL=FR=300mm)

不改感觉编码公式 / ToF 掩码 / readout 权重。

运行：  python -m loop.brain_readout_compare [--gain 0.15] [--steps 100]
"""
from __future__ import annotations

import argparse

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import TofFrame
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup

MPC = 0.40
SEED = 64

CONDS = {
    "A_all_no_target": {},
    "B_right_front_wall": {"FR": 300.0},
    "C_left_front_wall": {"FL": 300.0},
    "D_symmetric_channel": {"FL": 300.0, "FR": 300.0},
}


def _frame(seq, t, rr):
    status = {n: ("VALID" if rr.get(n) is not None else "NO_TARGET") for n in ORDER}
    ranges = {n: rr.get(n) for n in ORDER}
    return TofFrame(session_id="cmp", seq=seq, t=t, healthy=True, ranges=ranges,
                    status=status, ages_ms={n: 10.0 for n in ORDER},
                    init_count=0, read_errors=0, reinit_count=0)


def run_cond(cfg, brain, groups, dn_idx, gain, rr, steps):
    brain.reset(SEED)                                     # 同一初始脑状态
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=MPC)
    cfg.decoder.mode = "readout"
    cfg.decoder.memory_gain = 0.0
    cfg.decoder.brain_gain = gain
    cfg.decoder.readout_path = "readout_real.npz"
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    bt, tc, om = [], [], []
    for i in range(steps):
        f = _frame(i, i * brain.dt, rr)
        d = enc.sense(f)
        enc.update(f, d)
        inj, info = enc.inject()
        fired = brain.step(inject=inj)
        dec.update(fired, info=info, stalled=False)
        bt.append(float(dec.last_brain_turn))
        tc.append(float(dec.last_turn_cmd))
        om.append(float(dec.omega))
    return bt, tc, om


def _stat(xs):
    import statistics as st
    return {"mean": st.fmean(xs), "min": min(xs), "max": max(xs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    cfg, maze, car, brain, groups, _e0, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device="cuda", seed=SEED,
        preset=None, sensory_input=False, maze_seed=1000, maze_cells=(6, 4),
        maze_scale=2, maze_loop=0.08, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    print(f"brain={type(brain).__name__} seed={SEED} readout gain={args.gain} steps={args.steps}")
    print(f"{'cond':24s} {'bt_mean':>9s} {'bt_min':>9s} {'bt_max':>9s} "
          f"{'tc_mean':>9s} {'om_mean':>9s} {'om_min':>9s} {'om_max':>9s}  dir")
    for name, rr in CONDS.items():
        bt, tc, om = run_cond(cfg, brain, groups, dn_idx, args.gain, rr, args.steps)
        bs, cs, os_ = _stat(bt), _stat(tc), _stat(om)
        direction = "RIGHT" if os_["mean"] < -1e-4 else ("LEFT" if os_["mean"] > 1e-4 else "~0")
        print(f"{name:24s} {bs['mean']:9.4f} {bs['min']:9.4f} {bs['max']:9.4f} "
              f"{cs['mean']:9.4f} {os_['mean']:9.4f} {os_['min']:9.4f} {os_['max']:9.4f}  {direction}")
    print("\n注：静态墙的感觉得分走 _side 的 size 项（growth≈0）；方向按 omega 均值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
