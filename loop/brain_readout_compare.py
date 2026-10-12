"""P2d-6 任务二：多 seed × 距离档的离线方向响应（无电机）。

同 seed 重置脑 + 重建 Encoder/Decoder，比较：
  A 五路 NO_TARGET | B FR-only 刺激 | C FL-only 刺激 | D FL/FR 对称刺激
距离档 300/500/800mm。真实 5 路角度、现有 Encoder 参数、现有 readout，均不改。

注意：FR-only=300mm 是**合成单射线**刺激，不等于完整真实右侧墙。

运行：  python -m loop.brain_readout_compare [--gain 0.15] [--steps 100]
"""
from __future__ import annotations

import argparse
import statistics as st

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import TofFrame
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup

MPC = 0.40
SEEDS = [64, 101, 202, 303, 404]
DISTS = [300.0, 500.0, 800.0]
CONDS = {
    "A_no_target": {},
    "B_FR_only": "FR",
    "C_FL_only": "FL",
    "D_FL_FR": "D",
}


def _rr(cond, dist):
    if cond == "A_no_target":
        return {}
    if cond == "B_FR_only":
        return {"FR": dist}
    if cond == "C_FL_only":
        return {"FL": dist}
    return {"FL": dist, "FR": dist}


def _frame(seq, t, rr):
    status = {n: ("VALID" if rr.get(n) is not None else "NO_TARGET") for n in ORDER}
    ranges = {n: rr.get(n) for n in ORDER}
    return TofFrame(session_id="cmp", seq=seq, t=t, healthy=True, ranges=ranges,
                    status=status, ages_ms={n: 10.0 for n in ORDER},
                    init_count=0, read_errors=0, reinit_count=0)


def run_once(cfg, brain, groups, dn_idx, seed, gain, rr, steps):
    brain.reset(seed)
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
    return float(np.mean(bt)), float(np.mean(tc)), float(np.mean(om))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    cfg, maze, car, brain, groups, _e0, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device="cuda", seed=SEEDS[0],
        preset=None, sensory_input=False, maze_seed=1000, maze_cells=(6, 4),
        maze_scale=2, maze_loop=0.08, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    print(f"brain={type(brain).__name__} gain={args.gain} steps/time={args.steps} seeds={SEEDS}")

    # A（距离无关）
    A = []
    for s in SEEDS:
        bt, tc, om = run_once(cfg, brain, groups, dn_idx, s, args.gain, {}, args.steps)
        A.append((bt, tc, om))
    a_om = [x[2] for x in A]
    print(f"\nA no_target          : bt_mean={st.fmean([x[0] for x in A]):+.4f} "
          f"om_mean={st.fmean(a_om):+.4f} om_std={st.pstdev(a_om):.4f} "
          f"dir={'RIGHT' if st.fmean(a_om)<-1e-4 else ('LEFT' if st.fmean(a_om)>1e-4 else '~0')} "
          f"consistency={max(sum(1 for v in a_om if v<0), sum(1 for v in a_om if v>0))}/{len(a_om)}")
    base = st.fmean(a_om)

    print(f"\n{'cond':22s} {'dist':>5s} {'om_mean':>9s} {'om_std':>8s} {'bt_mean':>9s} "
          f"{'-base':>9s} {'dir':>6s} {'consist':>8s}")
    for dist in DISTS:
        for cond in ("B_FR_only", "C_FL_only", "D_FL_FR"):
            rr = _rr(cond, dist)
            oms, bts = [], []
            for s in SEEDS:
                bt, tc, om = run_once(cfg, brain, groups, dn_idx, s, args.gain, rr, args.steps)
                oms.append(om)
                bts.append(bt)
            m = st.fmean(oms)
            neg = sum(1 for v in oms if v < 0)
            pos = sum(1 for v in oms if v > 0)
            d = "RIGHT" if m < -1e-4 else ("LEFT" if m > 1e-4 else "~0")
            print(f"{cond:22s} {dist:5.0f} {m:+9.4f} {st.pstdev(oms):8.4f} "
                  f"{st.fmean(bts):+9.4f} {m-base:+9.4f} {d:>6s} {max(neg,pos)}/{len(oms)}")
    print("\n注：静态墙经 _side 的 size 项入感；-base = 相对 A 的 omega 变化；"
          "consist = 同号 seed 数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
