"""P1 synthetic 静态对照 v2（只读冻结 P1 `Encoder`，不改 P1）。

三种对照（同一平行墙 raycast 场景、同一脑初态、同量纲、同观察时长）：

  A  P1 original          15 rays / FOV 240 / avoidance ±60   （实际入掩码 ±64.3,±48,±32,±16,0 中 ≤±60 的）
  B  P1 narrow mask       15 rays / FOV 240 / avoidance ±30   （实际入掩码 ±17.14, 0）
  C  physical-angle       精确 FL+30 / F 0 / FR-30，按 P2 Encoder 方式算注入
  D  physical + extrinsics 五个实测光学窗口外参 + 理想侧墙命中（安装位置影响）

走廊矩阵：0.60 / 0.50 / 0.40 / 0.35 m；`meters_per_cell=0.40`。
**静态场景下 sense 恒定 → 50Hz vs 10Hz 零阶保持无差异**（本工具按 P1 原 50Hz 每步 sense）。

输出每个 (mode,width)：dmin_L/R/F、size_L/R、chase/loom/threat_L/R、bias、
brain_turn(mean/std/min/max)、feat_l2/mean/std/DN；并另存每 combo 的 1314 维 feats 均值
向量供 per-dimension 对比。**不改 readout / brain_gain / omega_sign / 注入公式 / 冻结 P1。**
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import sys

import numpy as np

from loop.decode import Decoder
from loop.encode import Encoder
from loop.run import make_setup
from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER

ORDER5 = ORDER


class ParallelWalls:
    """理想平行墙 raycast（+y=左，-y=右）。side: both/right/left。"""

    def __init__(self, half_width: float, side: str = "both"):
        self.hw = float(half_width)
        self.side = side

    def raycast_many(self, x, y, angles, max_range):
        out = []
        for a in np.asarray(angles, dtype=float):
            dy = math.sin(a)
            if abs(dy) < 1e-9:
                out.append(float(max_range)); continue
            if dy > 0:
                if self.side == "right":
                    out.append(float(max_range)); continue
                wall = self.hw
            else:
                if self.side == "left":
                    out.append(float(max_range)); continue
                wall = -self.hw
            t = (wall - y) / dy
            out.append(float(min(max_range, max(0.05, t))))
        return np.array(out, np.float32)


class _Car:
    def __init__(self, x, y, theta):
        self.pose = (x, y, theta)


class PhysicalAngleEncoder(Encoder):
    """C 模式：只用 ±30/0 三路（P2 神经通道的角度），复用冻结 `_side` 公式。"""

    def __init__(self, groups, cfg, brain_dt, car_max_speed, car_radius,
                 angles_deg=(30.0, 0.0, -30.0)):
        super().__init__(groups, cfg, brain_dt, car_max_speed, car_radius)
        self.angles = np.deg2rad(np.array(angles_deg, dtype=float))
        self.left = np.array([True, True, False])      # FL(+30), F(0)
        self.right = np.array([False, True, True])     # F(0), FR(-30)
        self.front = np.abs(self.angles) <= np.deg2rad(cfg.front_cone_deg / 2)


def _build_encoder(mode, groups, cfg, brain, enc_cfg):
    if mode == "A":
        return Encoder(groups, enc_cfg, brain.dt, car_max_speed=cfg.car.max_speed,
                       car_radius=cfg.car.radius)
    if mode == "B":
        e = Encoder(groups, enc_cfg, brain.dt, car_max_speed=cfg.car.max_speed,
                    car_radius=cfg.car.radius)
        avoid_half = np.deg2rad(30.0)
        center = np.isclose(e.angles, 0.0, atol=1e-9)
        fwd = np.abs(e.angles) <= avoid_half
        e.left = fwd & ((e.angles > 0) | center)
        e.right = fwd & ((e.angles < 0) | center)
        return e
    if mode == "C":
        return PhysicalAngleEncoder(groups, enc_cfg, brain.dt, cfg.car.max_speed,
                                    cfg.car.radius)
    raise ValueError(mode)


def run_combo(mode, half_width_cells, cfg, brain, groups, enc_cfg, dec, steps):
    enc = _build_encoder(mode, groups, cfg, brain, enc_cfg)
    maze = ParallelWalls(half_width_cells, side="both")
    car = _Car(0.0, 0.0, 0.0)
    trace, readout = dec.trace, dec.readout
    feat_sum = None
    acc = {k: [] for k in ("dmin_L", "dmin_R", "dmin_F", "size_L", "size_R",
                           "chase_L", "chase_R", "loom_L", "loom_R",
                           "threat_L", "threat_R", "prox_L", "prox_R", "bias",
                           "brain", "feat_l2", "feat_mean", "feat_std", "DN")}
    for step in range(steps):
        dists = enc.sense(maze, car)
        loom_l, chase_l, thr_l, size_l, dmin_l = enc._side(dists, enc.left, "L")
        loom_r, chase_r, thr_r, size_r, dmin_r = enc._side(dists, enc.right, "R")
        dmin_f = max(float(dists[enc.front].min()) - enc.car_radius, 0.05)
        inject = []
        for name, dv in (("loom_L", loom_l), ("loom_R", loom_r), ("chase_L", chase_l),
                         ("chase_R", chase_r), ("threat_L", thr_l), ("threat_R", thr_r)):
            if dv > 0:
                inject.append((enc.groups[name], float(dv * enc.per_step)))
        if enc_cfg.forward_drive_dv > 0:
            inject.append((enc.groups["fwd"], float(enc_cfg.forward_drive_dv * enc.per_step)))
        fired = brain.step(inject=inject)
        feats = np.asarray(trace.observe(fired), dtype=np.float32)
        raw = float(readout.predict(feats))
        pl = 1.0 - dmin_l / enc_cfg.max_range
        pr = 1.0 - dmin_r / enc_cfg.max_range
        acc["dmin_L"].append(dmin_l); acc["dmin_R"].append(dmin_r); acc["dmin_F"].append(dmin_f)
        acc["size_L"].append(size_l); acc["size_R"].append(size_r)
        acc["chase_L"].append(chase_l); acc["chase_R"].append(chase_r)
        acc["loom_L"].append(loom_l); acc["loom_R"].append(loom_r)
        acc["threat_L"].append(thr_l); acc["threat_R"].append(thr_r)
        acc["prox_L"].append(pl); acc["prox_R"].append(pr); acc["bias"].append(pr - pl)
        acc["brain"].append(raw)
        acc["feat_l2"].append(float(np.linalg.norm(feats)))
        acc["feat_mean"].append(float(feats.mean())); acc["feat_std"].append(float(feats.std()))
        acc["DN"].append(int(len(fired)))
        feat_sum = feats.astype(np.float64) if feat_sum is None else feat_sum + feats
    n = max(1, len(acc["DN"]))
    return acc, (feat_sum / n if feat_sum is not None else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P1 对照 v2（A/B/C 掩码/角度矩阵）")
    ap.add_argument("--modes", default="A,B,C")
    ap.add_argument("--widths", default="0.60,0.50,0.40,0.35",
                    help="走廊实宽(m) 列表")
    ap.add_argument("--half-width-cells", type=float, default=None,
                    help="只跑单一 half-width（兼容旧用法）")
    ap.add_argument("--side", choices=["both", "right", "left"], default="both")
    ap.add_argument("--meters-per-cell", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--readout", default="readout_real.npz")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--csv", default="")
    ap.add_argument("--feats-npz", default="p1_compare_feats.npz")
    args = ap.parse_args(argv)

    cfg, _m0, _c0, brain, groups, _e0, _ = make_setup(
        brain_kind="real", map_name="track", device=args.device, seed=args.seed,
        sensory_input=False, make_decoder=False, verbose=False)
    cfg.decoder.mode = "readout"; cfg.decoder.memory_gain = 0.0; cfg.decoder.brain_gain = 1.0
    cfg.decoder.readout_path = args.readout
    enc_cfg = cfg.encoder
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    if dec.trace is None or dec.readout is None:
        print("[p1] 需要 readout；退出"); return 2
    steps = max(1, int(round(args.seconds / brain.dt)))

    widths = ([args.half_width_cells] if args.half_width_cells is not None
              else [float(w) / args.meters_per_cell / 2.0 for w in args.widths.split(",")])
    modes = args.modes.split(",")
    rows = []
    feats_store = {}
    print(f"[p1v2] modes={modes}  widths(cells)={[round(w,3) for w in widths]}  steps={steps}")
    for w in widths:
        for mode in modes:
            try:
                brain.reset(args.seed)
            except Exception:                                   # noqa: BLE001
                pass
            acc, fmean = run_combo(mode, w, cfg, brain, groups, enc_cfg, dec, steps)
            brain_turn = acc["brain"]
            feat_dims = fmean if fmean is not None else np.zeros(1)
            key = f"{mode}@{w*args.meters_per_cell*2:.2f}m"
            feats_store[key] = feat_dims.astype(np.float32)
            r = dict(mode=mode, width_m=w * args.meters_per_cell * 2,
                     dmin_L=st.mean(acc["dmin_L"]), dmin_R=st.mean(acc["dmin_R"]),
                     dmin_F=st.mean(acc["dmin_F"]),
                     size_L=st.mean(acc["size_L"]), size_R=st.mean(acc["size_R"]),
                     chase_L=st.mean(acc["chase_L"]), chase_R=st.mean(acc["chase_R"]),
                     loom_L=st.mean(acc["loom_L"]), loom_R=st.mean(acc["loom_R"]),
                     threat_L=st.mean(acc["threat_L"]), threat_R=st.mean(acc["threat_R"]),
                     bias=st.mean(acc["bias"]),
                     brain_mean=st.mean(brain_turn),
                     brain_std=st.pstdev(brain_turn) if len(brain_turn) > 1 else 0.0,
                     brain_min=min(brain_turn), brain_max=max(brain_turn),
                     feat_l2=st.mean(acc["feat_l2"]), feat_mean=st.mean(acc["feat_mean"]),
                     feat_std=st.mean(acc["feat_std"]), DN=st.mean(acc["DN"]))
            rows.append(r)
            print(f"  {key:>10}: chase_L/R={r['chase_L']:.3f}/{r['chase_R']:.3f} "
                  f"dmin_L/R={r['dmin_L']:.3f}/{r['dmin_R']:.3f} "
                  f"bias={r['bias']:+.3f} brain={r['brain_mean']:+.3f} "
                  f"[{r['brain_min']:+.3f},{r['brain_max']:+.3f}] "
                  f"feat_l2={r['feat_l2']:.2f} DN={r['DN']:.0f}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            wr = _csv.writer(fh); wr.writerow(list(rows[0].keys()))
            for r in rows:
                wr.writerow([r[k] for k in rows[0].keys()])
        print(f"[p1v2] csv -> {args.csv}")
    if feats_store:
        np.savez_compressed(args.feats_npz, **feats_store)
        print(f"[p1v2] feats -> {args.feats_npz}  ({len(feats_store)} combos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
