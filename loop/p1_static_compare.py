"""P1 synthetic 静态对照（只读冻结 P1 `Encoder`，不改 P1）。

用**平行墙** raycast 构造对应静态场景（车在正中、头朝 +x），走**冻结 P1** 的
15 rays / 240° FOV / ±60° avoidance mask，产出与 `physical_static_diag` 兼容的指标，
供与 P2 五路实测 ToF 对比：

  dmin_L/R/F, size_L/R, growth_L/R, chase_L/R, loom_L/R, threat_L/R,
  prox_L/R, bias, DN feat 统计, readout_raw/clipped, brain_turn

重要差异（**不隐瞒、也不偷偷改 P1**）
------------------------------------
- P1 用 **15 条理想射线**；P2 用 **5 路实测光轴**。本工具**不**把 15 裁剪成 5，
  而是让冻结 Encoder 照常吃它的 15 rays；对照时看的是**同名字段**的数量级，
  以及"P1 若只有 5 路光轴"应另做采样对照（本工具不做）。
- P1 sensing 每 brain.dt(50Hz)；P2 ToF ~10Hz 帧间零阶保持。本工具**按 P1 原条件**
  每步 sense（静态场景下每步 raycast 相同）。时间对齐另做，不在本轮。
- 每个场景独立重建 brain（固定 seed），不跨场景继承神经状态。

用法
----
  python -m loop.p1_static_compare --half-width-cells 1.25   # 对应 1.0m corridor
  python -m loop.p1_static_compare --half-width-cells 0.5    # 对应 0.4m corridor
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


class ParallelWalls:
    """理想平行墙 raycast：两面墙在 y=±half_width(cells)，无限长，无前后墙。"""

    def __init__(self, half_width: float):
        self.hw = float(half_width)

    def raycast_many(self, x, y, angles, max_range):
        out = []
        for a in np.asarray(angles, dtype=float):
            dy = math.sin(a)
            if abs(dy) < 1e-9:
                out.append(float(max_range))
                continue
            wall = self.hw if dy > 0 else -self.hw
            t = (wall - y) / dy
            out.append(float(min(max_range, max(0.05, t))))
        return np.array(out, np.float32)


class _Car:
    def __init__(self, x, y, theta):
        self.pose = (x, y, theta)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P1 synthetic 静态对照（只读冻结 Encoder）")
    ap.add_argument("--half-width-cells", type=float, required=True,
                    help="平行墙半宽（cells）。1.0m corridor -> 1.25")
    ap.add_argument("--scroll-cells", type=float, default=6.0)
    ap.add_argument("--meters-per-cell", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--readout", default="readout_real.npz")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="弃掉前若干秒（首帧瞬态），默认 0（与 P2 static 一致保留全部）")
    args = ap.parse_args(argv)

    cfg, _m0, _c0, brain, groups, _e0, _ = make_setup(
        brain_kind="real", map_name="track", device=args.device, seed=args.seed,
        sensory_input=False, make_decoder=False, verbose=False)
    cfg.decoder.mode = "readout"
    cfg.decoder.memory_gain = 0.0
    cfg.decoder.brain_gain = 1.0
    cfg.decoder.readout_path = args.readout
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    trace, readout = dec.trace, dec.readout
    if trace is None or readout is None:
        print("[p1] 需要 --mode readout + 有效 --readout；退出")
        return 2

    maze = ParallelWalls(args.half_width_cells)
    car = _Car(0.0, 0.0, 0.0)      # 车在正中、头朝 +x
    enc = Encoder(groups, cfg.encoder, brain.dt, car_max_speed=cfg.car.max_speed,
                  car_radius=cfg.car.radius)

    steps = max(1, int(round(args.seconds / brain.dt)))
    warm = int(round(args.warmup / brain.dt))
    acc = {k: [] for k in ("dmin_L", "dmin_R", "dmin_F", "size_L", "size_R",
                           "growth_L", "growth_R", "chase_L", "chase_R",
                           "loom_L", "loom_R", "threat_L", "threat_R",
                           "prox_L", "prox_R", "bias",
                           "brain_raw", "brain_clip", "feat_l2", "feat_mean",
                           "feat_std", "DN")}
    for step in range(steps):
        dists = enc.sense(maze, car)
        # 直接用冻结 Encoder 的 _side（只读）拿到 size/chase/threat/loom/dmin；
        # 它内部更新 prev_size（growth 用 next 帧 size 差，静态场景 ~0）。
        loom_l, chase_l, thr_l, size_l, dmin_l = enc._side(dists, enc.left, "L")
        loom_r, chase_r, thr_r, size_r, dmin_r = enc._side(dists, enc.right, "R")
        dmin_f = max(float(dists[enc.front].min()) - enc.car_radius, 0.05)
        inject = []
        for name, dv in (("loom_L", loom_l), ("loom_R", loom_r), ("chase_L", chase_l),
                         ("chase_R", chase_r), ("threat_L", thr_l), ("threat_R", thr_r)):
            if dv > 0:
                inject.append((enc.groups[name], float(dv * enc.per_step)))
        if cfg.encoder.forward_drive_dv > 0:
            inject.append((enc.groups["fwd"], float(cfg.encoder.forward_drive_dv * enc.per_step)))
        fired = brain.step(inject=inject)
        feats = np.asarray(trace.observe(fired), dtype=np.float32)
        raw = float(readout.predict(feats))
        if step < warm:
            continue
        prox_l = 1.0 - dmin_l / enc.cfg.max_range
        prox_r = 1.0 - dmin_r / enc.cfg.max_range
        acc["dmin_L"].append(dmin_l); acc["dmin_R"].append(dmin_r); acc["dmin_F"].append(dmin_f)
        acc["size_L"].append(size_l); acc["size_R"].append(size_r)
        acc["chase_L"].append(chase_l); acc["chase_R"].append(chase_r)
        acc["loom_L"].append(loom_l); acc["loom_R"].append(loom_r)
        acc["threat_L"].append(thr_l); acc["threat_R"].append(thr_r)
        acc["prox_L"].append(prox_l); acc["prox_R"].append(prox_r)
        acc["bias"].append(float(prox_r - prox_l))
        acc["brain_raw"].append(raw); acc["brain_clip"].append(float(np.clip(raw, -1, 1)))
        acc["feat_l2"].append(float(np.linalg.norm(feats)))
        acc["feat_mean"].append(float(feats.mean())); acc["feat_std"].append(float(feats.std()))
        acc["DN"].append(int(len(fired)))

    def s(k):
        v = acc[k]
        return f"mean={st.mean(v):+.4f} min={min(v):+.4f} max={max(v):+.4f}" if v else "-"

    print(f"[p1] half_width={args.half_width_cells} cells = "
          f"{args.half_width_cells*args.meters_per_cell*2:.2f} m corridor   steps={len(acc['DN'])}")
    for k in ("dmin_L", "dmin_R", "dmin_F", "size_L", "size_R",
              "chase_L", "chase_R", "loom_L", "loom_R", "threat_L", "threat_R",
              "prox_L", "prox_R", "bias", "brain_raw", "brain_clip",
              "feat_l2", "feat_mean", "feat_std", "DN"):
        print(f"  {k:>10}: {s(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
