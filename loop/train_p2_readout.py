"""P2d-12 任务三：训练 P2 五路 ToF 专用 readout 候选（不修改 loop/train_readout.py）。

老师 = **局部避障**（非旧右侧贴墙）：
    turn_target = clip(k * (prox_R - prox_L), -1, 1)   # 正=偏左
    -> 无障碍≈0；FR 近(prox_R 高)->偏左；FL 近(prox_L 高)->偏右；双侧按剩余空间。
    仅用当前/历史允许的感知（不含隐藏地图/全局目标/Dijkstra）；前方停车是独立层，不伪装成神经读出。
特征 = 1314 个下行神经元 Trace；连接组固定，仅训线性 readout。teacher 驱动采数据，可选 1 轮 DAgger。

**不覆盖原 readout**：默认 `readout_real_p2_5tof_candidate.npz`。
用法：
  python -m loop.train_p2_readout --brain real --train-seeds 3001 3002 3003 \
      --val-seeds 3051 3052 --seconds 30 --out readout_real_p2_5tof_candidate.npz
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.dynamic_tof_sim import frame_from_pose, load_rig
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup
from world.car import DiffDriveCar
from world.maze import generate, Maze


def teacher_turn(info, k=3.0):
    """局部避障老师（仅用当前感知）；正=偏左。"""
    return float(np.clip(k * (info.get("prox_r", 0.0) - info.get("prox_l", 0.0)), -1.0, 1.0))


def rollout(cfg, brain, groups, dn_idx, enc_proto, seeds, seconds, dt, rig, mpc,
            drive_fn, cells=(6, 4), scale=2, loop=0.08, seed0=64):
    from flybrain import Trace
    X, Y = [], []
    for s in seeds:
        maze = Maze(generate(cells[0], cells[1], seed=s, scale=scale, loop_chance=loop))
        sx, sy, sth = maze.start
        car = DiffDriveCar(sx, sy, sth, cfg.car)
        enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                                 sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                                 car_radius=cfg.car.radius, meters_per_cell=mpc)
        trace = Trace(brain, idx=dn_idx, tau=cfg.decoder.readout_tau)
        brain.reset(seed0)
        n = int(round(seconds / dt))
        for step in range(n):
            fr = frame_from_pose(maze, car.x, car.y, car.theta, step * dt, step,
                                 rig=rig, meters_per_cell=mpc)
            d = enc.sense(fr)
            enc.update(fr, d)
            inj, info = enc.inject()
            fired = brain.step(inject=inj)
            feats = trace.observe(fired)
            turn_t = teacher_turn(info)
            X.append(feats)
            Y.append(turn_t)
            v, omega = drive_fn(info, feats, turn_t, cfg)
            car.set_command(v, omega)
            car.step(maze, dt)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                           # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["fake", "real"], default="real")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-seeds", nargs="+", type=int, default=[3001, 3002, 3003])
    ap.add_argument("--val-seeds", nargs="+", type=int, default=[3051, 3052])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--dagger", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--components", nargs="+", type=int, default=[10, 40, 120])
    ap.add_argument("--meters-per-cell", type=float, default=0.40)
    ap.add_argument("--out", default="readout_real_p2_5tof_candidate.npz")
    args = ap.parse_args(argv)

    from flybrain import Readout

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=64,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    cfg.decoder.mode = "readout"
    dn = np.asarray(brain.cells(["descending_neuron"]))
    rig = load_rig()
    mpc = args.meters_per_cell

    def drive_teacher(info, feats, turn_t, cfg):
        v = cfg.car.max_speed * cfg.decoder.readout_fwd_frac
        v *= 1.0 - cfg.decoder.turn_slowdown * abs(turn_t)
        return v, cfg.car.max_omega * turn_t

    t0 = time.perf_counter()
    print(f"[data] 老师驱动采集 {len(args.train_seeds)} 局 x {args.seconds}s ...")
    X, Y = rollout(cfg, brain, groups, dn, None, args.train_seeds, args.seconds, brain.dt,
                   rig, mpc, drive_teacher)
    print(f"       样本 {X.shape[0]} 用时 {time.perf_counter()-t0:.1f}s, "
          f"label mean {Y.mean():+.3f} std {Y.std():.3f}")

    rd = None
    for rnd in range(args.dagger + 1):
        Xs, Ys = X[::args.stride], Y[::args.stride]
        rd = Readout.fit(Xs, Ys, kind="ridge", components=tuple(args.components),
                         lambdas=(1e-3, 1e-2, 1e-1, 1.0), verbose=True)
        print(f"[train {rnd}] cv neg-MSE {rd.cv_score:+.5f} ({rd.components} PCs, lam {rd.lam:g})")
        rd.save(args.out)
        if rnd == args.dagger:
            break
        Xn, Yn = rollout(cfg, brain, groups, dn, None, args.train_seeds, args.seconds, brain.dt,
                         rig, mpc, lambda info, f, t, c: (c.car.max_speed * c.decoder.readout_fwd_frac,
                                                          c.car.max_omega * float(np.clip(rd.predict(f), -1, 1))))
        X = np.concatenate([X, Xn])
        Y = np.concatenate([Y, Yn])
        print(f"[train {rnd+1}] DAgger 新增 {Xn.shape[0]}，累计 {X.shape[0]}")

    # 验证集：与老师的 MSE + 无遮挡偏置
    print(f"\n[eval] 验证迷宫 {args.val_seeds} ...")
    Xe, Ye = rollout(cfg, brain, groups, dn, None, args.val_seeds, args.seconds, brain.dt,
                     rig, mpc, drive_teacher)
    mse = float(np.mean((rd.predict(Xe) - Ye) ** 2))
    base = float(np.mean((Ye - Ye.mean()) ** 2))
    print(f"  val MSE {mse:.5f}  常数基线 {base:.5f}  R2 {1 - mse/base:.3f}")
    print(f"[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
