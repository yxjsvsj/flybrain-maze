"""训练下行神经元读出头（reservoir computing）。

连接组固定不动。把全部下行神经元（真脑 1314 个）的脉冲痕迹当特征，用线性回归
去拟合一个"老师"控制器的转向指令。老师是 --mode scripted（手写贴墙），也就是目前
最强的无记忆控制器（6 局能到 2 局）。

关键对照：同时训一个**只有 2 个特征（左右墙距）**的线性控制器。
如果 1314 个神经元打不过 2 个数字，那连接组在这个任务上就没加价值——这个结论
比"读出头能不能开"重要得多。

用法：
    python -m loop.train_readout --brain real --seeds 1 3 7 11 --out readout_real.npz
    python -m loop.train_readout --brain fake --out readout_fake.npz
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from brain.neurons import resolve
from brain.wrap import make_brain
from config import Config, apply_preset
from loop.encode import Encoder
from loop.run import DEFAULT_DT, make_maze
from world.car import DiffDriveCar


def teacher_turn(cfg, info) -> float:
    """老师（手写贴墙）在给定传感器状态下的转向指令，归一化到 [-1, 1]。"""
    key = "dmin_R" if cfg.decoder.follow_side == "R" else "dmin_L"
    dist = float(info.get(key, 99.0))
    return float(np.clip(cfg.decoder.follow_gain * (cfg.decoder.follow_distance - dist),
                         -1.0, 1.0))


def rollout(brain, groups, dn_idx, cfg, seeds, steps, turn_fn, cells, scale, loop):
    """跑若干局，每步记录 (下行神经元痕迹, 老师指令, 传感器特征)。"""
    from flybrain import Trace

    X, Y, S = [], [], []
    for seed in seeds:
        maze = make_maze("gen", seed, cells, scale, loop)
        sx, sy, sth = maze.start
        car = DiffDriveCar(sx, sy, sth, cfg.car)
        enc = Encoder(groups, cfg.encoder, brain.dt, car_max_speed=cfg.car.max_speed)
        trace = Trace(brain, idx=dn_idx, tau=cfg.decoder.readout_tau)
        brain.reset(cfg.brain.seed)
        for _ in range(steps):
            dists = enc.sense(maze, car)
            inject, info = enc.inject(dists)
            fired = brain.step(inject=inject)
            feats = trace.observe(fired)
            X.append(feats)
            Y.append(teacher_turn(cfg, info))
            S.append((info["dmin_L"], info["dmin_R"]))
            car.set_command(cfg.car.max_speed * cfg.decoder.follow_speed_frac,
                            cfg.car.max_omega * float(np.clip(turn_fn(info, feats), -1, 1)))
            car.step(maze, brain.dt)
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(S, np.float32))


def main(argv=None):
    ap = argparse.ArgumentParser(description="训练下行神经元读出头")
    ap.add_argument("--brain", choices=["fake", "real"], default="real")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 3, 7, 11])
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=[23, 42, 5, 17],
                    help="只用来评估、绝不参与训练，防止过拟合到训练迷宫")
    ap.add_argument("--steps", type=int, default=3000, help="每局步数")
    ap.add_argument("--dagger", type=int, default=2,
                    help="DAgger 轮数：用当前读出头跑闭环，把新状态下的老师指令加进训练集")
    ap.add_argument("--cells", default="9x6")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--loop", type=float, default=0.08)
    ap.add_argument("--out", default="readout.npz")
    ap.add_argument("--components", nargs="+", type=int, default=[10, 40, 120])
    ap.add_argument("--stride", type=int, default=4,
                    help="拟合前按时间降采样。相邻步的下行神经元痕迹几乎相同，"
                         "stride=4 能砍掉大部分冗余，SVD 快好几倍")
    args = ap.parse_args(argv)

    from flybrain import Readout

    cells = tuple(int(v) for v in args.cells.lower().split("x"))
    cfg = Config()
    cfg.brain.kind = args.brain
    cfg.brain.device = args.device
    cfg.brain.dt = DEFAULT_DT[args.brain]
    apply_preset(cfg, args.brain)

    brain = make_brain(cfg.brain, verbose=True)
    groups = resolve(brain, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    print(f"[data] 下行神经元 {len(dn_idx)} 个，特征维度 {len(dn_idx)}")

    t0 = time.perf_counter()
    print(f"\n[round 0] 用老师驱动采集 {len(args.seeds)} 局 x {args.steps} 步 ...")
    X, Y, S = rollout(brain, groups, dn_idx, cfg, args.seeds, args.steps,
                      lambda info, f: teacher_turn(cfg, info), cells, args.scale, args.loop)
    print(f"          样本 {X.shape[0]}，用时 {time.perf_counter() - t0:.1f}s")

    rd_dn = rd_sensor = None      # 循环至少跑一次，下面一定被赋值
    for rnd in range(args.dagger + 1):
        t1 = time.perf_counter()
        Xs, Ys, Ss = X[::args.stride], Y[::args.stride], S[::args.stride]
        print(f"\n[train {rnd}] 拟合（{Xs.shape[0]} 样本 x {Xs.shape[1]} 特征，"
              f"stride={args.stride}）...")
        rd_dn = Readout.fit(Xs, Ys, kind="ridge", components=tuple(args.components),
                            lambdas=(1e-3, 1e-2, 1e-1, 1.0), verbose=True)
        rd_sensor = Readout.fit(Ss, Ys, kind="ridge", components=(None,),
                                lambdas=(1e-3, 1e-2, 1e-1, 1.0), verbose=True)
        print(f"          DN 读出头  cv neg-MSE {rd_dn.cv_score:+.5f} "
              f"({rd_dn.components} PCs, lam {rd_dn.lam:g})")
        print(f"          传感器基线 cv neg-MSE {rd_sensor.cv_score:+.5f}  (2 个特征)")
        print(f"          用时 {time.perf_counter() - t1:.1f}s")
        # 每轮都存盘，中途被打断也不用重跑
        rd_dn.save(args.out)
        rd_sensor.save(args.out.replace(".npz", "_sensor.npz"))
        print(f"          [save] {args.out}")

        if rnd == args.dagger:
            break
        # DAgger：用当前读出头跑闭环，在它真正走到的状态上补老师指令
        print(f"\n[round {rnd + 1}] DAgger 采集：用读出头驱动 ...")
        Xn, Yn, Sn = rollout(brain, groups, dn_idx, cfg, args.seeds, args.steps,
                             lambda info, f: float(rd_dn.predict(f)), cells, args.scale, args.loop)
        X = np.concatenate([X, Xn])
        Y = np.concatenate([Y, Yn])
        S = np.concatenate([S, Sn])
        print(f"          新增 {Xn.shape[0]} 样本，累计 {X.shape[0]}")

    # 留出迷宫上的离线评估。
    # 注意：传感器基线是**天花板**不是对手——老师本身就是 dmin 的线性函数，
    # 2 个特征能把它拟到 R2=1。所以真正要看的是 DN 读出头能逼近它多少。
    print(f"\n[eval] 在没参与训练的迷宫 {args.eval_seeds} 上采集 ...")
    Xe, Ye, Se = rollout(brain, groups, dn_idx, cfg, args.eval_seeds, args.steps,
                         lambda info, f: teacher_turn(cfg, info), cells, args.scale, args.loop)
    base = float(np.mean((Ye - Ye.mean()) ** 2))
    for name, rd, F in (("DN 读出头", rd_dn, Xe), ("传感器基线(天花板)", rd_sensor, Se)):
        mse = float(np.mean((rd.predict(F) - Ye) ** 2))
        print(f"          {name:18s} MSE {mse:.5f}   常数基线 {base:.5f}   "
              f"R2 {1 - mse / base:.3f}")

    rd_dn.save(args.out)
    sensor_path = args.out.replace(".npz", "_sensor.npz")
    rd_sensor.save(sensor_path)
    print(f"\n[save] {args.out}\n[save] {sensor_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
