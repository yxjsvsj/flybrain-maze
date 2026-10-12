"""P2d-12a：真脑主导的局部避障最小验证台（离线，无 Dijkstra 转向，无 Bridge/电机/硬件）。

只检验一件事：真实 MaleCNS 下行神经元活动能否主导虚拟小车在**必须左转/右转/连续左右转**
的走廊里完成局部绕障。所有控制器 **memory_gain=0**（关闭全局规划转向），前方停车保留。

控制器（同一场景下起点/朝向/速度上限/ToF/预算完全一致）：
  DN  冻结 Decoder 的 mode="dn"（下行神经元 turn_L/turn_R 直接映射转向，含逃逸/卡死）
  N   现有 P2 候选 DN readout（mode="readout", brain_gain=1, memory_gain=0）
  Z   神经转向置零（同一 readout 前进速度，brain_gain=0）——消融
  T   直接 ToF 规则 clip(K*bias)（仅工程对照，不是老师/成功标准）

几何检查：用**虚拟圆盘外廓**（半径 = CarConfig.radius）确认"直行必撞"且"圆盘存在可行路径"。
虚拟碰撞/间隙**不代表实体外廓或实体防撞安全**。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import deque

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.decode import Decoder
from loop.dynamic_tof_sim import load_rig, run_dynamic
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup
from world.car import DiffDriveCar
from world.maze import Maze

CONTROLLERS = ("DN", "N", "Z", "T")

# 车体虚拟圆盘半径 = CarConfig.radius(0.22)；仅用于几何检查/间隙，不代表实体外廓。
CAR_R = 0.22

# 场景：起点 S 朝 +x。宽度>=2 走廊，圆盘可通过。wall 布局不因控制器而变。
# maze 坐标：行号 y 向下增大。必须"向上(−y)"= 一种 90° 转，"向下(+y)"= 反向 90° 转。
SCENARIOS = {
    "must_left": [                       # 直行撞 col6 墙 -> 必须向上绕
        "#############",
        "#...........#",
        "#.........G.#",
        "#.....#.....#",
        "#S....#.....#",
        "#############",
    ],
    "must_right": [                      # 直行撞 col6 墙 -> 必须向下绕
        "#############",
        "#S....#.....#",
        "#.....#.....#",
        "#...........#",
        "#.........G.#",
        "#############",
    ],
    "s_shape": [                         # 连续：右->上->左->上->右（交替 90° 转）
        "############",
        "#........G.#",
        "#..........#",
        "#..#########",  # 左端留口(col1,2)
        "#..........#",
        "#..........#",
        "#########..#",   # 右端留口(col9,10)
        "#..........#",
        "#S.........#",
        "############",
    ],
}


# ---------------------------------------------------------------- geometry ----
def _disk_free(maze, x, y, r, n=16):
    if maze.is_wall(x, y):
        return False
    for a in np.linspace(0.0, 2 * math.pi, n, endpoint=False):
        if maze.is_wall(float(x + r * math.cos(a)), float(y + r * math.sin(a))):
            return False
    return True


def straight_pass(maze, sx, sy, th, r, max_len=80.0, step=0.05):
    """直行：返回 (是否到达终点, 撞墙前走过的距离)。"""
    L = 0.0
    while L <= max_len:
        x, y = sx + L * math.cos(th), sy + L * math.sin(th)
        if not _disk_free(maze, x, y, r):
            return False, L
        if maze.dist_to_goal(x, y) == 0:
            return True, L
        L += step
    return False, L


def path_exists(maze, sx, sy, r, res=0.1):
    """在"圆盘可放置"的格点上做 4-邻接 BFS，判断起点到终点格是否连通。"""
    i0, j0 = int(round(sx / res)), int(round(sy / res))
    if not _disk_free(maze, i0 * res, j0 * res, r):
        return False
    seen = {(i0, j0)}
    q = deque([(i0, j0)])
    while q:
        i, j = q.popleft()
        if maze.dist_to_goal(i * res, j * res) == 0:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = (i + di, j + dj)
            if k in seen:
                continue
            if _disk_free(maze, k[0] * res, k[1] * res, r):
                seen.add(k)
                q.append(k)
    return False


def wall_clearance(maze, x, y, r, n=24):
    """圆盘边到最近墙的间隙（格）。用 24 方向 raycast 取最小，再减半径。"""
    best = math.inf
    for a in np.linspace(0.0, 2 * math.pi, n, endpoint=False):
        best = min(best, maze.raycast(x, y, float(a), 8.0))
    return best - r


def geometry_check(rows, r=CAR_R):
    maze = Maze(rows)
    sx, sy, sth = maze.start
    sp, sd = straight_pass(maze, sx, sy, sth, r)
    return {
        "maze": maze,
        "straight_pass": bool(sp),
        "straight_len": float(sd),
        "path_exists": bool(path_exists(maze, sx, sy, r)),
        "goal_dy": float(maze.goal[1] - sy) if maze.goal else float("nan"),
    }


# -------------------------------------------------------------- controllers ---
def build_decoder(cfg, brain, groups, dn, controller, readout_path):
    if controller == "DN":
        cfg.decoder.mode = "dn"
        return Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn)
    cfg.decoder.mode = "readout"
    cfg.decoder.readout_path = readout_path
    return Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn)


def gain_fn_for(controller):
    if controller == "DN":
        return lambda k, dec: (1.0, 0.0)
    if controller == "N":
        return lambda k, dec: (1.0, 0.0)
    # Z(置零) / T(规则) 都关掉读出的转向分量
    return lambda k, dec: (0.0, 0.0)


def omega_fn_for(controller, k_rule=3.0):
    if controller != "T":
        return None
    def fn(k, dec, info, v, omega):
        return float(dec.car_cfg.max_omega * np.clip(k_rule * info.get("bias", 0.0), -1.0, 1.0))
    return fn


def run_one(cfg, rows, brain, groups, dn, controller, readout_path, *, seconds=40.0,
            seed=64, rig=None, mpc=0.40, trace_path=None):
    maze = Maze(rows)
    sx, sy, sth = maze.start
    cfg.decoder.memory_gain = 0.0
    cfg.decoder.brain_gain = 1.0
    dec = build_decoder(cfg, brain, groups, dn, controller, readout_path)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    brain.reset(seed)
    traj, s = run_dynamic(cfg, maze, car, brain, enc, dec, memory_on=False,
                          seconds=seconds, tof_hz=10.0, rig=rig or load_rig(),
                          seed=seed, gain_fn=gain_fn_for(controller),
                          omega_fn=omega_fn_for(controller))
    clr = min(wall_clearance(maze, p["x"], p["y"], CAR_R) for p in traj) if traj else float("nan")
    # 卡死：最后 8s 内 best_goal_dist 无改善
    best, last_improve = float("inf"), 0.0
    for p in traj:
        d = p["goal_dist"]
        if d >= 0 and d < best:
            best, last_improve = d, p["t"]
    T = traj[-1]["t"] if traj else 0.0
    stuck = (not s["reached_goal"]) and (T - last_improve) > 8.0
    out = {
        "reached_goal": bool(s["reached_goal"]),
        "time_to_goal_s": float(s["time_to_goal_s"]),
        "collision_events": int(s["collision_events"]),
        "collision_steps": int(s["collision_steps"]),
        "min_clearance_cells": float(clr),
        "best_goal_dist": int(s["best_goal_dist"]),
        "stuck": bool(stuck),
        "timeout": bool(s["timeout"]),
        "path_len_cells": float(s["path_len_cells"]),
        "front_safety_events": int(s["front_safety_events"]),
    }
    if trace_path and traj:
        with open(trace_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "x", "y", "theta", "v", "omega", "brain_turn", "turn_cmd", "goal_dist"])
            for p in traj:
                w.writerow([f"{p['t']:.4f}", f"{p['x']:.5f}", f"{p['y']:.5f}",
                            f"{p['theta']:.5f}", f"{p['v']:.5f}", f"{p['omega']:.5f}",
                            f"{p['brain_turn']:.5f}", f"{p['turn_cmd']:.5f}", p["goal_dist"]])
    return out


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="P2 候选 readout .npz（N/Z/T 用）")
    ap.add_argument("--brain", default="real")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--meters-per-cell", type=float, default=0.40)
    ap.add_argument("--out-dir", default=None, help="轨迹 CSV 输出目录（不提交）")
    args = ap.parse_args(argv)

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=args.seed,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn = np.asarray(brain.cells(["descending_neuron"]))

    # ---- 1) 几何检查：直行必撞 + 圆盘存在可行路径 ----
    print("[geometry] virtual disk radius = CarConfig.radius = "
          f"{CAR_R:g} cells（不代表实体外廓）")
    print(f"{'scenario':11s} {'straight_pass':>13s} {'path_exists':>11s} "
          f"{'goal_dy(y-down+)':>16s} {'start(theta=0)':>14s}")
    geos = {}
    for name, rows in SCENARIOS.items():
        g = geometry_check(rows)
        geos[name] = g
        print(f"{name:11s} {str(g['straight_pass']):>13s} {str(g['path_exists']):>11s} "
              f"{g['goal_dy']:>16.1f} {str(Maze(rows).start):>14s}")

    # ---- 2) 四控制对照 ----
    print(f"\n[local-avoidance] memory_gain=0 (no Dijkstra). seconds={args.seconds} seed={args.seed}")
    print(f"{'scenario':11s} {'ctrl':4s} {'reach':>5s} {'t_goal':>7s} {'cev':>4s} {'cstep':>6s} "
          f"{'minGap':>7s} {'bestD':>6s} {'stuck':>5s} {'timeout':>7s} {'path':>6s}")
    table = []
    for name, rows in SCENARIOS.items():
        maze = Maze(rows)
        for ctrl in CONTROLLERS:
            tp = None
            if args.out_dir:
                os.makedirs(args.out_dir, exist_ok=True)
                tp = os.path.join(args.out_dir, f"la_{name}_{ctrl}.csv")
            r = run_one(cfg, rows, brain, groups, dn, ctrl, args.new,
                        seconds=args.seconds, seed=args.seed, mpc=args.meters_per_cell,
                        trace_path=tp)
            table.append((name, ctrl, r))
            print(f"{name:11s} {ctrl:4s} {str(r['reached_goal'])[:1]:>5s} "
                  f"{r['time_to_goal_s']:7.2f} {r['collision_events']:4d} {r['collision_steps']:6d} "
                  f"{r['min_clearance_cells']:7.3f} {r['best_goal_dist']:6d} "
                  f"{str(r['stuck'])[:1]:>5s} {str(r['timeout'])[:1]:>7s} {r['path_len_cells']:6.2f}")
    if args.out_dir:
        print(f"\n[traces] -> {args.out_dir} (t,x,y,theta,v,omega,brain_turn; uncommitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
