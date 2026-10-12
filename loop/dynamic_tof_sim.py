"""P2d-9 任务二：动态五路 ToF 虚拟环境（离线，无 Bridge/电机/硬件）。

回路：
    虚拟车位姿 -> 按 tof_geometry 的五路安装位置/角度 raycast -> ~10Hz ToF 状态/距离
      -> PhysicalToFEncoder(50Hz ZOH) -> RealFlyBrain + 当前 readout -> Decoder(+可选 Memory/Dijkstra)
      -> DiffDriveCar 更新位姿（复用 world.car 的虚拟碰撞）-> 下一次测距

量程说明：**模拟有效量程**由 `max_valid_mm`（默认 1200mm≈VL53L0X）决定；超出即 NO_TARGET，
**不等同于**神经 Encoder 的 `max_range`(6 cells=2.4m，是注入用的占位)。噪声/丢失可配。
虚拟碰撞来自 world.car 的圆盘模型，**不代表实体防撞安全**。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from hardware.tof5 import ORDER
from hardware.tof5_client import TofFrame

_GEOM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "hardware", "tof_geometry.json")

_DEFAULT_RIG = {
    "L": (0.10, 0.06, 70.0), "FL": (0.14, 0.03, 30.0), "F": (0.15, 0.00, 0.0),
    "FR": (0.13, -0.04, -30.0), "R": (0.09, -0.07, -70.0),
}


def load_rig(path: str | None = None):
    """返回 {name: (x_m, y_m, yaw_deg)}。缺文件则用内置默认。"""
    p = path or _GEOM_PATH
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        return {n: (float(s["x_m"]), float(s["y_m"]), float(s["yaw_deg"]))
                for n, s in data["sensors"].items()}
    except (OSError, KeyError, ValueError):
        return dict(_DEFAULT_RIG)


def frame_from_pose(maze, x, y, th, t, seq, *, rig, meters_per_cell,
                    max_valid_mm=1200.0, too_near_mm=40.0, noise_mm=0.0, dropout_p=0.0, rng=None):
    """从车体位姿 raycast 出五路 ToF 帧。status ∈ VALID/NO_TARGET/TOO_NEAR。"""
    rng = rng or np.random.default_rng(0)
    max_cells = max_valid_mm / 1000.0 / meters_per_cell
    ranges, status = {}, {}
    for n in ORDER:
        sx, sy, yaw = rig.get(n, _DEFAULT_RIG[n])
        px = x + (sx * math.cos(th) - sy * math.sin(th)) / meters_per_cell
        py = y + (sx * math.sin(th) + sy * math.cos(th)) / meters_per_cell
        d_cells = maze.raycast(px, py, th + math.radians(yaw), max_cells)
        mm = d_cells * meters_per_cell * 1000.0
        if dropout_p > 0 and rng.random() < dropout_p:
            status[n], ranges[n] = "NO_TARGET", None
        elif d_cells >= max_cells - 1e-6:
            status[n], ranges[n] = "NO_TARGET", None
        elif mm <= too_near_mm:
            status[n], ranges[n] = "TOO_NEAR", mm
        else:
            if noise_mm > 0:
                mm = max(too_near_mm, mm + float(rng.normal(0.0, noise_mm)))
            status[n], ranges[n] = "VALID", mm
    return TofFrame(session_id="sim", seq=seq, t=t, healthy=True, ranges=ranges, status=status,
                    ages_ms={n: 1000.0 / max(1e-6, 10.0) for n in ORDER},
                    init_count=0, read_errors=0, reinit_count=0)


def build_memory(cfg, maze, memory_on, geometry=None):
    """memory_on 且有 geometry -> P2 ToFMemory（逐原点射线，TOO_NEAR≠精确墙命中）；
    否则退化为车体中心的 OccupancyMemory。均不修改冻结 nav.memory。"""
    if not memory_on:
        return None
    if geometry:
        from loop.memory_tof import ToFMemory
        cls = ToFMemory
    else:
        from nav.memory import OccupancyMemory
        cls = OccupancyMemory
    m = cls(maze.w, maze.h, cfg.encoder.max_range,
            unknown_cost=cfg.nav.unknown_cost, lookahead=cfg.nav.lookahead,
            arrive_dist=cfg.nav.arrive_dist, off_path_tol=cfg.nav.off_path_tol,
            free_conflict_threshold=cfg.nav.free_conflict_threshold)
    if maze.goal is not None:
        m.set_goal(maze.goal[0], maze.goal[1])
    return m


def _pursuit(memory, cfg, dec, x, y, th, info):
    target = memory.next_target(x, y)
    if target is None:
        return
    tx, ty, dd = target
    bearing = math.atan2(ty - y, tx - x)
    err = (bearing - th + np.pi) % (2 * np.pi) - np.pi
    LL = max(dd, 0.35)
    v_used = max(abs(dec.v), 0.35 * cfg.car.max_speed)
    turn_pp = 2.0 * v_used * math.sin(err) / (LL * cfg.car.max_omega)
    info["pursuit_turn"] = float(np.clip(turn_pp, -1.0, 1.0))
    info["target_dist"] = float(dd)


def run_dynamic(cfg, maze, car, brain, enc, dec, *, memory_on, seconds, tof_hz=10.0,
                rig=None, max_valid_mm=1200.0, too_near_mm=40.0, noise_mm=0.0,
                dropout_p=0.0, seed=0, stop_on_goal=True, geometry=None):
    """动态闭环仿真。返回 (traj, summary)。不接触硬件。"""
    dt = brain.dt
    mpc = enc.m_per_cell
    rig = rig or load_rig()
    rng = np.random.default_rng(seed)
    memory = build_memory(cfg, maze, memory_on, geometry)
    n = max(1, int(round(seconds / dt)))
    period = 1.0 / tof_hz
    frame, seq, last_t = None, 0, -1e9
    x0, y0 = car.x, car.y
    traj = []
    reached, t_goal, best = False, -1.0, float("inf")
    for k in range(n):
        t = k * dt
        if frame is None or (t - last_t) >= period - 1e-9:
            seq += 1
            frame = frame_from_pose(maze, car.x, car.y, car.theta, t, seq, rig=rig,
                                    meters_per_cell=mpc, max_valid_mm=max_valid_mm,
                                    too_near_mm=too_near_mm, noise_mm=noise_mm,
                                    dropout_p=dropout_p, rng=rng)
            last_t = t
        dists = enc.sense(frame)
        is_new = enc.update(frame, dists)
        inject, info = enc.inject()
        if memory is not None:
            if is_new:
                if geometry:
                    from loop.tof_geometry import frame_to_cell_rays
                    rays = frame_to_cell_rays(frame, car.x, car.y, car.theta, geometry, mpc)
                    memory.update_rays(car.x, car.y, rays, tof_seq=frame.seq)
                else:
                    vmask = enc.valid_mask(frame)
                    memory.update(car.x, car.y, car.theta + enc.angles[vmask], dists[vmask])
                memory.check_invariants()
            _pursuit(memory, cfg, dec, car.x, car.y, car.theta, info)
        fired = brain.step(inject=inject)
        v, omega = dec.update(fired, info=info, stalled=False)
        car.set_command(v, omega)
        car.step(maze, dt)
        d = maze.dist_to_goal(car.x, car.y)
        if d >= 0:
            best = min(best, d)
        if d == 0 and not reached:
            reached, t_goal = True, (k + 1) * dt
        traj.append({"t": t, "x": car.x, "y": car.y, "theta": car.theta, "v": v, "omega": omega,
                     "brain_turn": dec.last_brain_turn, "turn_cmd": dec.last_turn_cmd,
                     "pursuit_turn": info.get("pursuit_turn", 0.0),
                     "collisions": car.collisions, "goal_dist": d})
        if stop_on_goal and reached:
            break
    optimal = maze.optimal_path()
    summ = {"reached_goal": reached, "time_to_goal_s": t_goal,
            "timeout": (not reached),
            "best_goal_dist": (best if best != float("inf") else -1),
            "path_len_cells": car.distance, "net_disp_cells": math.hypot(car.x - x0, car.y - y0),
            "collision_steps": car.collisions, "collision_events": car.collision_events,
            "front_safety_events": dec.front_blocks,
            "replans": (memory.replans if memory else 0),
            "no_path_events": (memory.invalidations.get("NO_PATH", 0) if memory else 0),
            "map_revision": (getattr(memory, "map_revision", 0) if memory else 0),
            "optimal_path_cells": optimal,
            "min_time_lower_bound_s": (optimal / cfg.car.max_speed if optimal > 0 else -1.0),
            "path_efficiency": (optimal / car.distance
                                if (reached and car.distance > 0 and optimal > 0) else -1.0),
            "steps": len(traj)}
    return traj, summ
