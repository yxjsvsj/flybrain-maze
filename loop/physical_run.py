"""P2b：真实物理闭环运行器（架构 A）。

真 ToF + 真里程计 + 虚拟环境/规划，HIL。**不改冻结 `loop/run.py`**——因为冻结 run()
每步都 `memory.update(...)`，无法按 ToF 帧门控，所以这里复刻一版带门控的循环。

回路
----
  真 ToF(5路) -> PhysicalToFEncoder(注入) ─┐
  真里程计 -> 真实位姿 ────────────────────┼-> 冻结 Decoder/Memory/Dijkstra -> (v,ω)
                                            └-> 物理 Pikachu 电机

关键门控
--------
- brain=50Hz；ToF=10Hz。每步用**最新** ToF 帧（帧间零阶保持）。
- `enc.update()` 只在**新 tof_seq** 时刷 size/growth 缓存（growth 用真实 ToF 帧间隔）。
- OccupancyMemory 只在**新 tof_seq** 时 update 一次——同一帧绝不重复消费 5 次。

安全
----
- odom stale/reset、ToF stale/unhealthy/reset、pacer 落后超限 -> 抛错 -> FAILSAFE latch
  -> STOP，不自动恢复运动。
- collisions/contact_ratio 在 P2b **不是物理碰撞测量**（无虚拟碰撞强制）。
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time

import numpy as np

from hardware.odometry_client import OdomClient, OdomError
from hardware.pikachu_bridge import BridgeState, PikachuBridge, PikachuConfig
from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import Tof5Client, TofError
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.odom_shadow_run import OdomRealCar
from loop.run import make_setup, print_stats
from loop.shadow_run import RealTimePacer


def _run_physical(cfg, maze, car, brain, enc, dec, tof, steps, *, stop_on_goal,
                  trace_fn, trace_every):
    """复刻冻结 run() 的循环，但：sense 用真 ToF、Memory 按 tof_seq 门控。"""
    dt = brain.dt
    reached_goal = False
    time_to_goal = -1.0
    best_goal = 10 ** 9
    stalled = False
    steps_done = 0

    memory = None
    if cfg.decoder.memory_gain > 0:
        from nav.memory import OccupancyMemory
        memory = OccupancyMemory(maze.w, maze.h, cfg.encoder.max_range,
                                 unknown_cost=cfg.nav.unknown_cost,
                                 lookahead=cfg.nav.lookahead,
                                 arrive_dist=cfg.nav.arrive_dist,
                                 off_path_tol=cfg.nav.off_path_tol,
                                 free_conflict_threshold=cfg.nav.free_conflict_threshold)
        if maze.goal is not None:
            memory.set_goal(maze.goal[0], maze.goal[1])

    for step in range(steps):
        steps_done = step + 1
        frame, _age = tof.latest()                 # stale/unhealthy/reset -> 抛 TofError
        dists = enc.sense(frame)
        is_new = enc.update(frame, dists)          # 新 seq 才刷缓存
        inject, info = enc.inject()

        if memory is not None:
            if is_new:                             # **只在 ToF 帧变化时 update 一次**
                memory.update(car.x, car.y, car.theta + enc.angles, dists)
                memory.check_invariants()
            target = memory.next_target(car.x, car.y)
            if target is not None:
                tx, ty, dist = target
                bearing = math.atan2(ty - car.y, tx - car.x)
                err = (bearing - car.theta + np.pi) % (2 * np.pi) - np.pi
                L = max(dist, 0.35)
                v_used = max(abs(dec.v), 0.35 * cfg.car.max_speed)
                turn_pp = 2.0 * v_used * math.sin(err) / (L * cfg.car.max_omega)
                info["pursuit_turn"] = float(np.clip(turn_pp, -1.0, 1.0))
                info["bearing_err"] = float(err)
                info["target"] = (float(tx), float(ty))
                info["target_dist"] = float(dist)

        fired = brain.step(inject=inject)
        v, omega = dec.update(fired, info=info, stalled=stalled)
        car.set_command(v, omega)
        car.step(maze, dt)                          # OdomRealCar -> 真实位姿

        stalled = (dec.front_blocked
                   or (abs(dec.v) > 0.15 * cfg.car.max_speed
                       and car.last_move < 0.3 * abs(dec.v) * dt))

        d = maze.dist_to_goal(car.x, car.y)
        if d >= 0:
            best_goal = min(best_goal, d)
            if d == 0 and not reached_goal:
                reached_goal = True
                time_to_goal = steps_done * dt

        if trace_fn is not None and step % trace_every == 0:
            trace_fn({
                "step": step, "t": step * dt,
                "x": car.x, "y": car.y, "theta": car.theta,
                "v": dec.v, "omega": dec.omega,
                "sim_v": dec.v, "sim_omega": dec.omega,
                "dmin_F": info.get("dmin_F", float("nan")),
                "dmin_L": info.get("dmin_L", float("nan")),
                "dmin_R": info.get("dmin_R", float("nan")),
                "front_blocked": dec.front_blocked,
                "target_dist": info.get("target_dist", float("nan")),
                "bearing_err": info.get("bearing_err", float("nan")),
                "tof_seq": frame.seq, "tof_new": bool(is_new),
                "replans": memory.replans if memory else 0,
                "goal_dist": d,
            })

        if stop_on_goal and reached_goal:
            break

    sim_time = steps_done * dt
    optimal = maze.optimal_path()
    stats = {
        "steps": steps, "steps_done": steps_done, "sim_time": sim_time,
        "distance": car.distance,
        "mean_speed": car.distance / sim_time if sim_time > 0 else 0.0,
        "collisions": car.collisions, "collision_events": car.collision_events,
        "contact_ratio": car.collisions / max(1, steps_done),
        "coverage": car.coverage(maze),
        "escapes": dec.escapes, "stalls": dec.stalls,
        "front_safety_events": dec.front_blocks,
        "best_goal_dist": best_goal if best_goal < 10 ** 9 else -1,
        "reached_goal": reached_goal, "time_to_goal": time_to_goal,
        "optimal_path_cells": optimal,
        "path_efficiency": (optimal / car.distance
                            if (reached_goal and car.distance > 0 and optimal > 0) else -1.0),
        "map_explored": memory.explored_fraction(len(maze.free_cells())) if memory else -1.0,
        "replans": memory.replans if memory else 0,
        "no_path_events": memory.invalidations["NO_PATH"] if memory else 0,
        "brain_ms_per_step": 0.0,
    }
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2b 真实物理闭环")
    ap.add_argument("--brain", choices=["fake", "real"], default="real")
    ap.add_argument("--map", default="gen")
    ap.add_argument("--maze-seed", type=int, default=1000)
    ap.add_argument("--maze-cells", default="6x4")
    ap.add_argument("--maze-scale", type=int, default=2)
    ap.add_argument("--maze-loop", type=float, default=0.08)
    ap.add_argument("--sim-seconds", type=float, default=60.0)
    ap.add_argument("--stop-on-goal", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--mode", choices=["dn", "scripted", "readout"], default="readout")
    ap.add_argument("--readout", default="readout_real.npz")
    ap.add_argument("--memory", type=float, default=1.0)
    ap.add_argument("--brain-gain", type=float, default=1.0)
    ap.add_argument("--pikachu-url", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=0.08)
    ap.add_argument("--max-v", type=float, default=1.0)
    ap.add_argument("--max-w", type=float, default=1.0)
    ap.add_argument("--max-motor-mix", type=float, default=1.0)
    ap.add_argument("--min-cmd", type=float, default=0.65)
    ap.add_argument("--v-gain", type=float, default=1.0)
    ap.add_argument("--w-gain", type=float, default=1.0)
    ap.add_argument("--omega-sign", type=float, default=1.0)
    ap.add_argument("--slew-rate", type=float, default=0.0)
    ap.add_argument("--log", default="physical_log.csv")
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--max-pacer-lag", type=float, default=0.25)
    # odom
    ap.add_argument("--odom-port", type=int, default=8888)
    ap.add_argument("--odom-bind", default="0.0.0.0")
    ap.add_argument("--max-odom-age", type=float, default=0.25)
    ap.add_argument("--meters-per-cell", type=float, required=True)
    # tof
    ap.add_argument("--tof-port", type=int, default=8889)
    ap.add_argument("--max-tof-age", type=float, default=0.25)
    ap.add_argument("--odom-wait", type=float, default=5.0)
    ap.add_argument("--tof-wait", type=float, default=5.0)
    args = ap.parse_args(argv)

    if not args.dry_run and not args.pikachu_url:
        print("需要 --pikachu-url（或 --dry-run）")
        return 2
    if not args.dry_run and args.no_realtime:
        print("实体运动必须开节拍。--no-realtime 只允许配合 --dry-run。")
        return 2

    parts = args.maze_cells.lower().split("x")
    cells = (int(parts[0]), int(parts[1]))
    cfg, maze, car, brain, groups, _enc, _ = make_setup(
        brain_kind=args.brain, map_name=args.map, dt=None, device=args.device,
        seed=args.seed, preset=None, sensory_input=False,
        maze_seed=args.maze_seed, maze_cells=cells, maze_scale=args.maze_scale,
        maze_loop=args.maze_loop, make_decoder=False, verbose=True)
    cfg.decoder.mode = args.mode
    cfg.decoder.memory_gain = args.memory
    cfg.decoder.brain_gain = args.brain_gain
    if args.readout:
        cfg.decoder.readout_path = args.readout
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    steps = max(2, int(round(args.sim_seconds / brain.dt)))

    # 真实 ToF encoder
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius)
    print(f"[tof] angles={ {n: DEFAULT_ANGLES_DEG[n] for n in ORDER} }  "
          f"left={ [ORDER[i] for i in np.where(enc.left)[0]] } "
          f"right={ [ORDER[i] for i in np.where(enc.right)[0]] } "
          f"front={ [ORDER[i] for i in np.where(enc.front)[0]] }")

    # 客户端
    odom = OdomClient(args.odom_port, max_age=args.max_odom_age, bind=args.odom_bind)
    tof = Tof5Client(args.tof_port, max_age=args.max_tof_age, bind=args.odom_bind)
    odom.start(); tof.start()
    for name, cli, wait in (("odom", odom, args.odom_wait), ("tof", tof, args.tof_wait)):
        t0 = time.time()
        while cli.age() is None:
            if time.time() - t0 > wait:
                print(f"[{name}] 等不到数据（UDP :{args.odom_port if name=='odom' else args.tof_port}）；退出")
                odom.stop(); tof.stop()
                return 1
            time.sleep(0.05)
    print(f"[odom] :{args.odom_port} ok   [tof] :{args.tof_port} ok")

    real_car = OdomRealCar(car.x, car.y, car.theta, cfg.car, odom, args.meters_per_cell)

    pcfg = PikachuConfig(
        base_url=args.pikachu_url or "http://127.0.0.1:8000",
        rate_hz=args.rate_hz, timeout_s=args.timeout,
        sim_max_speed=cfg.car.max_speed, sim_max_omega=cfg.car.max_omega,
        v_gain=args.v_gain, w_gain=args.w_gain, omega_sign=args.omega_sign,
        max_v=args.max_v, max_w=args.max_w, max_motor_mix=args.max_motor_mix,
        min_cmd=args.min_cmd, slew_rate=args.slew_rate,
        dry_run=args.dry_run, log_path=args.log)
    bridge = PikachuBridge(pcfg)
    hardware_mode = not args.dry_run
    pacer = RealTimePacer(max_lag_s=args.max_pacer_lag, enabled=not args.no_realtime,
                          allow_resync=not hardware_mode)

    def trace_callback(rec: dict) -> None:
        lag = pacer.wait(rec["t"])
        bridge.note_lag(lag)
        if hardware_mode and lag > args.max_pacer_lag:
            bridge.enter_failsafe(
                f"pacer 落后 {lag*1000:.0f}ms > {args.max_pacer_lag*1000:.0f}ms")
        bridge.on_step(rec)

    def _on_signal(signum, _f):
        print(f"\n[physical] 信号 {signum}，停车退出")
        bridge.stop(reason=f"signal {signum}"); odom.stop(); tof.stop()
        raise KeyboardInterrupt

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _on_signal)
        except (ValueError, OSError):
            pass

    print(f"[physical] sim {args.sim_seconds:.0f}s = {steps} steps @ {brain.dt*1000:.0f}ms；"
          f"ToF 10Hz。collisions/contact_ratio 不是物理碰撞。")
    stats = None
    try:
        if not bridge.start():
            print("[physical] 桥接未 RUNNING（预检失败）——仿真照跑，实体不动")
        stats = _run_physical(cfg, maze, real_car, brain, enc, dec, tof, steps,
                              stop_on_goal=args.stop_on_goal,
                              trace_fn=trace_callback, trace_every=1)
    except (OdomError, TofError) as exc:
        print(f"\n[physical] *** {type(exc).__name__}: {exc}")
        print("[physical] *** -> FAILSAFE（停车 + latch，不自动恢复）***")
        bridge.enter_failsafe(f"{type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        print("[physical] 中断")
    finally:
        bridge.stop(reason="finally"); odom.stop(); tof.stop()

    if stats is not None:
        print_stats(stats)
    print("\n[physical] bridge:", bridge.stats())
    print("[physical] odom:", odom.stats())
    print("[physical] tof:", tof.stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
