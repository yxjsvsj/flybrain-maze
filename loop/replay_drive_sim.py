"""P2d-9 任务一：**时间正确的**真实 ToF 回放（离线，无 Bridge/电机/硬件）。

修正 P2d-8 的时间语义错误：旧版每来一帧 ToF 只推进一步 brain.step(dt=0.02)，把 ~10Hz 的
~11s 录制压成 ~2s 神经时间。本版按 CSV 时间戳以 brain.dt(50Hz) 推进：

  - 50Hz brain 步；ToF 新帧按时间到达、按 seq 仅 update 一次；帧间 zero-order hold；
  - Encoder growth 用真实帧间隔、Decoder Trace 按 50Hz；
  - 明确区分 tof_frames / brain_steps / recorded_duration_s / sim_duration_s；
  - 单位：长度一律 cells（路径 = path_len_cells），不再误名 "m"。

此模式**固定原始采集位姿**（静态录制），不代表动态避障轨迹。

用法：
  python -m loop.replay_drive_sim --csv A.csv --brain real --memory 0
  python -m loop.replay_drive_sim --csv A.csv --per-frame   # 复现旧 P2d-8 行为（历史）
"""
from __future__ import annotations

import argparse
import csv
import math
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import TofFrame
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup


def load_frames(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"{path}: empty CSV")
    frames = []
    for i, r in enumerate(rows):
        status = {n: (r.get(f"{n}_status") or "NO_TARGET") for n in ORDER}
        ranges = {n: (float(r[f"{n}_mm"]) if r.get(f"{n}_mm") not in (None, "") else None)
                  for n in ORDER}
        frames.append(TofFrame(session_id="replay", seq=i, t=float(r.get("t", i * 0.02)),
                               healthy=True, ranges=ranges, status=status,
                               ages_ms={n: 10.0 for n in ORDER},
                               init_count=0, read_errors=0, reinit_count=0))
    return frames


def _build_memory(cfg, maze, memory_on):
    if not memory_on:
        return None
    from nav.memory import OccupancyMemory
    memory = OccupancyMemory(maze.w, maze.h, cfg.encoder.max_range,
                             unknown_cost=cfg.nav.unknown_cost, lookahead=cfg.nav.lookahead,
                             arrive_dist=cfg.nav.arrive_dist, off_path_tol=cfg.nav.off_path_tol,
                             free_conflict_threshold=cfg.nav.free_conflict_threshold)
    if maze.goal is not None:
        memory.set_goal(maze.goal[0], maze.goal[1])
    return memory


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


def run_replay(cfg, maze, car, brain, enc, dec, frames, *, memory_on, per_frame=False):
    """时间正确回放。per_frame=True 复现旧 P2d-8（每帧一步，压缩时间）。"""
    dt = brain.dt
    memory = _build_memory(cfg, maze, memory_on)
    x, y, th = float(car.x), float(car.y), float(car.theta)
    x0, y0, th0 = x, y, th
    path = 0.0
    traj = []
    fs = sorted(frames, key=lambda f: f.t)
    t0 = fs[0].t
    if per_frame:
        plan = [(fs[k].t, fs[k]) for k in range(len(fs))]           # 旧行为：每帧一步
    else:
        n = max(1, int(round((fs[-1].t - t0) / dt)) + 1)
        idx, cur, plan = 0, fs[0], []
        for k in range(n):
            t = t0 + k * dt
            while idx < len(fs) and fs[idx].t <= t + 1e-9:          # ZOH：取此前最近帧
                cur = fs[idx]
                idx += 1
            plan.append((t, cur))
    prev_seq = None
    for t, fr in plan:
        dists = enc.sense(fr)
        is_new = enc.update(fr, dists)                              # 同 seq 不重复 update
        inject, info = enc.inject()
        if memory is not None:
            if is_new:
                vmask = enc.valid_mask(fr)
                memory.update(x, y, th + enc.angles[vmask], dists[vmask])
                memory.check_invariants()
            _pursuit(memory, cfg, dec, x, y, th, info)
        fired = brain.step(inject=inject)
        v, omega = dec.update(fired, info=info, stalled=False)
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += omega * dt
        path += abs(v) * dt
        traj.append({"t": t, "x": x, "y": y, "theta": th, "v": v, "omega": omega,
                     "brain_turn": dec.last_brain_turn, "turn_cmd": dec.last_turn_cmd,
                     "pursuit_turn": info.get("pursuit_turn", 0.0),
                     "tof_seq": fr.seq, "new_frame": bool(is_new)})
        prev_seq = fr.seq
    summ = {"net_heading_deg": math.degrees(th - th0),
            "path_len_cells": path, "net_disp_cells": math.hypot(x - x0, y - y0),
            "tof_frames": len(frames), "brain_steps": len(traj),
            "recorded_duration_s": (fs[-1].t - fs[0].t),
            "sim_duration_s": len(traj) * dt,
            "replans": (memory.replans if memory else None)}
    return traj, summ


def _dump(path, traj):
    cols = ["t", "x", "y", "theta", "v", "omega", "brain_turn", "turn_cmd",
            "pursuit_turn", "tof_seq", "new_frame"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in traj:
            w.writerow(r)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                           # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="P2d-9 time-correct real-ToF replay")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--brain", choices=["fake", "real"], default="real")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["dn", "scripted", "readout"], default="readout")
    ap.add_argument("--readout", default="readout_real.npz")
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--brain-gain", type=float, default=0.15)
    ap.add_argument("--memory", type=float, default=0.0)
    ap.add_argument("--meters-per-cell", type=float, default=0.40)
    ap.add_argument("--map", default="gen")
    ap.add_argument("--per-frame", action="store_true", help="旧 P2d-8 行为（每帧一步，压缩时间）")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args(argv)

    cfg, maze, car, brain, groups, _e0, _ = make_setup(
        brain_kind=args.brain, map_name=args.map, dt=None, device=args.device,
        seed=args.seed, preset=None, sensory_input=False, make_decoder=False, verbose=False)
    cfg.decoder.mode = args.mode
    cfg.decoder.memory_gain = args.memory
    cfg.decoder.brain_gain = args.brain_gain
    cfg.decoder.readout_path = args.readout
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=args.meters_per_cell)
    frames = load_frames(args.csv)
    traj, s = run_replay(cfg, maze, car, brain, enc, dec, frames,
                         memory_on=args.memory > 0, per_frame=args.per_frame)
    bt = [r["brain_turn"] for r in traj]
    om = [r["omega"] for r in traj]
    print(f"[replay] csv={args.csv} brain={args.brain} gain={args.brain_gain} "
          f"memory={args.memory} per_frame={args.per_frame}")
    print(f"  tof_frames={s['tof_frames']} brain_steps={s['brain_steps']} "
          f"recorded={s['recorded_duration_s']:.2f}s sim={s['sim_duration_s']:.2f}s")
    print(f"  brain_turn mean={np.mean(bt):+.4f} std={np.std(bt):.4f}  "
          f"omega mean={np.mean(om):+.4f} std={np.std(om):.4f}")
    print(f"  net_heading={s['net_heading_deg']:+.2f}deg  path={s['path_len_cells']:.3f} cells  "
          f"net_disp={s['net_disp_cells']:.3f} cells  replans={s['replans']}")
    if args.out_csv:
        _dump(args.out_csv, traj)
        print(f"  traj -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
