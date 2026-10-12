"""P2d-8：用**录制的真实 ToF CSV** 驱动物理编码器 -> RealFlyBrain -> Decoder，做离线仿真。

不启动 Bridge、不发任何电机指令、不碰硬件。喂入录制帧（场景静止，逐帧播出），
用 Decoder 输出的 (v, omega) 积分一条**模拟轨迹**；可开/关 Memory（Dijkstra 全局规划）
以区分**神经局部转向**与**全局规划**的作用。

用法：
  python -m loop.replay_drive_sim --csv cal17/static_B_FR_500.csv --brain real --memory 0
  python -m loop.replay_drive_sim --csv ... --memory 1 --out-csv traj.csv
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


def run_replay(cfg, maze, car, brain, enc, dec, frames, *, memory_on):
    dt = brain.dt
    memory = None
    if memory_on:
        from nav.memory import OccupancyMemory
        memory = OccupancyMemory(maze.w, maze.h, cfg.encoder.max_range,
                                 unknown_cost=cfg.nav.unknown_cost,
                                 lookahead=cfg.nav.lookahead,
                                 arrive_dist=cfg.nav.arrive_dist,
                                 off_path_tol=cfg.nav.off_path_tol,
                                 free_conflict_threshold=cfg.nav.free_conflict_threshold)
        if maze.goal is not None:
            memory.set_goal(maze.goal[0], maze.goal[1])

    x, y, th = float(car.x), float(car.y), float(car.theta)
    x0, y0, th0 = x, y, th
    path = 0.0
    traj = []
    for fr in frames:
        dists = enc.sense(fr)
        is_new = enc.update(fr, dists)
        inject, info = enc.inject()
        if memory is not None:
            if is_new:
                vmask = enc.valid_mask(fr)
                memory.update(x, y, th + enc.angles[vmask], dists[vmask])
                memory.check_invariants()
            target = memory.next_target(x, y)
            if target is not None:
                tx, ty, dd = target
                bearing = math.atan2(ty - y, tx - x)
                err = (bearing - th + np.pi) % (2 * np.pi) - np.pi
                LL = max(dd, 0.35)
                v_used = max(abs(dec.v), 0.35 * cfg.car.max_speed)
                turn_pp = 2.0 * v_used * math.sin(err) / (LL * cfg.car.max_omega)
                info["pursuit_turn"] = float(np.clip(turn_pp, -1.0, 1.0))
                info["target_dist"] = float(dd)
        fired = brain.step(inject=inject)
        v, omega = dec.update(fired, info=info, stalled=False)
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += omega * dt
        path += abs(v) * dt
        traj.append({"t": fr.t, "x": x, "y": y, "theta": th, "v": v, "omega": omega,
                     "brain_turn": dec.last_brain_turn, "turn_cmd": dec.last_turn_cmd,
                     "pursuit_turn": info.get("pursuit_turn", 0.0)})
    return traj, {"net_heading_deg": math.degrees(th - th0), "path_len_m": path,
                  "net_disp_m": math.hypot(x - x0, y - y0),
                  "n_frames": len(frames), "replans": (memory.replans if memory else None)}


def _dump(path, traj):
    cols = ["t", "x", "y", "theta", "v", "omega", "brain_turn", "turn_cmd", "pursuit_turn"]
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
    ap = argparse.ArgumentParser(description="P2d-8 real-ToF replay brain-drive sim")
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
    ap.add_argument("--max-frames", type=int, default=0)
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
    if args.max_frames:
        frames = frames[:args.max_frames]

    traj, summ = run_replay(cfg, maze, car, brain, enc, dec, frames, memory_on=args.memory > 0)
    bt = [r["brain_turn"] for r in traj]
    om = [r["omega"] for r in traj]
    print(f"[replay] csv={args.csv} brain={args.brain} gain={args.brain_gain} "
          f"memory={args.memory} frames={summ['n_frames']}")
    print(f"  brain_turn mean={np.mean(bt):+.4f} std={np.std(bt):.4f}  "
          f"omega mean={np.mean(om):+.4f} std={np.std(om):.4f}")
    print(f"  net_heading={summ['net_heading_deg']:+.2f}deg  path={summ['path_len_m']:.3f}cells  "
          f"net_disp={summ['net_disp_m']:.3f}cells  replans={summ['replans']}")
    if args.out_csv:
        _dump(args.out_csv, traj)
        print(f"  traj -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
