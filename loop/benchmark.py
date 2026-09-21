"""导航 benchmark。

**两个随机维度必须分开：**
  maze seeds  = 环境泛化（不同迷宫）
  reps        = 同一 maze + 同一 brain seed 下，CUDA 数值非确定性导致的 rollout 差异
不要把 seeds*reps 当成那么多个独立迷宫样本。

**每个 rep 必须完整重建**：brain.reset(brain_seed) + 新建 Encoder / Decoder /
Trace / Memory / Car，不允许任何动态状态从前一个 rep 泄漏。
用 `--check-repro` 自检：CPU 是逐位确定的，同一个 maze seed 跑 N 次必须得到
**完全相同**的统计量；不一致就说明有状态泄漏。

**建议的最终实验分工：**
  CPU  deterministic benchmark  -> 主可复现结果，样本量来自更多独立 maze seeds
  CUDA repeated benchmark       -> 数值鲁棒性 / run-to-run variance

    python -m loop.benchmark --device cpu --reps 1
    python -m loop.benchmark --device cuda --reps 5 --csv runs.csv
    python -m loop.benchmark --check-repro --device cpu --reps 3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time

import numpy as np

from brain.neurons import resolve
from brain.wrap import make_brain
from config import Config, apply_preset
from loop.decode import Decoder
from loop.encode import Encoder
from loop.run import DEFAULT_DT, make_maze, run
from world.car import DiffDriveCar

# 调参用（可以随便用这些 seed 试参数）
TUNE_SEEDS = [1, 3, 7, 11, 17]
# 开发/诊断用。调参和诊断都只能在这批上做，不能碰 HOLDOUT。
DEV_SEEDS = [23, 42, 5, 31, 61, 71, 83, 97, 109, 127,
             131, 149, 163, 181, 197, 211, 227, 239, 251, 269]
# 最终测试用。**在参数冻结之前不要生成、不要看、不要跑。**
HOLDOUT_SEEDS: list[int] = []


def make_holdout_seeds(n: int = 40, base: int = 1000) -> list[int]:
    """生成一批全新种子，避开 TUNE/DEV。只在参数冻结之后调用一次。"""
    used = set(TUNE_SEEDS) | set(DEV_SEEDS)
    out: list[int] = []
    s = base
    while len(out) < n:
        if s not in used:
            out.append(s)
        s += 1
    return out


CONTROLLERS = {
    "real_nomem": dict(kind="real", mode="dn", memory=0.0, brain_gain=1.0),
    "real_mem": dict(kind="real", mode="dn", memory=1.0, brain_gain=1.0),
    "real_readout_mem": dict(kind="real", mode="readout", memory=1.0, brain_gain=1.0,
                             readout="readout_real.npz"),
    "real_mem_nobrain": dict(kind="real", mode="dn", memory=1.0, brain_gain=0.0),
    "fake_mem": dict(kind="fake", mode="dn", memory=1.0, brain_gain=1.0),
}

# CSV 每个 rollout 一行
ROLLOUT_FIELDS = ["controller", "maze_seed", "brain_seed", "device", "rep",
                  "reached_goal", "time_to_goal", "distance", "path_efficiency",
                  "contact_ratio", "collision_events", "stalls", "escapes",
                  "replans", "off_path_events", "max_offpath_dist", "max_target_dist",
                  "front_safety_events", "best_goal_dist",
                  "coverage", "map_explored", "steps_done", "sim_time",
                  "optimal_path_cells", "mean_speed", "brain_ms_per_step"]


# --------------------------------------------------------------------------- 元数据
def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return ""


def sha256_file(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def metadata(device: str, brain_seed: int, seeds, reps: int, dt: float,
             readout: str) -> dict:
    import flybrain
    meta = {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "device_requested": device,
        "gpu": _run(["nvidia-smi", "--query-gpu=name,driver_version",
                     "--format=csv,noheader"]),
        "brain_seed": brain_seed,
        "maze_seeds": list(seeds),
        "n_maze_seeds": len(seeds),
        "reps": reps,
        "dt": dt,
        "readout_file": readout,
        "readout_sha256": sha256_file(readout) if readout else "",
        "flybrain_version": getattr(flybrain, "__version__", "?"),
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cupy_version": "",
    }
    try:
        import cupy
        meta["cupy_version"] = cupy.__version__
        meta["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
    except Exception:                                     # noqa: BLE001
        pass
    return meta


# --------------------------------------------------------------------------- 装配
def build_static(kind, mode, memory, brain_gain, readout, device, brain_seed):
    """静态部分：Config / Brain / groups / dn_idx。可以跨 rep 复用。

    Brain 是静态的（权重来自连接组），动态状态靠 brain.reset() 清干净。
    """
    cfg = Config()
    cfg.brain.kind = kind
    cfg.brain.device = device
    cfg.brain.dt = DEFAULT_DT[kind]
    cfg.brain.seed = brain_seed
    apply_preset(cfg, kind)
    cfg.decoder.mode = mode
    cfg.decoder.memory_gain = memory
    cfg.decoder.brain_gain = brain_gain
    if readout:
        cfg.decoder.readout_path = readout
    brain = make_brain(cfg.brain, verbose=False)
    groups = resolve(brain, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    return cfg, brain, groups, dn_idx


def rollout(static, seed, cells, scale, loop_chance, sim_seconds, brain_seed):
    """一次完整 rollout。

    brain 重置到 brain_seed；Encoder / Decoder（内含 Trace + Readout）/
    Memory / Car 全部新建 —— 上一次 rollout 的任何动态状态都不会泄漏进来。
    """
    cfg, brain, groups, dn_idx = static
    brain.reset(brain_seed)

    maze = make_maze("gen", seed, cells, scale, loop_chance)
    sx, sy, sth = maze.start
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    enc = Encoder(groups, cfg.encoder, brain.dt, car_max_speed=cfg.car.max_speed,
                  car_radius=cfg.car.radius)
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    steps = max(2, int(round(sim_seconds / brain.dt)))
    # Memory 在 run() 内部按 cfg.decoder.memory_gain 新建
    return run(cfg, maze, car, brain, enc, dec, steps, verbose=False, stop_on_goal=True)


def check_repro(args, cells):
    """状态泄漏自检：CPU 逐位确定，同一个 maze seed 跑 N 次统计量必须完全相同。"""
    spec = CONTROLLERS[args.controllers[0]]
    seed = args.seeds[0]
    static = build_static(spec["kind"], spec["mode"], spec["memory"],
                          spec["brain_gain"], str(spec.get("readout") or ""),
                          "cpu", args.brain_seed)
    print(f"[check-repro] controller={args.controllers[0]}  device=cpu  "
          f"maze_seed={seed}  reps={args.reps}")
    print("  CPU 是逐位确定的，所以 N 次必须完全一致；不一致 = 有状态泄漏\n")
    sigs = []
    for rep in range(args.reps):
        st = rollout(static, seed, cells, args.scale, args.loop,
                     args.sim_seconds, args.brain_seed)
        sig = (st["reached_goal"], round(st["distance"], 9), st["steps_done"],
               round(st["contact_ratio"], 9), st["replans"],
               st["front_safety_events"], st["best_goal_dist"])
        sigs.append(sig)
        print(f"  rep {rep + 1}: reached={sig[0]} distance={sig[1]:.6f} "
              f"steps={sig[2]} contact={sig[3]:.6f} replans={sig[4]} "
              f"front={sig[5]} goal={sig[6]}")
    ok = len(set(sigs)) == 1
    print(f"\n  {'PASS' if ok else 'FAIL'} —— {len(set(sigs))} 个不同结果 / {len(sigs)} 次")
    if not ok:
        print("  说明 rollout 之间存在状态泄漏，先修这个再做统计。")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="导航 benchmark（seeds 与 reps 严格区分）")
    ap.add_argument("--controllers", nargs="+", default=["real_nomem", "real_mem"],
                    choices=sorted(CONTROLLERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=DEV_SEEDS)
    ap.add_argument("--reps", type=int, default=1,
                    help="每个 maze seed 重复几次（衡量 CUDA 数值非确定性）。"
                         "CPU 逐位确定，reps=1 即可，样本量靠更多 maze seeds")
    ap.add_argument("--brain-seed", type=int, default=64, help="所有 rep 用同一个 brain seed")
    ap.add_argument("--sim-seconds", type=float, default=900.0, help="每局上限（到终点提前结束）")
    ap.add_argument("--cells", default="6x4")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--loop", type=float, default=0.08)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--csv", default="", help="每个 rollout 一行的原始结果")
    ap.add_argument("--meta", default="", help="元数据 json（默认跟 --csv 同名 _meta.json）")
    ap.add_argument("--check-repro", action="store_true",
                    help="只做状态泄漏自检（用 CPU，同 seed 重复 reps 次）")
    args = ap.parse_args(argv)

    cells = tuple(int(v) for v in args.cells.lower().split("x"))

    if args.check_repro:
        return check_repro(args, cells)

    print(f"迷宫 {cells[0]}x{cells[1]}   maze seeds {len(args.seeds)} 个 × "
          f"reps {args.reps} 次 = {len(args.seeds) * args.reps} 个 rollout")
    print(f"每局上限 {args.sim_seconds:.0f}s（到终点提前结束）  device={args.device}  "
          f"brain_seed={args.brain_seed}")
    print("注意：reps 衡量的是同一 maze+brain seed 下的 CUDA 数值非确定性，"
          "不是额外的迷宫样本\n")

    rows = []
    rollout_rows = []
    meta_written = False
    for name in args.controllers:
        spec = CONTROLLERS[name]
        t0 = time.perf_counter()
        static = build_static(spec["kind"], spec["mode"], spec["memory"],
                              spec["brain_gain"], str(spec.get("readout") or ""),
                              args.device, args.brain_seed)
        resolved_device = getattr(getattr(static[1], "raw", None), "device", args.device)

        if not meta_written:
            meta = metadata(args.device, args.brain_seed, args.seeds, args.reps,
                            float(static[0].brain.dt), str(spec.get("readout") or ""))
            meta["resolved_device"] = resolved_device
            meta["controllers"] = args.controllers
            meta_path = args.meta or (args.csv.replace(".csv", "") + "_meta.json"
                                      if args.csv else "benchmark_meta.json")
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, ensure_ascii=False)
            print(f"[meta] {meta_path}  device={resolved_device}  "
                  f"commit={meta['git_commit'][:8]}{'-dirty' if meta['git_dirty'] else ''}  "
                  f"cupy={meta['cupy_version']}\n")
            meta_written = True

        per_seed: dict[int, list[bool]] = {}
        all_stats = []
        for seed in args.seeds:
            per_seed[seed] = []
            for rep in range(args.reps):
                st = rollout(static, seed, cells, args.scale, args.loop,
                             args.sim_seconds, args.brain_seed)
                per_seed[seed].append(bool(st["reached_goal"]))
                all_stats.append(st)
                rollout_rows.append({
                    "controller": name, "maze_seed": seed,
                    "brain_seed": args.brain_seed, "device": resolved_device,
                    "rep": rep, **{k: st.get(k) for k in ROLLOUT_FIELDS
                                   if k in st},
                })
                mark = "*" if st["reached_goal"] else " "
                print(f"  {name:<18} maze {seed:>4} rep {rep + 1}/{args.reps}  "
                      f"cov={st['coverage'] * 100:5.1f}%  goal={st['best_goal_dist']:3d}{mark}  "
                      f"t_goal={st['time_to_goal']:7.1f}  "
                      f"contact={st['contact_ratio']:.3f}  replan={st['replans']:5d}  "
                      f"front={st['front_safety_events']:4d}")
                sys.stdout.flush()

        # ---- 汇总：先算每个 maze 的成功概率，再跨 maze 汇总 ----
        p_maze = np.array([sum(v) / len(v) for v in per_seed.values()])
        reached = [s for s in all_stats if s["reached_goal"]]
        agg = {
            "controller": name,
            "device": resolved_device,
            "n_maze_seeds": len(args.seeds),
            "reps": args.reps,
            "n_rollouts": len(all_stats),
            "success_rate": float(p_maze.mean()),
            "success_rate_std_across_mazes": float(p_maze.std()),
            "seeds_always": int(sum(1 for v in per_seed.values() if all(v))),
            "seeds_never": int(sum(1 for v in per_seed.values() if not any(v))),
            "seeds_flaky": int(sum(1 for v in per_seed.values() if any(v) and not all(v))),
            "time_to_goal_mean": float(np.mean([s["time_to_goal"] for s in reached])) if reached else -1.0,
            "time_to_goal_median": float(np.median([s["time_to_goal"] for s in reached])) if reached else -1.0,
            "path_efficiency_mean": float(np.mean([s["path_efficiency"] for s in reached])) if reached else -1.0,
            "distance_mean": float(np.mean([s["distance"] for s in all_stats])),
            "coverage_mean": float(np.mean([s["coverage"] for s in all_stats])),
            "contact_ratio_mean": float(np.mean([s["contact_ratio"] for s in all_stats])),
            "collision_events_mean": float(np.mean([s["collision_events"] for s in all_stats])),
            "stalls_mean": float(np.mean([s["stalls"] for s in all_stats])),
            "front_safety_mean": float(np.mean([s["front_safety_events"] for s in all_stats])),
            "replans_mean": float(np.mean([s["replans"] for s in all_stats])),
            "off_path_events_mean": float(np.mean([s["off_path_events"] for s in all_stats])),
            "max_offpath_dist_mean": float(np.mean([s["max_offpath_dist"] for s in all_stats])),
            "max_target_dist_mean": float(np.mean([s["max_target_dist"] for s in all_stats])),
            "map_explored_mean": float(np.mean([s["map_explored"] for s in all_stats])),
            "brain_ms_per_step": float(np.mean([s["brain_ms_per_step"] for s in all_stats])),
            "wall_seconds": time.perf_counter() - t0,
        }
        rows.append(agg)
        print(f"  -> {name}: 成功率 {agg['success_rate'] * 100:.0f}% "
              f"(跨迷宫 std {agg['success_rate_std_across_mazes'] * 100:.0f}%)  "
              f"全成功 {agg['seeds_always']} / 全失败 {agg['seeds_never']} / "
              f"不稳 {agg['seeds_flaky']}\n")
        sys.stdout.flush()

    print("=" * 118)
    print(f"{'controller':<20}{'success':>10}{'rate':>7}{'std':>6}{'always':>8}{'never':>7}"
          f"{'flaky':>7}{'t_goal':>9}{'pathEff':>9}{'contact':>9}{'replan':>8}"
          f"{'offpath':>9}{'maxTgt':>8}{'front':>7}")
    print("-" * 118)
    for r in rows:
        print(f"{r['controller']:<20}{r['n_maze_seeds']:>4}x{r['reps']:<5}"
              f"{r['success_rate'] * 100:>6.0f}%{r['success_rate_std_across_mazes'] * 100:>5.0f}%"
              f"{r['seeds_always']:>8}{r['seeds_never']:>7}{r['seeds_flaky']:>7}"
              f"{r['time_to_goal_mean']:>9.1f}{r['path_efficiency_mean']:>9.3f}"
              f"{r['contact_ratio_mean']:>9.3f}{r['replans_mean']:>8.0f}"
              f"{r['off_path_events_mean']:>9.1f}{r['max_target_dist_mean']:>8.2f}"
              f"{r['front_safety_mean']:>7.1f}")
    print("=" * 118)
    print("success = n_maze_seeds x reps；rate = 各 maze 成功概率的均值（每个 maze 等权）")
    print("std    = 各 maze 成功概率的离散度（跨迷宫，不是跨 rollout）")
    print("always/never/flaky = 全成功 / 全失败 / 结果不稳的 maze 数")
    print("t_goal = 到达终点的平均耗时(s)，只统计成功的 rollout")
    print("pathEff = 最优格数 / 实际行驶距离（越接近 1 越高效）")
    print("offpath = 因明显偏离路径而失效重规划的均值；maxTgt = 最大瞄准点距离均值")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ROLLOUT_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rollout_rows)
        print(f"[csv] {args.csv}  ({len(rollout_rows)} 行，每个 rollout 一行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
