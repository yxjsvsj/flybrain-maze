"""从最终 HOLDOUT CSV 生成图表。

A/B/C 的数据**全部直接来自 CSV**，不手工重录。
D 需要轨迹（CSV 里没有），所以对指定 seed 重跑两条 rollout —— 只读控制器，不改任何参数。

    python -m loop.plots --outdir figures
    python -m loop.plots --csv holdout_nomem.csv holdout_mem.csv --traj-seed 1000
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

LABEL = {
    "real_nomem": "real brain\n(no memory)",
    "real_mem": "real brain\n+ memory",
    "real_readout_mem": "real readout\n+ memory",
}
COLOR = {
    "real_nomem": "#B0B0B0",
    "real_mem": "#4C72B0",
    "real_readout_mem": "#DD8452",
}


def load(csv_paths):
    rows = []
    for p in csv_paths:
        rows += list(csv.DictReader(open(p, encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["controller"], []).append(r)
    return by


def nums(rs, key):
    return np.array([float(r[key]) for r in rs])


def reached(rs):
    return [r for r in rs if r["reached_goal"] == "True"]


# --------------------------------------------------------------------------- A
def fig_success(by, order, outdir, plt):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.arange(len(order))
    rates, labels = [], []
    for c in order:
        rs = by[c]
        n_ok = len(reached(rs))
        rates.append(n_ok / len(rs) * 100)
        labels.append(f"{n_ok}/{len(rs)}")
    bars = ax.bar(xs, rates, color=[COLOR[c] for c in order], width=0.55)
    for b, t in zip(bars, labels):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, t,
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[c] for c in order])
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 115)
    ax.axhline(100, color="k", lw=0.6, ls=":")
    n = len(by[order[0]])
    ax.set_title(f"Final HOLDOUT: {n} unseen mazes (seeds 1000-1039), 600 s limit")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_a_success.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- B/C
def _box_strip(ax, by, order, key, ylabel, title, plt, only_reached=True):
    data, pos = [], []
    for i, c in enumerate(order):
        rs = reached(by[c]) if only_reached else by[c]
        if not rs:
            continue
        v = nums(rs, key)
        data.append(v)
        pos.append(i)
    bp = ax.boxplot(data, positions=pos, widths=0.45, patch_artist=True,
                    showfliers=False, medianprops=dict(color="k", lw=1.4))
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(COLOR[c])
        patch.set_alpha(0.45)
    rng = np.random.default_rng(0)
    for i, c in zip(pos, order):
        rs = reached(by[c]) if only_reached else by[c]
        v = nums(rs, key)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=14,
                   color=COLOR[c], alpha=0.75, zorder=3, edgecolors="none")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL[c] for c in order])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def fig_time_to_goal(by, order, outdir, plt):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _box_strip(ax, by, order, "time_to_goal", "time to goal (s)",
               "Time to goal, successful runs only (lower is better)", plt)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_b_time_to_goal.png"), dpi=150)
    plt.close(fig)


def fig_path_efficiency(by, order, outdir, plt):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _box_strip(ax, by, order, "path_efficiency", "path efficiency",
               "Path efficiency = optimal cells / distance travelled (1.0 = optimal)", plt)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_c_path_efficiency.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- D
def fig_trajectories(seed, outdir, plt, device, sim_seconds):
    """对指定 HOLDOUT seed 重跑 real_mem 与 real_readout_mem，画轨迹对比。"""
    from brain.neurons import resolve
    from brain.wrap import make_brain
    from config import Config, apply_preset
    from loop.benchmark import CONTROLLERS
    from loop.decode import Decoder
    from loop.encode import Encoder
    from loop.run import DEFAULT_DT, make_maze, run
    from world.car import DiffDriveCar

    cells, scale, loop_chance = (6, 4), 2, 0.08
    panels = []
    for name in ("real_mem", "real_readout_mem"):
        spec = CONTROLLERS[name]
        kind = str(spec["kind"])
        cfg = Config()
        cfg.brain.kind = kind
        cfg.brain.device = device
        cfg.brain.dt = DEFAULT_DT[kind]
        cfg.brain.seed = 64
        apply_preset(cfg, kind)
        cfg.decoder.mode = str(spec["mode"])
        cfg.decoder.memory_gain = float(spec["memory"])          # type: ignore[arg-type]
        cfg.decoder.brain_gain = float(spec["brain_gain"])       # type: ignore[arg-type]
        ro = str(spec.get("readout") or "")
        if ro:
            cfg.decoder.readout_path = ro

        maze = make_maze("gen", seed, cells, scale, loop_chance)
        brain = make_brain(cfg.brain, verbose=False)
        groups = resolve(brain, verbose=False)
        dn_idx = np.asarray(brain.cells(["descending_neuron"]))
        sx, sy, sth = maze.start
        car = DiffDriveCar(sx, sy, sth, cfg.car)
        enc = Encoder(groups, cfg.encoder, brain.dt, car_max_speed=cfg.car.max_speed,
                      car_radius=cfg.car.radius)
        dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
        brain.reset(cfg.brain.seed)

        traj = []
        st = run(cfg, maze, car, brain, enc, dec,
                 max(2, int(round(sim_seconds / brain.dt))), verbose=False,
                 stop_on_goal=True,
                 trace_fn=lambda r: traj.append((r["x"], r["y"])), trace_every=5)
        panels.append((name, maze, np.array(traj), st))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, (name, maze, traj, st) in zip(axes, panels):
        ax.imshow(maze.grid, cmap="gray_r", origin="lower",
                  extent=[0, maze.w, 0, maze.h], vmin=0, vmax=1, alpha=0.28)
        ax.plot(traj[:, 0], traj[:, 1], "-", lw=1.6, color=COLOR[name])
        ax.plot(traj[0, 0], traj[0, 1], "o", ms=8, color="tab:green", zorder=5)
        ax.plot(*maze.goal[:2], "*", ms=17, color="tab:red", zorder=5)
        ax.set_xlim(0, maze.w)
        ax.set_ylim(0, maze.h)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{LABEL[name].replace(chr(10), ' ')}\n"
                     f"t_goal {st['time_to_goal']:.0f}s   "
                     f"pathEff {st['path_efficiency']:.3f}   "
                     f"off-path {st['off_path_events']}",
                     fontsize=10)
    fig.suptitle(f"HOLDOUT maze seed {seed} (never seen during development)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_d_trajectories.png"), dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="从最终 HOLDOUT CSV 生成图表")
    ap.add_argument("--csv", nargs="+",
                    default=["holdout_nomem.csv", "holdout_mem.csv"])
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--traj-seed", type=int, default=1000)
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--sim-seconds", type=float, default=600.0)
    ap.add_argument("--no-traj", action="store_true", help="跳过图 D（不重跑 rollout）")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.outdir, exist_ok=True)
    by = load(args.csv)
    order = [c for c in ("real_nomem", "real_mem", "real_readout_mem") if c in by]
    print(f"读入 {sum(len(v) for v in by.values())} 行，控制器 {order}")

    fig_success(by, order, args.outdir, plt)
    fig_time_to_goal(by, order, args.outdir, plt)
    fig_path_efficiency(by, order, args.outdir, plt)
    print(f"[A] {args.outdir}/fig_a_success.png")
    print(f"[B] {args.outdir}/fig_b_time_to_goal.png")
    print(f"[C] {args.outdir}/fig_c_path_efficiency.png")

    if not args.no_traj:
        fig_trajectories(args.traj_seed, args.outdir, plt, args.device, args.sim_seconds)
        print(f"[D] {args.outdir}/fig_d_trajectories.png  (seed {args.traj_seed} 重跑)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
