"""闭环主循环 + 可视化。

    python -m loop.run --brain fake --map track --steps 4000
    python -m loop.run --brain real --map track --check
    python -m loop.compare                      # 两种脑的对比表

注意：必须在项目根目录运行（这些模块用的是绝对导入）。
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from brain.neurons import resolve
from brain.wrap import make_brain
from config import PRESETS, Config, apply_preset
from loop.decode import Decoder
from loop.encode import Encoder
from world.car import DiffDriveCar
from world.maze import MAPS, Maze, generate

# 两种脑的时间尺度不同：真脑的预设是按 dt=20ms 标的，假脑用 10ms 控制更细。
DEFAULT_DT = {"fake": 0.01, "real": 0.02}
# 自检跑多久（**物理秒数**，不是步数）。两种脑的 dt 不同，用秒才能公平比较。
CHECK_SIM_SECONDS = {"fake": 120.0, "real": 120.0}


def make_maze(map_name: str, seed: int = 0, cells: tuple[int, int] = (9, 6),
              scale: int = 2, loop_chance: float = 0.08) -> Maze:
    """map_name 为 "gen" 时现场生成迷宫，否则查内置表或读 .txt。"""
    if map_name == "gen":
        return Maze(generate(cells[0], cells[1], seed=seed, scale=scale,
                             loop_chance=loop_chance))
    if map_name.endswith(".txt"):
        return Maze.load(map_name)
    return Maze.named(map_name)


def make_setup(brain_kind: str = "fake", map_name: str = "track", dt: float | None = None,
               device: str = "auto", seed: int = 64, preset: str | None = None,
               sensory_input: bool = False, brain=None, verbose: bool = True,
               maze_seed: int = 0, maze_cells: tuple[int, int] = (9, 6),
               maze_scale: int = 2, maze_loop: float = 0.08,
               make_decoder: bool = True):
    """装配一次仿真的全部部件。传 brain 可以复用已建好的脑（对比脚本用）。

    make_decoder=False 时不创建 Decoder，留给调用方在**应用完所有 CLI 覆盖参数之后**
    再创建。否则 --mode readout 这类需要在 __init__ 里加载读出头/建 Trace 的配置
    会来不及生效（Decoder 已经按旧 cfg 建好了）。
    """
    cfg = Config()
    cfg.brain.kind = brain_kind
    cfg.brain.device = device
    cfg.brain.seed = seed
    cfg.brain.sensory_input = sensory_input
    cfg.brain.dt = DEFAULT_DT[brain_kind] if dt is None else dt
    apply_preset(cfg, preset or brain_kind)

    maze = make_maze(map_name, maze_seed, maze_cells, maze_scale, maze_loop)
    sx, sy, sth = maze.start
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    if verbose:
        path = maze.optimal_path()
        goal = f", optimal path {path} cells" if path > 0 else ""
        print(f"[world] map={map_name} {maze.w}x{maze.h} start=({sx:.1f},{sy:.1f}){goal}")

    if brain is None:
        brain = make_brain(cfg.brain, verbose=verbose)
    groups = resolve(brain, verbose=verbose)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    enc = Encoder(groups, cfg.encoder, brain.dt, car_max_speed=cfg.car.max_speed,
                  car_radius=cfg.car.radius)
    dec = None
    if make_decoder:
        dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    return cfg, maze, car, brain, groups, enc, dec


def build(args):
    """从命令行参数装配。

    顺序很重要：先建 Config 和应用全部 CLI 覆盖，**最后**才建 Decoder。
    """
    cells = args.maze_cells.lower().split("x")
    if len(cells) != 2:
        raise SystemExit(f"--maze-cells 要写成 WxH，比如 13x9，收到 {args.maze_cells!r}")
    cfg, maze, car, brain, groups, enc, _ = make_setup(
        brain_kind=args.brain, map_name=args.map, dt=args.dt, device=args.device,
        seed=args.seed, preset=args.preset, sensory_input=args.sensory_input,
        maze_seed=args.maze_seed, maze_cells=(int(cells[0]), int(cells[1])),
        maze_scale=args.maze_scale, maze_loop=args.maze_loop,
        make_decoder=False,
    )

    cfg.decoder.mode = args.mode
    cfg.decoder.follow_side = args.follow_side
    cfg.decoder.memory_gain = args.memory
    cfg.decoder.brain_gain = args.brain_gain
    if args.turn_sign is not None:
        cfg.decoder.turn_sign = args.turn_sign
    if args.turn_gain is not None:
        cfg.decoder.turn_gain = args.turn_gain
    if args.fwd_gain is not None:
        cfg.decoder.fwd_gain = args.fwd_gain
    if args.readout:
        cfg.decoder.readout_path = args.readout
    if args.no_front_safety:
        cfg.decoder.front_safety = False
    if args.unknown_cost is not None:
        cfg.nav.unknown_cost = args.unknown_cost
    if args.lookahead is not None:
        cfg.nav.lookahead = args.lookahead
    for attr, val in (("range_noise_std", args.range_noise),
                      ("range_dropout_prob", args.range_dropout),
                      ("pose_xy_noise_std", args.pose_xy_noise),
                      ("pose_theta_noise_std", args.pose_theta_noise),
                      ("motor_tau", args.motor_tau)):
        if val is not None:
            setattr(cfg.noise, attr, val)

    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    return cfg, maze, car, brain, groups, enc, dec


# --------------------------------------------------------------------------- 可视化
def _pick_writer(path: str, fps: int):
    """按扩展名挑动画 writer。mp4 用 imageio-ffmpeg 自带的 ffmpeg 二进制。"""
    import os

    from matplotlib import animation

    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        return animation.PillowWriter(fps=fps)
    if ext in (".mp4", ".m4v", ".mov"):
        return animation.FFMpegWriter(fps=fps, bitrate=2400,
                                      extra_args=["-pix_fmt", "yuv420p"])
    raise SystemExit(f"不认识的视频格式 {ext!r}，用 .gif 或 .mp4")


class Viewer:
    """左面板世界，右面板下行神经元活动。video 非空时录文件而不是开窗口。"""

    def __init__(self, maze, video: str = "", fps: int = 25, dpi: int = 110):
        import matplotlib

        # 必须在第一次 import pyplot 之前选后端
        if video:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if video and video.lower().endswith((".mp4", ".m4v", ".mov")):
            try:
                import imageio_ffmpeg
                matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError as exc:
                raise SystemExit("写 mp4 需要 ffmpeg：pip install imageio-ffmpeg"
                                 "（或改用 --video out.gif）") from exc

        self.plt = plt
        self.video = video
        self.fps = fps
        self.writer = None
        self.frames = 0

        self.fig = plt.figure(figsize=(12.5, 6.4))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.16)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.bx = self.fig.add_subplot(gs[0, 1])

        self.ax.imshow(maze.grid, cmap="gray_r", origin="lower",
                       extent=[0, maze.w, 0, maze.h], vmin=0, vmax=1, alpha=0.28)
        self.ax.set_xlim(0, maze.w)
        self.ax.set_ylim(0, maze.h)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        if maze.goal:
            gx, gy, _ = maze.goal
            self.ax.plot(gx, gy, "*", ms=18, color="tab:green", zorder=5)

        self.tx, self.ty = [], []
        self.traj, = self.ax.plot([], [], "-", lw=1.3, color="tab:blue", alpha=0.85)
        self.rays, = self.ax.plot([], [], "-", lw=0.5, color="tab:red", alpha=0.4)
        self.body, = self.ax.plot([], [], "-", lw=2.0, color="black")
        self.title = self.ax.set_title("")

        names = ["loom_L", "loom_R", "chase_L", "chase_R", "threat_L", "threat_R",
                 "turn_L", "turn_R", "fwd", "back", "escape"]
        self.bar_names = names
        self.bx.set_title("descending neurons (Hz)")
        self.bx.set_xlim(0, 30)
        self.bx.set_ylim(-0.7, len(names) - 0.3)
        self.bx.set_yticks(range(len(names)))
        self.bx.set_yticklabels(names, family="monospace", fontsize=9)
        self.bx.invert_yaxis()
        self.bx.grid(axis="x", alpha=0.25)
        colors = ["tab:red" if n.startswith(("loom", "threat")) else
                  "tab:orange" if n.startswith("chase") else
                  "tab:purple" if n.startswith("turn") else
                  "tab:green" if n == "fwd" else
                  "tab:brown" if n == "back" else "tab:gray" for n in names]
        self.bars = self.bx.barh(range(len(names)), [0] * len(names), color=colors, height=0.62)
        self.bx.text(0.98, 0.02, "", transform=self.bx.transAxes, ha="right",
                     va="bottom", family="monospace", fontsize=9)

        self.fig.tight_layout()
        if video:
            self.writer = _pick_writer(video, fps)
            self.writer.setup(self.fig, video, dpi=dpi)
            print(f"[viz] recording {video} @ {fps} fps")
        else:
            plt.ion()
            plt.show(block=False)

    def update(self, maze, car, enc, dec, dists, step, dt):
        x, y, th = car.pose
        self.tx.append(x)
        self.ty.append(y)
        self.traj.set_data(self.tx, self.ty)

        ang = th + enc.angles
        ex, ey = x + dists * np.cos(ang), y + dists * np.sin(ang)
        n = len(dists)
        X, Y = np.empty(3 * n), np.empty(3 * n)
        X[0::3], X[1::3], X[2::3] = x, ex, np.nan
        Y[0::3], Y[1::3], Y[2::3] = y, ey, np.nan
        self.rays.set_data(X, Y)

        r = car.cfg.radius
        pts = np.array([[2.0 * r, 0.0], [-1.2 * r, 1.3 * r], [-1.2 * r, -1.3 * r], [2.0 * r, 0.0]])
        c, s = np.cos(th), np.sin(th)
        p = pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])
        self.body.set_data(p[:, 0], p[:, 1])

        rt, g = dec.rates, dec.index
        for bar, name in zip(self.bars, self.bar_names):
            bar.set_width(float(rt[g[name]]))

        self.title.set_text(f"t={step * dt:6.1f}s   v={dec.v:+.2f}   w={dec.omega:+.2f}")
        self.bx.texts[0].set_text(
            f"cov {car.coverage(maze) * 100:4.1f}%\n"
            f"hit {car.collision_events:3d}\n"
            f"esc {dec.escapes:3d}\n"
            f"stl {dec.stalls:3d}"
        )

        if self.writer is not None:
            self.writer.grab_frame()
            self.frames += 1
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)

    def close(self, path=""):
        if self.writer is not None:
            self.writer.finish()
            print(f"[viz] {self.video}  ({self.frames} frames)")
        if path:
            self.fig.savefig(path, dpi=130)
            print(f"[viz] saved {path}")
        if not self.video:
            self.plt.ioff()


# --------------------------------------------------------------------------- 主循环
def run(cfg, maze, car, brain, enc, dec, steps, viewer=None, render_every: float = 5,
        log_every=None, verbose=True, stop_on_goal=False, trace_fn=None,
        trace_every=1, event_fn=None):
    """render_every 是"每 N 步重绘一次"，可以是小数——录制时传 1/(fps*dt)，
    内部按时间累加对帧，避免取整让视频时长和仿真时间对不上。

    stop_on_goal=True 时一到终点立刻结束，并按**实际跑了多少步**算统计。
    否则 1200 秒的局里如果 200 秒就到终点，后面 1000 秒的乱跑会污染
    coverage / distance / collision / stall / explored 全部指标。

    trace_fn(rec) 每 trace_every 步被调一次，rec 是一个诊断 dict。只用于分析，
    不影响控制。"""
    dt = brain.dt
    log_every = log_every or max(1, int(round(1.0 / dt)))
    render_period = float(render_every) * dt
    next_render = 0.0
    brain_ns = 0.0
    stalled = False
    best_goal = 10 ** 9
    reached_goal = False
    time_to_goal = -1.0
    distance_to_goal = -1.0
    t0 = time.perf_counter()
    steps_done = 0

    # 记忆层：只用射线 + 位姿建图，**不接收 Maze 实例**
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
        dists = enc.sense(maze, car)
        inject, info = enc.inject(dists)

        if memory is not None:
            memory.update(car.x, car.y, car.theta + enc.angles, dists)
            memory.check_invariants()
            target = memory.next_target(car.x, car.y)
            if target is not None:
                tx, ty, dist = target
                bearing = math.atan2(ty - car.y, tx - car.x)
                err = (bearing - car.theta + np.pi) % (2 * np.pi) - np.pi
                # 纯追踪曲率 omega = 2*v*sin(err)/L。L 必须用**到目标的实际距离**：
                # 用名义前瞻值的话，目标比前瞻近时曲率恒定，车会绕着目标画圈。
                L = max(dist, 0.35)
                v_used = max(abs(dec.v), 0.35 * cfg.car.max_speed)
                turn_pp = 2.0 * v_used * math.sin(err) / (L * cfg.car.max_omega)
                info["pursuit_turn"] = float(np.clip(turn_pp, -1.0, 1.0))
                info["bearing_err"] = float(err)
                info["target"] = (float(tx), float(ty))
                info["target_dist"] = float(dist)
            # 路径失效事件：补上控制侧的量再交给调用方
            if event_fn is not None:
                for ev in memory.drain_events():
                    ev.update({
                        "t": step * dt,
                        "theta": float(car.theta),
                        "target_dist": info.get("target_dist", float("nan")),
                        "bearing_err": info.get("bearing_err", float("nan")),
                        "v": float(dec.v), "omega": float(dec.omega),
                        "brain_turn": float(dec.last_brain_turn),
                        "pursuit_turn": float(dec.last_pursuit_turn),
                        "front_blocked": bool(dec.front_blocked),
                    })
                    event_fn(ev)

        tb = time.perf_counter()
        fired = brain.step(inject=inject)
        brain_ns += time.perf_counter() - tb

        v, omega = dec.update(fired, info=info, stalled=stalled)
        car.set_command(v, omega)
        car.step(maze, dt)

        # 卡死判据：被前方安全层硬停，或者指令上想动实际却没挪窝
        stalled = (dec.front_blocked
                   or (abs(dec.v) > 0.15 * cfg.car.max_speed
                       and car.last_move < 0.3 * abs(dec.v) * dt))

        d = maze.dist_to_goal(car.x, car.y)
        if d >= 0:
            best_goal = min(best_goal, d)
            if d == 0 and not reached_goal:
                reached_goal = True
                time_to_goal = steps_done * dt
                distance_to_goal = car.distance

        if viewer is not None and step * dt >= next_render - 1e-9:
            viewer.update(maze, car, enc, dec, dists, step, dt)
            next_render += render_period
        elif viewer is None and verbose and step % log_every == 0:
            print(f"t={step * dt:6.2f}s v={dec.v:+.2f} w={dec.omega:+.2f} "
                  f"cov={car.coverage(maze) * 100:4.1f}% hit={car.collision_events} "
                  f"esc={dec.escapes} stall={dec.stalls} "
                  f"replan={memory.replans if memory else 0}")

        if trace_fn is not None and step % trace_every == 0:
            trace_fn({
                "step": step, "t": step * dt,
                "x": car.x, "y": car.y, "theta": car.theta,
                "v": dec.v, "omega": dec.omega,
                "brain_turn": dec.last_brain_turn,
                "pursuit_turn": dec.last_pursuit_turn,
                "turn_cmd": dec.last_turn_cmd,
                "dmin_F": info.get("dmin_F", float("nan")),
                "dmin_L": info.get("dmin_L", float("nan")),
                "dmin_R": info.get("dmin_R", float("nan")),
                "front_blocked": dec.front_blocked,
                "target": info.get("target"),
                "target_dist": info.get("target_dist", float("nan")),
                "bearing_err": info.get("bearing_err", float("nan")),
                "path_len": memory.path_len if memory else 0,
                "replans": memory.replans if memory else 0,
                "off_path_events": memory.off_path_events if memory else 0,
                "offpath_dist": memory.last_offpath_dist if memory else float("nan"),
                "goal_dist": d,
            })

        if stop_on_goal and reached_goal:
            break

    wall = time.perf_counter() - t0
    sim_time = steps_done * dt
    optimal = maze.optimal_path()
    stats = {
        "steps": steps,
        "steps_done": steps_done,
        "sim_time": sim_time,
        "wall_time": wall,
        "brain_ms_per_step": brain_ns / max(1, steps_done) * 1e3,
        "distance": car.distance,
        "mean_speed": car.distance / sim_time if sim_time > 0 else 0.0,
        "collisions": car.collisions,
        "collision_events": car.collision_events,
        "contact_ratio": car.collisions / max(1, steps_done),
        "coverage": car.coverage(maze),
        "escapes": dec.escapes,
        "stalls": dec.stalls,
        "front_safety_events": dec.front_blocks,
        "best_goal_dist": best_goal if best_goal < 10 ** 9 else -1,
        "reached_goal": reached_goal,
        "time_to_goal": time_to_goal,
        "distance_to_goal": distance_to_goal,
        "optimal_path_cells": optimal,
        # 路径效率：最优格数 / 实际行驶距离（越接近 1 越好；>1 说明实际比最优短，
        # 那只能是没到终点，看 time_to_goal 一起判断）
        "path_efficiency": (optimal / car.distance
                            if (reached_goal and car.distance > 0 and optimal > 0) else -1.0),
        "map_explored": memory.explored_fraction(len(maze.free_cells())) if memory else -1.0,
        "replans": memory.replans if memory else 0,
        "off_path_events": memory.off_path_events if memory else 0,
        "max_offpath_dist": memory.max_offpath_dist if memory else -1.0,
        "max_offpath_polyline": memory.max_offpath_polyline if memory else -1.0,
        "max_target_dist": memory.max_target_dist if memory else -1.0,
        "invalidations": dict(memory.invalidations) if memory else {},
        "no_path_events": memory.invalidations["NO_PATH"] if memory else 0,
        "map_revision": memory.map_revision if memory else 0,
        "invariant_failures": memory.invariant_failures if memory else 0,
        "realtime_factor": sim_time / wall if wall > 0 else float("inf"),
    }
    return stats


def print_stats(stats):
    print("\n" + "=" * 60)
    print(f"  sim {stats['sim_time']:.1f}s / {stats['steps_done']} steps")
    print(f"  distance      {stats['distance']:.2f} cells   mean speed {stats['mean_speed']:.3f} cell/s")
    print(f"  collisions    {stats['collision_events']} events "
          f"({stats['collisions']} steps in contact, ratio {stats['contact_ratio']:.3f})")
    print(f"  coverage      {stats['coverage'] * 100:.1f}%")
    if stats["best_goal_dist"] >= 0:
        print(f"  goal          closest {stats['best_goal_dist']} cells"
              f"{'   REACHED' if stats['reached_goal'] else ''}")
    if stats["reached_goal"]:
        print(f"  time to goal  {stats['time_to_goal']:.1f}s   "
              f"optimal {stats['optimal_path_cells']} cells   "
              f"path efficiency {stats['path_efficiency']:.3f}")
    print(f"  escapes {stats['escapes']}   stalls {stats['stalls']}   "
          f"front-safety {stats['front_safety_events']}   replans {stats['replans']}")
    if stats["map_explored"] >= 0:
        print(f"  map explored  {stats['map_explored'] * 100:.1f}%")
    print(f"  brain cost    {stats['brain_ms_per_step']:.2f} ms/step")
    print(f"  realtime x    {stats['realtime_factor']:.2f}")
    print("=" * 60)


def main(argv=None):
    ap = argparse.ArgumentParser(description="果蝇脑 -> 四驱车 闭环仿真")
    ap.add_argument("--brain", choices=["fake", "real"], default="fake")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="增益预设，默认跟随 --brain")
    ap.add_argument("--map", default="track",
                    help=f"内置地图 {sorted(MAPS)}、'gen' 现场生成、或 .txt 路径")
    ap.add_argument("--maze-seed", type=int, default=0, help="--map gen 的迷宫种子")
    ap.add_argument("--maze-cells", default="9x6", help="--map gen 的迷宫单元数 WxH")
    ap.add_argument("--maze-scale", type=int, default=2, help="走廊宽度（格）")
    ap.add_argument("--maze-loop", type=float, default=0.08,
                    help="打通的墙的比例，制造环路；0 = 纯树状迷宫（到处死胡同）")
    ap.add_argument("--steps", type=int, default=None,
                    help="脑步数。优先用 --sim-seconds（两种脑的 dt 不同，步数没法直接比）")
    ap.add_argument("--sim-seconds", type=float, default=None,
                    help="仿真时长（秒）。内部换算 steps = round(sim_seconds/dt)，"
                         "这样 fake/real 跑的是同一个物理时长，指标才可比")
    ap.add_argument("--stop-on-goal", action="store_true",
                    help="一到终点立刻结束，并按实际步数算统计（导航 benchmark 必开）")
    ap.add_argument("--dt", type=float, default=None,
                    help=f"脑步长（秒），默认 fake={DEFAULT_DT['fake']} real={DEFAULT_DT['real']}")
    ap.add_argument("--seed", type=int, default=64, help="随机种子，同种子结果可复现")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--headless", action="store_true", help="不渲染，只打日志")
    ap.add_argument("--video", default="", help="录制成 .gif 或 .mp4（不开窗口）")
    ap.add_argument("--fps", type=int, default=25, help="录制帧率")
    ap.add_argument("--every", type=int, default=0,
                    help="每 N 步录一帧（延时摄影）。默认按 fps 实时对帧；"
                         "跑很久的局用 --every 50 --fps 25 可以压成 1/50 时长的快放")
    ap.add_argument("--render-every", type=int, default=5, help="实时窗口每 N 步重绘一次")
    ap.add_argument("--png", default="", help="结束后保存一张截图")
    ap.add_argument("--save", default="", help="保存轨迹到 .npz")
    ap.add_argument("--check", action="store_true", help="无头自检，返回退出码")
    ap.add_argument("--turn-sign", type=float, default=None, help="转向符号，反了就填 -1")
    ap.add_argument("--turn-gain", type=float, default=None)
    ap.add_argument("--fwd-gain", type=float, default=None)
    ap.add_argument("--mode", choices=["dn", "scripted", "readout"], default="dn",
                    help="dn=读下行神经元；scripted=手写贴墙基线；readout=群体读出头")
    ap.add_argument("--readout", default="", help="--mode readout 用的 .npz 路径")
    ap.add_argument("--follow-side", choices=["L", "R"], default="R", help="scripted 贴哪侧墙")
    ap.add_argument("--memory", type=float, default=0.0,
                    help="记忆层权重：>0 开启占用栅格 + 目标导向规划（1.0 是正常强度）")
    ap.add_argument("--brain-gain", type=float, default=1.0, help="脑转向权重，0 = 不要脑")
    ap.add_argument("--unknown-cost", type=float, default=None,
                    help="未知格通行代价（默认 4.0）。1=很敢穿未知区，越大越保守")
    ap.add_argument("--lookahead", type=float, default=None, help="纯追踪前瞻距离（格）")
    ap.add_argument("--no-front-safety", action="store_true",
                    help="关掉前方安全限速层（消融用；只限速，不决定方向）")
    # P1-8 仿真噪声，默认全 0
    ap.add_argument("--range-noise", type=float, default=None, help="测距高斯噪声标准差（格）")
    ap.add_argument("--range-dropout", type=float, default=None, help="每条射线丢失概率")
    ap.add_argument("--pose-xy-noise", type=float, default=None, help="位姿 xy 噪声")
    ap.add_argument("--pose-theta-noise", type=float, default=None, help="位姿朝向噪声")
    ap.add_argument("--motor-tau", type=float, default=None, help="电机一阶延迟（秒）")
    ap.add_argument("--sensory-input", action="store_true", help="保留感觉神经元上的突触（默认切掉）")
    args = ap.parse_args(argv)

    if args.check:
        args.headless = True
        args.video = ""
    sim_seconds = args.sim_seconds
    if sim_seconds is None and args.steps is None:
        sim_seconds = CHECK_SIM_SECONDS[args.brain] if args.check else 400.0

    cfg, maze, car, brain, groups, enc, dec = build(args)

    # 步数由物理时长换算（dt 此时才确定）
    if args.steps is None:
        if sim_seconds is None:
            raise SystemExit("内部错误：sim_seconds 未解析")
        steps = max(2, int(round(sim_seconds / brain.dt)))
    else:
        steps = args.steps
    sim_seconds = steps * brain.dt

    viewer, render_every = None, args.render_every
    if args.video:
        viewer = Viewer(maze, video=args.video, fps=args.fps)
        # 录制对帧：--every 显式指定每 N 步一帧（延时摄影），否则按 fps 实时对帧
        render_every = float(args.every) if args.every > 0 else 1.0 / (args.fps * brain.dt)
        steps = max(steps, 2)
    elif not args.headless:
        viewer = Viewer(maze)

    stats = run(cfg, maze, car, brain, enc, dec, steps, viewer=viewer,
                render_every=render_every, stop_on_goal=args.stop_on_goal)

    if viewer is not None:
        viewer.close(args.png)
    print_stats(stats)

    if args.save:
        np.savez(args.save, distance=stats["distance"], coverage=stats["coverage"],
                 collisions=stats["collision_events"])
        print(f"[save] {args.save}")

    if args.check:
        ok = (stats["distance"] > 2.0
              and stats["collision_events"] < 0.25 * stats["steps_done"]
              and stats["coverage"] > 0.10)
        print(f"\nselfcheck: {'PASS' if ok else 'FAIL'} "
              f"(need distance>2, collision events<25%, coverage>10%)")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
