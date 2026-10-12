"""P2d-13b：DNa02 神经响应的基线校正 + 最小动态转向验证（严格离线）。

目标：验证**现有 MaleCNS 下行神经活动**能否稳定驱动左右避障（不是训练人为老师）。

最小神经转向读出（P2 专用）：
  raw_LR   = r[turn_L] - r[turn_R]（DNa02 左右发放差）
  raw_turn = turn_gain * turn_sign * raw_LR        （保留真脑预设 turn_sign=-1 的符号）
  b0       = 无遮挡 A 状态的 raw_turn 稳态均值      （基线，按 seed 标定后固定）
  x        = raw_turn - b0                          （相对基线的变化）
  turn     = clip(EMA_tau(x), -1, 1)                （固定时间窗平滑 + 限幅，抑制稀疏脉冲 bang-bang）
**不使用** ToF 的 bias/距离直接算转向；不用 Dijkstra/隐藏地图/手写避障规则给方向。

闭环只走现有 `P2Control` 公平包络（同 fwd_frac/smooth/front-stop，memory_gain=0）。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import TofFrame
from loop.dn_population_probe import SIZE_KEYS
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.local_avoidance_bench import CAR_R, SCENARIOS, wall_clearance
from loop.run import make_setup
from world.car import DiffDriveCar
from world.maze import Maze

STIM = {"A": {}, "B": {"FR": 400.0}, "C": {"FL": 400.0}, "D": {"F": 400.0},
        "E": {"FL": 400.0, "FR": 400.0}}

# 固定超参（验证前确定，不按验证结果调）
TAU_S = 0.5        # 转向平滑时间常数（秒）
CLAMP = 1.0        # 限幅
SWITCH_TURNS = (("A_B_A", ("A", "B", "A")), ("A_C_A", ("A", "C", "A")),
                ("A_D", ("A", "D")), ("A_E", ("A", "E")))
SEG_S = 3.0        # 每段时长
A_SEG_S = 2.0      # 前导/恢复 A 段时长


class DNReadout:
    """最小 DNa02 转向读出：基线校正 + 固定窗平滑 + 限幅。"""

    def __init__(self, dt, *, b0=0.0, tau_s=TAU_S, clamp=CLAMP, use_baseline=True):
        self.dt = dt
        self.b0 = float(b0)
        self.a = float(1.0 - np.exp(-dt / tau_s))
        self.clamp = float(clamp)
        self.use_baseline = bool(use_baseline)
        self.ema = 0.0
        self.last_raw = 0.0

    def reset(self):
        self.ema = 0.0

    def __call__(self, raw_turn):
        self.last_raw = float(raw_turn)
        x = self.last_raw - (self.b0 if self.use_baseline else 0.0)
        self.ema = (1 - self.a) * self.ema + self.a * x
        return float(np.clip(self.ema, -self.clamp, self.clamp))


def make_frame(stim, seq, t):
    ranges = {n: None for n in ORDER}
    status = {n: "NO_TARGET" for n in ORDER}
    for n, mm in stim.items():
        ranges[n], status[n] = float(mm), "VALID"
    return TofFrame(session_id="sw", seq=seq, t=t, healthy=True, ranges=ranges, status=status,
                    ages_ms={n: 100.0 for n in ORDER}, init_count=0, read_errors=0, reinit_count=0)


def calibrate_baseline(cfg, brain, groups, dn_idx, seed, *, seconds=2.0, tof_hz=10.0, mpc=0.40):
    """用无遮挡 A 状态标定 raw_turn 基线 b0（固定后不再改）。"""
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    brain.reset(seed)
    dt = brain.dt
    n = int(round(seconds / dt))
    period = 1.0 / tof_hz
    k0 = n // 2
    frame, seq, last_t, vals = None, 0, -1e9, []
    for k in range(n):
        t = k * dt
        if frame is None or (t - last_t) >= period - 1e-9:
            seq += 1
            frame = make_frame({}, seq, t)
            last_t = t
        d = enc.sense(frame)
        enc.update(frame, d)
        inject, info = enc.inject()
        fired = brain.step(inject=inject)
        dec.update(fired, info)
        if k >= k0:
            vals.append(float(dec.last_brain_turn))
    return float(np.mean(vals)) if vals else 0.0


def run_switch_test(cfg, brain, groups, dn_idx, seed, readout, plan, *, tof_hz=10.0, mpc=0.40):
    """单条连续时间序列 A→刺激→A：脑连续运行，**不在切换时重置**。返回各段 turn 数组。"""
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    brain.reset(seed)
    readout.reset()
    dt = brain.dt
    period = 1.0 / tof_hz
    frame, seq, last_t, total = None, 0, -1e9, 0
    segs = []
    for name, dur in plan:
        m = int(round(dur / dt))
        arr = []
        for _ in range(m):
            tt = total * dt
            if frame is None or (tt - last_t) >= period - 1e-9:
                seq += 1
                frame = make_frame(STIM[name], seq, tt)
                last_t = tt
            d = enc.sense(frame)
            enc.update(frame, d)
            inject, info = enc.inject()
            fired = brain.step(inject=inject)
            dec.update(fired, info)
            arr.append(readout(dec.last_brain_turn))
            total += 1
        segs.append({"seg": name, "turn": np.array(arr, float)})
    return segs


def summarize_test(segs, dt):
    """对一条 A→刺激→A 序列给出：基线/刺激/恢复均值、饱和、符号、响应延迟。"""
    a0 = float(np.mean(segs[0]["turn"])) if len(segs[0]["turn"]) else 0.0
    stim = segs[1]["turn"]
    sm = float(np.mean(stim)) if len(stim) else 0.0
    a1 = float(np.mean(segs[-1]["turn"])) if len(segs) > 2 and len(segs[-1]["turn"]) else float("nan")
    sat = float(np.mean(np.abs(stim) >= 0.999)) * 100 if len(stim) else 0.0
    lat = next((j * dt for j, v in enumerate(stim) if abs(v) >= 0.5), -1.0)
    return {"a0": a0, "stim": sm, "a1": a1, "sat": sat, "sign": float(np.sign(sm)),
            "lat": lat, "seg": segs[1]["seg"]}


def _dynamic(cfg, brain, groups, dn, seed, kind, readout, rows, *, seconds=40.0, mpc=0.40):
    from loop.dynamic_tof_sim import load_rig, run_dynamic
    from loop.p2_control import P2Control, turn_dn_single
    maze = Maze(rows)
    sx, sy, sth = maze.start
    cfg.decoder.mode = "dn"
    cfg.decoder.escape_thresh = 1e9
    fe = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn)
    if kind == "DN_orig":
        src = turn_dn_single
    elif kind == "DN_readout":
        src = lambda c: readout(c.last_brain_turn)
    elif kind == "DN_zero":
        src = lambda c: 0.0
    else:                                   # DN_swap
        src = lambda c: -readout(c.last_brain_turn)
    ctl = P2Control(cfg.decoder, fe, src, name=kind)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    brain.reset(seed)
    readout.reset()
    traj, s = run_dynamic(cfg, maze, car, brain, enc, ctl, memory_on=False, seconds=seconds,
                          tof_hz=10.0, rig=load_rig(), seed=seed)
    clr = min(wall_clearance(maze, p["x"], p["y"], CAR_R) for p in traj) if traj else float("nan")
    best, last_improve = float("inf"), 0.0
    for p in traj:
        if p["goal_dist"] >= 0 and p["goal_dist"] < best:
            best, last_improve = p["goal_dist"], p["t"]
    T = traj[-1]["t"] if traj else 0.0
    return {"reach": bool(s["reached_goal"]), "cev": int(s["collision_events"]),
            "cstep": int(s["collision_steps"]), "minGap": float(clr), "bestD": int(s["best_goal_dist"]),
            "stuck": bool((not s["reached_goal"]) and (T - last_improve) > 8.0),
            "sat%": ctl.saturation_ratio() * 100, "stop%": ctl.stop_ratio() * 100,
            "t_goal": float(s["time_to_goal_s"])}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="64,101,202,303")
    ap.add_argument("--val-seed", type=int, default=512, help="未参与调试的新 seed")
    ap.add_argument("--brain", default="real")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dyn-seconds", type=float, default=40.0)
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=seeds[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn = np.asarray(brain.cells(["descending_neuron"]))
    print(f"[P2d-13b] DNa02 minimal readout  tau={TAU_S}s clamp={CLAMP}  "
          f"seeds={seeds} val={args.val_seed}")
    print("       基线 b0 由无遮挡 A 标定后固定；不用 ToF bias/距离做方向\n")

    # 标定 b0（每 seed，一次，固定）
    b0 = {sd: calibrate_baseline(cfg, brain, groups, dn, sd) for sd in seeds + [args.val_seed]}

    # ---- 任务2/3：连续刺激切换 ----
    print("任务2 连续刺激切换（脑连续不重置；turn=基线校正+平滑 DNa02）")
    print(f"{'seed':>5s} {'test':6s} {'A0':>7s} {'stim':>7s} {'A1':>7s} {'stimSat%':>8s} "
          f"{'sign':>5s} {'lat_s':>6s}")
    for sd in list(seeds) + [args.val_seed]:
        for tname, steps in SWITCH_TURNS:
            plan = [("A", A_SEG_S), (steps[1], SEG_S)]
            if len(steps) > 2:
                plan.append(("A", SEG_S))
            rd = DNReadout(brain.dt, b0=b0[sd])
            segs = run_switch_test(cfg, brain, groups, dn, sd, rd, plan)
            s = summarize_test(segs, brain.dt)
            print(f"{sd:5d} {tname:6s} {s['a0']:+7.3f} {s['stim']:+7.3f} {s['a1']:+7.3f} "
                  f"{s['sat']:8.1f} {s['sign']:+5.0f} {s['lat']:6.2f}")
        print()

    # ---- 任务4/5：动态 must_left / must_right（公平包络，memory_gain=0） ----
    print("任务4/5 动态验证（P2Control 同包络, memory_gain=0, seconds=%.0f）" % args.dyn_seconds)
    print(f"{'scenario':11s} {'ctrl':10s} {'reach':>5s} {'t_goal':>7s} {'cev':>4s} {'cstep':>6s} "
          f"{'minGap':>7s} {'bestD':>6s} {'sat%':>5s} {'stop%':>6s} {'stuck':>5s}")
    for name in ("must_left", "must_right"):
        rows = SCENARIOS[name]
        for kind in ("DN_orig", "DN_readout", "DN_zero", "DN_swap"):
            r = _dynamic(cfg, brain, groups, dn, seeds[0], kind,
                         DNReadout(brain.dt, b0=b0[seeds[0]]), rows, seconds=args.dyn_seconds)
            print(f"{name:11s} {kind:10s} {str(r['reach'])[:1]:>5s} {r['t_goal']:7.2f} "
                  f"{r['cev']:4d} {r['cstep']:6d} {r['minGap']:7.3f} {r['bestD']:6d} "
                  f"{r['sat%']:5.1f} {r['stop%']:6.1f} {str(r['stuck'])[:1]:>5s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
