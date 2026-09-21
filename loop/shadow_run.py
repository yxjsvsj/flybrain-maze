"""影子运行：仿真照常跑，**同一份 (v, omega)** 同时发给 Pikachu 实体车。

**不修改任何冻结 P1 文件。** 挂点是 loop.run() 里已有的 trace_fn —— 它每步给出
dec.v / dec.omega，正是这一步真正下发给仿真车的指令。

第一阶段：实体不提供任何反馈（不接摄像头/里程计），只是"影子执行器"。

    # 1) 干跑（不发 HTTP），只看日志和节拍
    python -m loop.shadow_run --dry-run --no-realtime \
        --brain real --mode readout --readout readout_real.npz --memory 1.0 \
        --map gen --maze-seed 1000 --maze-cells 6x4 --sim-seconds 20

    # 2) 真跑：仿真 1s ≈ 墙钟 1s，实体跟随
    python -m loop.shadow_run --pikachu-url http://192.168.1.50:8000 \
        --brain real --mode readout --readout readout_real.npz --memory 1.0 \
        --map gen --maze-seed 1000 --maze-cells 6x4 --sim-seconds 120 \
        --stop-on-goal --max-v 0.30 --max-w 0.30 --max-motor-mix 0.30 \
        --log hardware_log.csv
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

import numpy as np

from brain.neurons import resolve
from config import Config, apply_preset
from hardware.pikachu_bridge import BridgeState, PikachuBridge, PikachuConfig
from loop.decode import Decoder
from loop.run import DEFAULT_DT, make_maze, make_setup, print_stats, run
from world.car import DiffDriveCar


class RealTimePacer:
    """让仿真时间跟墙钟对齐：sim 1s ≈ wall 1s。

    用绝对目标时间 `t0 + sim_t` 而不是"每次睡一个周期"，避免累计漂移。
    落后超过 max_lag 时重新对齐，避免"追赶"造成指令突发。

    只在 shadow runner 里阻塞；Bridge 本身仍是非阻塞的，冻结的 run() 也不变。
    """

    def __init__(self, max_lag_s: float = 0.5, enabled: bool = True):
        self.enabled = bool(enabled)
        self.max_lag = float(max_lag_s)
        self.t0 = time.perf_counter()
        self.resyncs = 0
        self.slept_s = 0.0
        self.max_lag_seen = 0.0

    def wait(self, sim_t: float) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        target = self.t0 + sim_t
        lag = now - target
        if lag > self.max_lag_seen:
            self.max_lag_seen = lag
        if lag > self.max_lag:                 # 落后太多：重新对齐，不追赶
            self.t0 = now - sim_t
            self.resyncs += 1
            return
        if lag < 0:
            d = -lag
            self.slept_s += d
            time.sleep(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pikachu 影子运行（不改冻结 P1）")
    # ---- 仿真侧（与 loop.run 的核心子集一致）----
    ap.add_argument("--brain", choices=["fake", "real"], default="real")
    ap.add_argument("--map", default="gen")
    ap.add_argument("--maze-seed", type=int, default=1000)
    ap.add_argument("--maze-cells", default="6x4")
    ap.add_argument("--maze-scale", type=int, default=2)
    ap.add_argument("--maze-loop", type=float, default=0.08)
    ap.add_argument("--sim-seconds", type=float, default=120.0)
    ap.add_argument("--stop-on-goal", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--mode", choices=["dn", "scripted", "readout"], default="dn")
    ap.add_argument("--readout", default="")
    ap.add_argument("--memory", type=float, default=1.0)
    ap.add_argument("--brain-gain", type=float, default=1.0)
    # ---- 桥接侧 ----
    ap.add_argument("--pikachu-url", default="", help="例如 http://192.168.1.50:8000")
    ap.add_argument("--dry-run", action="store_true", help="不发 HTTP，只记录")
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=0.08, help="必须 < 1/rate-hz")
    ap.add_argument("--max-v", type=float, default=0.30)
    ap.add_argument("--max-w", type=float, default=0.30)
    ap.add_argument("--max-motor-mix", type=float, default=0.30,
                    help="Nano 内部 motorA=V+W / motorB=V-W，限制单电机不超过它")
    ap.add_argument("--v-gain", type=float, default=1.0)
    ap.add_argument("--w-gain", type=float, default=1.0)
    ap.add_argument("--omega-sign", type=float, default=1.0, help="实物左右接反时改 -1")
    ap.add_argument("--slew-rate", type=float, default=0.0,
                    help="归一化单位/秒；第一阶段保持 0（已有 Decoder 平滑 + Nano PWM ramp）")
    ap.add_argument("--log", default="hardware_log.csv")
    # ---- 节拍 ----
    ap.add_argument("--no-realtime", action="store_true",
                    help="不按墙钟节拍（干跑用；真跑实体时必须开节拍）")
    ap.add_argument("--max-lag", type=float, default=0.5)
    args = ap.parse_args(argv)

    if not args.dry_run and not args.pikachu_url:
        print("需要 --pikachu-url（或先加 --dry-run 干跑）")
        return 2

    cells = tuple(int(v) for v in args.maze_cells.lower().split("x"))
    # 顺序很重要：先应用全部配置，最后建 Decoder（P0-3 修复的口径）
    cfg, maze, car, brain, groups, enc, _ = make_setup(
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
    pcfg = PikachuConfig(
        base_url=args.pikachu_url or "http://127.0.0.1:8000",
        rate_hz=args.rate_hz, timeout_s=args.timeout,
        sim_max_speed=cfg.car.max_speed, sim_max_omega=cfg.car.max_omega,
        v_gain=args.v_gain, w_gain=args.w_gain, omega_sign=args.omega_sign,
        max_v=args.max_v, max_w=args.max_w, max_motor_mix=args.max_motor_mix,
        slew_rate=args.slew_rate, dry_run=args.dry_run, log_path=args.log)
    bridge = PikachuBridge(pcfg)
    pacer = RealTimePacer(max_lag_s=args.max_lag, enabled=not args.no_realtime)

    print(f"[shadow] sim {args.sim_seconds:.0f}s = {steps} steps @ dt={brain.dt*1000:.0f}ms")
    print(f"[shadow] 缩放: v/{cfg.car.max_speed:.2f}*{args.v_gain:g} -> |V|<={args.max_v:.2f}   "
          f"omega/{cfg.car.max_omega:.2f}*{args.w_gain:g}*{args.omega_sign:+g} -> |W|<={args.max_w:.2f}   "
          f"mix<={args.max_motor_mix:.2f}")
    print(f"[shadow] 发送 {args.rate_hz:g}Hz  timeout {args.timeout*1000:.0f}ms  "
          f"realtime={'off' if args.no_realtime else 'on'}  log={args.log}")

    def trace_callback(rec: dict) -> None:
        pacer.wait(rec["t"])
        bridge.on_step(rec)

    def _on_signal(signum, _frame):
        print(f"\n[shadow] 收到信号 {signum}，停车并退出")
        bridge.stop(reason=f"signal {signum}")
        raise KeyboardInterrupt

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _on_signal)
        except (ValueError, OSError):
            pass

    stats = None
    try:
        with bridge:
            if bridge.state is not BridgeState.RUNNING:
                print("[shadow] !! 桥接未进入 RUNNING（预检失败）。"
                      "仿真照常跑，但实体不会动。")
            stats = run(cfg, maze, car, brain, enc, dec, steps,
                        trace_fn=trace_callback, trace_every=1,
                        stop_on_goal=args.stop_on_goal, verbose=False)
    except KeyboardInterrupt:
        print("[shadow] 中断")
    finally:
        bridge.stop(reason="finally")

    if stats is not None:
        print_stats(stats)
    print("\n[shadow] bridge:", bridge.stats())
    print(f"[shadow] pacer: slept={pacer.slept_s:.1f}s resyncs={pacer.resyncs} "
          f"max_lag_seen={pacer.max_lag_seen*1000:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
