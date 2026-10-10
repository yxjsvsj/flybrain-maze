"""P2a.5：里程计在环影子运行器（hardware-in-the-loop）。

回路
----
    虚拟迷宫射线（从**真实位姿**发出）
      -> RealFlyBrain / readout + Memory / Dijkstra
      -> (v, omega)
      -> 物理 Pikachu 电机
      -> MG513 编码器 -> Pi 里程计 -> 50Hz UDP
      -> Windows 端真实位姿
      -> 下一步

= 物理执行器 + 物理编码器反馈 + **虚拟**环境/射线 = HIL。

约束与语义（重要）
------------------
- **不修改任何 frozen P1 文件**。位姿通过 `OdomRealCar(DiffDriveCar)` 子类注入冻结的 `run()`，
  其 `step()` 从真实里程计取位姿（米 -> 迷宫格），不做仿真积分、不做虚拟碰撞强制。
- **坐标变换**：Pi 里程计是米，`car.x/y` 是迷宫格，不能直接赋值：
      alpha     = maze_start.theta - odom0.theta
      maze_xy   = maze_start.xy + R(alpha) * (odom_xy - odom0.xy) / meters_per_cell
      maze_theta= wrap(odom.theta + alpha)
  `--meters-per-cell` 必填，不写死。
- **stale 硬失败**：odom age > `--max-odom-age` → `OdomStaleError` → 从 `run()` 抛出 →
  本运行器捕获 → `bridge.enter_failsafe()` → STOP / latch → 本轮 abort，**不自动恢复运动**。
  （在 car.step() 抛出，所以本周期 trace/hardware command 还没发就被截住。）
- **collisions/contact_ratio 在 P2a.5 不是物理碰撞测量**，不能当作真车零碰撞。
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time

import numpy as np

from config import Config, apply_preset  # noqa: F401  (保持与 shadow_run 一致的导入面)
from hardware.odometry import CPR, LEFT_SIGN, RIGHT_SIGN, WHEEL_DIAM
from hardware.odometry_client import OdomClient, OdomError
from hardware.pikachu_bridge import BridgeState, PikachuBridge, PikachuConfig
from loop.decode import Decoder
from loop.run import make_setup, print_stats, run
from loop.shadow_run import RealTimePacer
from world.car import DiffDriveCar

M_PER_COUNT = math.pi * WHEEL_DIAM / CPR


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class OdomRealCar(DiffDriveCar):
    """冻结 run() 用的车：step() 用真实里程计位姿（米->格）。不做虚拟碰撞强制。"""

    def __init__(self, x0: float, y0: float, th0: float, car_cfg,
                 client: OdomClient, meters_per_cell: float):
        super().__init__(x0, y0, th0, car_cfg)
        self.client = client
        self.m_per_cell = float(meters_per_cell)
        self._sx, self._sy, self._sth = float(x0), float(y0), float(th0)
        self._ox0 = self._oy0 = self._oth0 = 0.0
        self._have_origin = False
        self._alpha = 0.0
        self._pl = self._pr = None
        self.origin_captured = False
        self.last_sample = None            # 最近一帧 OdomSample（供 stall detector 复用）
        self.last_age = None

    def _ensure_origin(self, s) -> None:
        if not self._have_origin:
            self._ox0, self._oy0, self._oth0 = s.x, s.y, s.theta
            self._alpha = self._sth - self._oth0
            self._have_origin = True
            self.origin_captured = True

    def step(self, maze, dt: float) -> bool:
        s, _age = self.client.latest()          # stale / reset -> 抛 OdomError
        self.last_sample = s
        self.last_age = _age
        self._ensure_origin(s)

        dx, dy = s.x - self._ox0, s.y - self._oy0
        ca, sa = math.cos(self._alpha), math.sin(self._alpha)
        self.x = self._sx + (ca * dx - sa * dy) / self.m_per_cell
        self.y = self._sy + (sa * dx + ca * dy) / self.m_per_cell
        self.theta = wrap_pi(s.theta + self._alpha)

        # last_move / distance 用编码器 tick 差（不看 pose 欧氏差），供冻结 stall detector
        pl, pr = self._pl, self._pr
        if pl is None or pr is None:
            self._pl, self._pr = s.left, s.right
            self.last_move = 0.0
        else:
            d_left = LEFT_SIGN * (s.left - pl) * M_PER_COUNT
            d_right = RIGHT_SIGN * (s.right - pr) * M_PER_COUNT
            self._pl, self._pr = s.left, s.right
            move = abs(0.5 * (d_left + d_right)) / self.m_per_cell
            self.last_move = move
            self.distance += move

        self._mark_visited()
        return False                            # P2a.5：无虚拟碰撞强制


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2a.5 里程计在环影子")
    # ---- 迷宫 / 脑（同 shadow_run）----
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
    # ---- 马达桥（同 shadow_run）----
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
    ap.add_argument("--log", default="odom_hardware_log.csv")
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--max-pacer-lag", type=float, default=0.25)
    # ---- 里程计 ----
    ap.add_argument("--odom-port", type=int, required=True, help="收 Pi UDP 遥测的端口")
    ap.add_argument("--odom-bind", default="0.0.0.0")
    ap.add_argument("--max-odom-age", type=float, default=0.25,
                    help="秒；超过即 ODOM_STALE -> FAILSAFE")
    ap.add_argument("--meters-per-cell", type=float, required=True,
                    help="真实世界里一个迷宫格多大（米）。必填，不写死")
    ap.add_argument("--odom-wait", type=float, default=5.0,
                    help="开跑前等第一个 odom 包的最长秒数")
    args = ap.parse_args(argv)

    if not args.dry_run and not args.pikachu_url:
        print("需要 --pikachu-url（或先加 --dry-run）")
        return 2
    if not args.dry_run and args.no_realtime:
        print("实体运动必须开节拍。--no-realtime 只允许配合 --dry-run。")
        return 2

    parts = args.maze_cells.lower().split("x")
    cells = (int(parts[0]), int(parts[1]))
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

    # ---- odom 客户端：先起，等到第一个包 ----
    client = OdomClient(args.odom_port, max_age=args.max_odom_age, bind=args.odom_bind)
    client.start()
    print(f"[odom] 监听 UDP :{args.odom_port}  bind={args.odom_bind}  "
          f"max_age={args.max_odom_age*1000:.0f}ms  meters_per_cell={args.meters_per_cell:g}")
    t_wait = time.time()
    while client.age() is None:
        if time.time() - t_wait > args.odom_wait:
            print("[odom] 等不到 odom 包；退出。检查 Pi 端是否在跑 "
                  "`odometry.py --udp <本机IP>:<port>`。")
            client.stop()
            return 1
        time.sleep(0.05)
    s0, _ = client.latest()
    print(f"[odom] 第一个包 seq={s0.seq} x={s0.x:+.3f} y={s0.y:+.3f} th={s0.theta:+.3f}")

    real_car = OdomRealCar(car.x, car.y, car.theta, cfg.car, client, args.meters_per_cell)
    print(f"[odom] 迷宫起点 maze=({car.x:.2f},{car.y:.2f},{car.theta:+.2f})  "
          f"odom0=({s0.x:+.3f},{s0.y:+.3f},{s0.theta:+.3f})  alpha 在首步确定")
    print("[note] collisions/contact_ratio 在 P2a.5 **不是物理碰撞测量**（无虚拟碰撞强制）")

    pcfg = PikachuConfig(
        base_url=args.pikachu_url or "http://127.0.0.1:8000",
        rate_hz=args.rate_hz, timeout_s=args.timeout,
        sim_max_speed=cfg.car.max_speed, sim_max_omega=cfg.car.max_omega,
        v_gain=args.v_gain, w_gain=args.w_gain, omega_sign=args.omega_sign,
        max_v=args.max_v, max_w=args.max_w, max_motor_mix=args.max_motor_mix,
        min_cmd=args.min_cmd,
        slew_rate=args.slew_rate, dry_run=args.dry_run, log_path=args.log)
    bridge = PikachuBridge(pcfg)
    hardware_mode = not args.dry_run
    pacer = RealTimePacer(max_lag_s=args.max_pacer_lag,
                          enabled=not args.no_realtime,
                          allow_resync=not hardware_mode)

    print(f"[odom] sim {args.sim_seconds:.0f}s = {steps} steps @ dt={brain.dt*1000:.0f}ms")
    print(f"[odom] 发送 {args.rate_hz:g}Hz  timeout {args.timeout*1000:.0f}ms  "
          f"realtime={'off' if args.no_realtime else 'on'}  log={args.log}")

    def trace_callback(rec: dict) -> None:
        lag = pacer.wait(rec["t"])
        bridge.note_lag(lag)
        if hardware_mode and lag > args.max_pacer_lag:
            bridge.enter_failsafe(
                f"pacer 落后 {lag*1000:.0f}ms > {args.max_pacer_lag*1000:.0f}ms："
                f"实体已比仿真多执行旧命令，停车")
        bridge.on_step(rec)

    def _on_signal(signum, _frame):
        print(f"\n[odom] 收到信号 {signum}，停车并退出")
        bridge.stop(reason=f"signal {signum}")
        client.stop()
        raise KeyboardInterrupt

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _on_signal)
        except (ValueError, OSError):
            pass

    stats = None
    try:
        if not bridge.start():
            print("[odom] !! 桥接未进入 RUNNING（预检失败）。仿真照常跑，但实体不会动。")
        stats = run(cfg, maze, real_car, brain, enc, dec, steps,
                    trace_fn=trace_callback, trace_every=1,
                    stop_on_goal=args.stop_on_goal, verbose=False)
    except OdomError as exc:
        print(f"\n[odom] *** {type(exc).__name__}: {exc}")
        print("[odom] *** -> FAILSAFE（停车 + latch，不自动恢复运动；需人工重启）***")
        bridge.enter_failsafe(f"odom {type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        print("[odom] 中断")
    finally:
        bridge.stop(reason="finally")
        client.stop()

    if stats is not None:
        print("[note] 下面 statistics 里的 collisions/contact_ratio **不是物理碰撞测量**")
        print_stats(stats)
    print("\n[odom] bridge:", bridge.stats())
    print("[odom] odom:", client.stats())
    print(f"[odom] pacer: slept={pacer.slept_s:.1f}s resyncs={pacer.resyncs} "
          f"late={pacer.late_events} max_lag_seen={pacer.max_lag_seen*1000:.0f}ms "
          f"lag_failures={bridge.lag_failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
