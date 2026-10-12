"""P2 时序埋点离线验证：模拟运动 odom 的完整控制循环（无 Pi、无电机）。

用真实 PikachuBridge(dry_run=True)（不接 HTTP/电机）挂 Timing，跑 _run_physical 满循环；
odom 模拟前进 -> 车在 grace 之后仍在动 -> 不得 mechanical。校验埋点计数/时长。

运行：  python -m loop.test_timing_loop
"""
from __future__ import annotations

import sys
import time

import numpy as np

from hardware.odometry import LEFT_SIGN, RIGHT_SIGN
from hardware.pikachu_bridge import PikachuBridge, PikachuConfig
from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.odom_shadow_run import ODOM_WHEEL_DIAM, OdomRealCar
from loop.physical_run import _run_physical
from loop.run import make_setup
from loop.test_stall_e2e import COUNT_PER_M, FakeOdomClient, FakeTofClient
from loop.timing_collect import Timing

MPC = 0.40
RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== P2 timing loop (offline, simulated moving odom) ===========")
    cfg, maze, car, brain, groups, _e0, _ = make_setup(
        brain_kind="fake", map_name="gen", dt=None, device="cpu", seed=64,
        preset=None, sensory_input=False, maze_seed=1000, maze_cells=(6, 4),
        maze_scale=2, maze_loop=0.08, make_decoder=False, verbose=False)
    cfg.decoder.mode = "dn"
    cfg.decoder.memory_gain = 0.0
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=MPC)

    def ticks(seq):
        d = max(0, seq - 2)
        fwd = int(round(0.004 * d * COUNT_PER_M))
        return int(LEFT_SIGN * fwd), int(RIGHT_SIGN * fwd)

    def pose(seq):
        d = max(0, seq - 2)
        return (0.004 * d, 0.0, 0.0)

    real_car = OdomRealCar(car.x, car.y, car.theta, cfg.car,
                           FakeOdomClient(ticks, pose), MPC, wheel_diam=ODOM_WHEEL_DIAM)  # type: ignore[arg-type]
    tof = FakeTofClient()
    bridge = PikachuBridge(PikachuConfig(
        dry_run=True, rate_hz=50.0, log_path="",
        sim_max_speed=cfg.car.max_speed, sim_max_omega=cfg.car.max_omega, min_cmd=0.0))
    timing = Timing()
    bridge.timing = timing  # type: ignore[assignment]
    bridge.start()

    def trace(rec):
        timing.count("on_step")
        bridge.on_step(rec)
        time.sleep(0.02)          # 让墙钟越过 grace(0.7)+sustain(0.5)，使“运动不误判”有意义

    steps = 75
    exc = None
    try:
        _run_physical(cfg, maze, real_car, brain, enc, dec, tof, steps,
                      stop_on_goal=False, trace_fn=trace, trace_every=1,
                      geometry=None, stall_policy="windowed", timing=timing)
    except Exception as e:                                     # noqa: BLE001
        exc = e
        import traceback
        traceback.print_exc()
    bridge.stop(reason="test")

    check("full loop ran with no MechanicalStallError (moving past grace)", exc is None)
    if exc is not None:
        print("  EXC:", repr(exc))
    c = timing.counts()
    check(f"control_step counted ({c.get('control_step')}) == steps", c.get("control_step") == steps)
    check(f"on_step counted ({c.get('on_step')}) == steps", c.get("on_step") == steps)
    check(f"dry bridge sent http DRIVE frames ({c.get('http_drive_dry', 0)})",
          (c.get("http_drive_dry", 0) or 0) > 0)
    check(f"no spurious STOP while RUNNING (http_stop={c.get('http_stop', 0)})",
          (c.get("http_stop_dry", 0) or 0) == 0)
    for k in ("brain_ms", "dec_update_ms", "odom_step_ms", "control_step_compute_ms", "tof_enc_ms"):
        check(f"{k}: n={timing.stats(k)['n']}", timing.stats(k)["n"] == steps)
    print("\n[timing] counts:", c)
    print("[timing] brain_ms:", timing.stats("brain_ms"))
    print("[timing] control_step_compute_ms:", timing.stats("control_step_compute_ms"))
    print("[timing] http_drive interval:", timing.stats("http_drive_interval_dry"))
    print("[timing] empty_mailbox:", c.get("empty_mailbox", 0))

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
