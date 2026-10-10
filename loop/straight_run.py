"""P2d-0：实体短直道导航验收（最小）。

真 ToF + 真里程计 + 真脑（readout + memory）+ 真电机，在一条**直通道**里从起点开到
`goal_cells`（默认 3 cells ≈ 1.2 m）处的目标并停住，然后量：

  纵向误差 |Δx| · 横向误差 |Δy| · 航向误差 |Δθ| · 碰撞 · 流健康 · bridge STOPPED

**不改冻结 P1**——复用 `loop/physical_run._run_physical` 的循环（ToF 帧门控那套）。

真实场地：直道中线对准目标；前方 ~1.6–1.7 m 放一块宽白板给 F 当墙。
虚拟地图：一条 (`width_cells` 宽 × `length_cells` 长) 的直通道，`S` 起点、`G` 在
`goal_cells` 之外（用临时 .txt 交给 make_setup，之后删掉）。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import sys
import tempfile
import time

import numpy as np

from hardware.odometry_client import OdomClient, OdomError
from hardware.pikachu_bridge import PikachuBridge, PikachuConfig
from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import Tof5Client, TofError
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.odom_shadow_run import OdomRealCar
from loop.physical_run import _run_physical
from loop.physical_stall import MechanicalStallError
from loop.run import make_setup
from loop.shadow_run import RealTimePacer

SENSORS = ORDER


def build_lane_rows(length_cells: int, width_cells: int, goal_cells: int) -> list[str]:
    """直通道 ASCII：四周墙；内部 width_cells 行 × length_cells 列；
    S 在第 1 列、G 在第 1+goal_cells 列，都在宽度方向中间行。"""
    inner_w = length_cells + 2                 # 1 墙 + length + 1 墙
    mid = width_cells // 2
    rows = ["#" * inner_w]
    for y in range(width_cells):
        if y == mid:
            line = list("#" + "." * (inner_w - 2) + "#")
            line[1] = "S"
            line[1 + goal_cells] = "G"
            rows.append("".join(line))
        else:
            rows.append("#" + "." * (inner_w - 2) + "#")
    rows.append("#" * inner_w)
    return rows


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2d-0 实体短直道导航验收")
    ap.add_argument("--brain", choices=["fake", "real"], default="real")
    ap.add_argument("--mode", choices=["dn", "scripted", "readout"], default="readout")
    ap.add_argument("--readout", default="readout_real.npz")
    ap.add_argument("--memory", type=float, default=1.0)
    ap.add_argument("--brain-gain", type=float, default=1.0)
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--sim-seconds", type=float, default=12.0)
    ap.add_argument("--stop-on-goal", action=argparse.BooleanOptionalAction, default=True,
                    help="到达目标即停（用 --no-stop-on-goal 关闭）")
    ap.add_argument("--length-cells", type=int, default=6)
    ap.add_argument("--width-cells", type=int, default=3)
    ap.add_argument("--goal-cells", type=int, default=3)
    ap.add_argument("--meters-per-cell", type=float, required=True)
    ap.add_argument("--pikachu-url", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=0.08)
    ap.add_argument("--max-v", type=float, default=1.0)
    ap.add_argument("--max-w", type=float, default=1.0)
    ap.add_argument("--max-motor-mix", type=float, default=1.0)
    ap.add_argument("--min-cmd", type=float, default=0.50)
    ap.add_argument("--v-gain", type=float, default=1.0)
    ap.add_argument("--w-gain", type=float, default=1.0)
    ap.add_argument("--omega-sign", type=float, default=1.0)
    ap.add_argument("--slew-rate", type=float, default=0.0)
    ap.add_argument("--max-pacer-lag", type=float, default=0.25)
    ap.add_argument("--log", default="straight_log.csv")
    ap.add_argument("--odom-port", type=int, default=8888)
    ap.add_argument("--odom-bind", default="0.0.0.0")
    ap.add_argument("--max-odom-age", type=float, default=0.25)
    ap.add_argument("--tof-port", type=int, default=8889)
    ap.add_argument("--tof-warning-age", type=float, default=0.25)
    ap.add_argument("--tof-hard-stale", type=float, default=0.35)
    ap.add_argument("--odom-wait", type=float, default=5.0)
    ap.add_argument("--tof-wait", type=float, default=5.0)
    ap.add_argument("--tof-geometry", default=None,
                    help="hardware/tof_geometry.json；启用逐原点射线建图（默认 OFF=旧行为）")
    ap.add_argument("--stall-policy", choices=["legacy", "observe", "windowed"], default="legacy",
                    help="stall 判定来源：legacy / observe(只记录，控制仍由 legacy 驱动 Decoder，"
                         "可能倒车；仅限离线/dry-run) / windowed(确认机械堵转 -> STOP+FAILSAFE，不自动倒车)")
    ap.add_argument("--tol-long", type=float, default=0.25)
    ap.add_argument("--tol-lat", type=float, default=0.20)
    ap.add_argument("--tol-head-deg", type=float, default=15.0)
    args = ap.parse_args(argv)

    if not args.dry_run and not args.pikachu_url:
        print("需要 --pikachu-url（或 --dry-run）")
        return 2

    geometry = None
    if args.tof_geometry:
        from loop.tof_geometry import load_geometry
        geometry = load_geometry(args.tof_geometry)
        print(f"[tof] geometry ON: {args.tof_geometry}  ({len(geometry)} sensors)")

    rows = build_lane_rows(args.length_cells, args.width_cells, args.goal_cells)
    print("[lane] virtual straight corridor:")
    for r in rows:
        print("   " + r)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        tmp.write("\n".join(rows))
        tmp.close()
        cfg, maze, car, brain, groups, _enc0, _ = make_setup(
            brain_kind=args.brain, map_name=tmp.name, device=args.device, seed=args.seed,
            sensory_input=False, make_decoder=False, verbose=True)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    cfg.decoder.mode = args.mode
    cfg.decoder.memory_gain = args.memory
    cfg.decoder.brain_gain = args.brain_gain
    if args.readout:
        cfg.decoder.readout_path = args.readout
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    steps = max(2, int(round(args.sim_seconds / brain.dt)))

    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=args.meters_per_cell)
    print(f"[tof] angles={ {n: DEFAULT_ANGLES_DEG[n] for n in ORDER} }")

    gx, gy = float(maze.goal[0]), float(maze.goal[1])
    start_x, start_y = float(car.x), float(car.y)
    print(f"[lane] start=({start_x:.2f},{start_y:.2f}) goal=({gx:.2f},{gy:.2f})  "
          f"= {args.goal_cells} cells = {args.goal_cells*args.meters_per_cell:.2f} m")

    odom = OdomClient(args.odom_port, max_age=args.max_odom_age, bind=args.odom_bind)
    tof = Tof5Client(args.tof_port, warning_age=args.tof_warning_age,
                     hard_stale=args.tof_hard_stale, bind=args.odom_bind)
    odom.start(); tof.start()
    for name, cli, wait, port in (("odom", odom, args.odom_wait, args.odom_port),
                                  ("tof", tof, args.tof_wait, args.tof_port)):
        t0 = time.time()
        while cli.age() is None:
            if time.time() - t0 > wait:
                print(f"[{name}] 等不到数据（UDP :{port}）；退出")
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
        dry_run=args.dry_run, log_path=args.log + ".bridge.csv")
    bridge = PikachuBridge(pcfg)
    hardware_mode = not args.dry_run
    pacer = RealTimePacer(max_lag_s=args.max_pacer_lag, enabled=True,
                          allow_resync=not hardware_mode)

    cols = ["t", "target_x", "target_y", "odom_x", "odom_y", "theta_deg",
            "v", "w", "goal_dist", "cross_track",
            "bearing_err", "brain_turn", "pursuit_turn", "turn_cmd",
            "odom_seq", "odom_age_ms", "odom_frame_new", "odom_dx_m", "odom_dtheta",
            "requested_v", "requested_w", "legacy_stalled", "windowed_stalled",
            "windowed_reason", "startup_grace_active", "stall_sustain_s",
            "decoder_in_stall", "decoder_v", "front_blocked"]
    for n in SENSORS:
        cols += [f"tof_{n}_mm", f"tof_{n}_st"]
    cols += ["stop_reason"]
    fh = open(args.log, "w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(cols)
    state = {"last_row": None, "reason": ""}

    def trace_callback(rec: dict) -> None:
        lag = pacer.wait(rec["t"])
        bridge.note_lag(lag)
        if hardware_mode and lag > args.max_pacer_lag:
            bridge.enter_failsafe(f"pacer 落后 {lag*1000:.0f}ms > {args.max_pacer_lag*1000:.0f}ms")
        bridge.on_step(rec)
        tgt = rec.get("target") or (float("nan"), float("nan"))
        x, y, th = rec["x"], rec["y"], rec["theta"]
        ranges = rec.get("tof_ranges") or {}
        status = rec.get("tof_status") or {}
        row = [f"{rec['t']:.3f}", f"{tgt[0]:.3f}", f"{tgt[1]:.3f}",
               f"{x:.4f}", f"{y:.4f}", f"{math.degrees(th):.1f}",
               f"{rec['v']:.3f}", f"{rec['omega']:.3f}",
               f"{math.hypot(gx - x, gy - y):.4f}", f"{y - start_y:.4f}",
               f"{rec.get('bearing_err', float('nan')):.3f}",
               f"{rec.get('brain_turn', 0.0):.3f}",
               f"{rec.get('pursuit_turn', 0.0):.3f}",
               f"{rec.get('turn_cmd', 0.0):.3f}"]
        for k in ("odom_seq", "odom_age_ms", "odom_frame_new", "odom_dx_m", "odom_dtheta",
                  "requested_v", "requested_w", "legacy_stalled", "windowed_stalled",
                  "windowed_reason", "startup_grace_active", "stall_sustain_s",
                  "decoder_in_stall", "decoder_v", "front_blocked"):
            vv = rec.get(k)
            row.append("" if vv is None else vv)
        for n in SENSORS:
            mm = ranges.get(n)
            row += ["" if mm is None else f"{mm:.0f}", status.get(n, "")]
        row += [state["reason"]]
        writer.writerow(row)
        state["last_row"] = row

    def _on_signal(signum, _f):
        print(f"\n[straight] 信号 {signum}，停车退出")
        bridge.stop(reason=f"signal {signum}"); odom.stop(); tof.stop()
        raise KeyboardInterrupt

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _on_signal)
        except (ValueError, OSError):
            pass

    print(f"[straight] sim {args.sim_seconds:.0f}s = {steps} steps @ {brain.dt*1000:.0f}ms；stop-on-goal")
    stats = None
    stop_reason = "timeout"
    mech_exc = None
    try:
        if not bridge.start():
            print("[straight] 桥接未 RUNNING（预检失败）——仿真照跑，实体不动")
        stats = _run_physical(cfg, maze, real_car, brain, enc, dec, tof, steps,
                              stop_on_goal=args.stop_on_goal, trace_fn=trace_callback,
                              trace_every=1, geometry=geometry, stall_policy=args.stall_policy)
        if stats.get("reached_goal"):
            stop_reason = "goal"
    except MechanicalStallError as exc:
        mech_exc = exc
        stop_reason = "mechanical_stall"
        print(f"\n[straight] *** 机械堵转（windowed 确认）: {exc}")
        print("[straight] *** -> 立即 STOP + latched FAILSAFE；禁止自动倒车 / 自动恢复 ***")
        bridge.enter_failsafe(f"mechanical stall: {exc}")
    except (OdomError, TofError) as exc:
        stop_reason = f"failsafe:{type(exc).__name__}"
        print(f"\n[straight] *** {type(exc).__name__}: {exc} -> FAILSAFE（停车 + latch）")
        bridge.enter_failsafe(f"{type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("[straight] 中断")
    finally:
        state["reason"] = stop_reason
        if state["last_row"] is not None:
            state["last_row"][-1] = stop_reason
            writer.writerow(state["last_row"])
        fh.close()
        bridge.stop(reason="finally"); odom.stop(); tof.stop()

    if mech_exc is not None:
        c = bridge.last_stop_confirmed
        if c is True:
            print("[straight] **机械堵转 STOP 已获串口确认** -> 实体停车已确认")
        elif c is False:
            print(f"[straight] !!! {bridge.last_stop_note} —— 不能声称实体已安全停止 !!!")
        else:
            print(f"[straight] 机械堵转 STOP 状态未知：{bridge.last_stop_note or 'N/A'}")

    fx, fy, fth = float(real_car.x), float(real_car.y), float(real_car.theta)
    long_err = abs(fx - gx) * args.meters_per_cell
    lat_err = abs(fy - gy) * args.meters_per_cell
    head_err = abs(math.degrees(_wrap(fth)))
    bstat, ostat, tstat = bridge.stats(), odom.stats(), tof.stats()
    reached = bool(stats and stats.get("reached_goal"))
    no_collision = (stats is None) or (stats.get("collisions", 0) == 0)
    flows_ok = (ostat.get("dropped", 0) == 0 and ostat.get("bad", 0) == 0
                and ostat.get("reset") is None
                and tstat.get("dropped", 0) == 0 and tstat.get("bad", 0) == 0
                and tstat.get("reset") is None)
    bridge_clean = (bstat.get("frames_failed", 0) == 0
                    and bstat.get("lag_failures", 0) == 0)
    bridge_stopped = (bstat.get("state") == "STOPPED")

    checks = [
        ("reached goal", reached),
        (f"long err {long_err*100:.0f}cm <= {args.tol_long*100:.0f}cm", long_err <= args.tol_long),
        (f"lat err {lat_err*100:.0f}cm <= {args.tol_lat*100:.0f}cm", lat_err <= args.tol_lat),
        (f"heading err {head_err:.0f}deg <= {args.tol_head_deg:.0f}deg",
         head_err <= args.tol_head_deg),
        ("no collision", no_collision),
        ("no stale/reset/fail (odom+tof)", flows_ok),
        ("bridge frames/lag clean", bridge_clean),
        ("bridge STOPPED", bridge_stopped),
    ]
    print("\n============== P2d-0 straight-lane acceptance ==============")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"  final pose x={fx:.3f} y={fy:.3f} theta={math.degrees(fth):.1f}deg  "
          f"goal=({gx:.2f},{gy:.2f})  stop={stop_reason}")
    print(f"  bridge: state={bstat.get('state')} frames_ok={bstat.get('frames_ok')} "
          f"failed={bstat.get('frames_failed')} lag_failures={bstat.get('lag_failures')}")
    print(f"  odom: {ostat}")
    print(f"  tof:  {tstat}")
    print(f"  log:  {args.log}")
    print("===========================================================")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
