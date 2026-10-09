"""P2d 静态领域诊断（**绝对不动车**）。

数据链：真实 ToF → `PhysicalToFEncoder` → `RealFlyBrain` → DN `Trace` → `readout.predict`
→ CSV（每个新 ToF 帧一行）+ 完整 feats 的 `.npz`。

安全
----
- **不启动 `PikachuBridge`、不发送任何电机指令、不需要 odom 运动**。
- odom 可选，仅作静态漂移参考。
- Ctrl+C 安全退出（关文件、存 npz、关 socket）。

用途
----
拿到"未做几何补偿"的基线中间量（size/growth/dmin/loom/chase/threat/prox/bias、
DN feats 摘要、readout 输出），供之后与 **P1 synthetic scene** 做 apples-to-apples
分布对比（PCA / z-score / cosine）。**不改冻结 `loop/encode.py` / `loop/decode.py`。**

用法
----
    python -m loop.physical_static_diag --readout readout_real.npz \
        --meters-per-cell 0.40 --tof-port 8889 --odom-port 8888 \
        --seconds 30 --scene open --log static_open.csv --feats-npz static_open_feats.npz
"""
from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time

import numpy as np

from hardware.odometry_client import OdomClient, OdomError
from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import Tof5Client, TofError
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2d 静态领域诊断（只读，不动车）")
    ap.add_argument("--brain", choices=["fake", "real"], default="real")
    ap.add_argument("--mode", choices=["dn", "scripted", "readout"], default="readout")
    ap.add_argument("--readout", default="readout_real.npz")
    ap.add_argument("--brain-gain", type=float, default=1.0)
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--meters-per-cell", type=float, required=True)
    ap.add_argument("--scene", default="static")
    ap.add_argument("--seconds", type=float, default=30.0, help="0 = 一直采到 Ctrl+C")
    ap.add_argument("--tof-port", type=int, default=8889)
    ap.add_argument("--tof-warning-age", type=float, default=0.25)
    ap.add_argument("--tof-hard-stale", type=float, default=0.35)
    ap.add_argument("--tof-wait", type=float, default=5.0)
    ap.add_argument("--odom-port", type=int, default=8888, help="0 = 不读 odom")
    ap.add_argument("--odom-bind", default="0.0.0.0")
    ap.add_argument("--odom-wait", type=float, default=5.0)
    ap.add_argument("--log", default="static_diag.csv")
    ap.add_argument("--feats-npz", default="static_diag_feats.npz")
    args = ap.parse_args(argv)

    cfg, _maze, _car, brain, groups, _enc0, _ = make_setup(
        brain_kind=args.brain, map_name="track", device=args.device, seed=args.seed,
        sensory_input=False, make_decoder=False, verbose=True)
    cfg.decoder.mode = args.mode
    cfg.decoder.brain_gain = args.brain_gain
    cfg.decoder.memory_gain = 0.0
    if args.readout:
        cfg.decoder.readout_path = args.readout
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    if dec.trace is None or dec.readout is None:
        print("[static] 需要 --mode readout + 有效 --readout；退出")
        return 2
    trace, readout = dec.trace, dec.readout

    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=args.meters_per_cell)
    print(f"[static] ToF angles={ {n: DEFAULT_ANGLES_DEG[n] for n in ORDER} }  "
          f"meters_per_cell={args.meters_per_cell}  mode={args.mode}  scene={args.scene}")
    print("[static] **NO motor, NO bridge.** Ctrl+C 结束。")

    tof = Tof5Client(args.tof_port, warning_age=args.tof_warning_age,
                     hard_stale=args.tof_hard_stale, bind=args.odom_bind)
    tof.start()
    odom = None
    if args.odom_port:
        odom = OdomClient(args.odom_port, max_age=0.5, bind=args.odom_bind)
        odom.start()

    t0w = time.time()
    while tof.age() is None:
        if time.time() - t0w > args.tof_wait:
            print(f"[static] 等不到 ToF（UDP :{args.tof_port}）；退出")
            tof.stop()
            if odom:
                odom.stop()
            return 1
        time.sleep(0.05)
    print(f"[static] ToF :{args.tof_port} ok" + (f"   odom :{args.odom_port} ok" if odom else ""))

    cols = ["t", "tof_seq",
            "L_mm", "FL_mm", "F_mm", "FR_mm", "R_mm",
            "L_status", "FL_status", "F_status", "FR_status", "R_status",
            "L_cell", "FL_cell", "F_cell", "FR_cell", "R_cell",
            "dmin_L", "dmin_R", "dmin_F",
            "size_L", "size_R", "growth_L", "growth_R",
            "chase_L", "chase_R", "loom_L", "loom_R", "threat_L", "threat_R",
            "prox_L", "prox_R", "bias",
            "feat_l2", "feat_mean", "feat_std", "feat_min", "feat_max",
            "brain_turn_raw", "brain_turn_clipped", "DN_fired_count"]
    if odom:
        cols += ["odom_x", "odom_y", "odom_theta_deg"]

    fh = open(args.log, "w", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    writer.writerow(cols)

    feats_buf: list[np.ndarray] = []
    t_buf: list[float] = []
    seq_buf: list[int] = []
    state = {"stop": False}

    def _on_signal(signum, _f):
        state["stop"] = True

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _on_signal)
        except (ValueError, OSError):
            pass

    t0 = time.monotonic()
    next_t = t0
    n_rows = 0
    t_last = t0
    try:
        while not state["stop"]:
            if args.seconds and (time.monotonic() - t0) >= args.seconds:
                break
            frame, _age = tof.latest()             # stale/unhealthy -> TofError
            dists = enc.sense(frame)
            is_new = enc.update(frame, dists)
            inject, info = enc.inject()
            fired = brain.step(inject=inject)
            feats = np.asarray(trace.observe(fired), dtype=np.float32)
            raw = float(readout.predict(feats))
            clipped = float(np.clip(raw, -1.0, 1.0))

            if is_new:
                cl = enc._cache.get("L", (0.0, 0.0, 0.0, 0.0, float("nan")))
                cr = enc._cache.get("R", (0.0, 0.0, 0.0, 0.0, float("nan")))
                row = [f"{time.monotonic() - t0:.3f}", int(frame.seq)]
                for n in ORDER:
                    mm = frame.ranges.get(n)
                    row.append("" if mm is None else f"{mm:.0f}")
                for n in ORDER:
                    row.append(str(frame.status.get(n, "")))
                for i, n in enumerate(ORDER):
                    row.append(f"{float(dists[i]):.4f}")
                row += [f"{_f(info.get('dmin_L')):.4f}", f"{_f(info.get('dmin_R')):.4f}",
                        f"{_f(info.get('dmin_F')):.4f}",
                        f"{_f(info.get('size_L')):.4f}", f"{_f(info.get('size_R')):.4f}",
                        f"{_f(info.get('growth_L')):.4f}", f"{_f(info.get('growth_R')):.4f}",
                        f"{float(cl[1]):.4f}", f"{float(cr[1]):.4f}",
                        f"{float(cl[0]):.4f}", f"{float(cr[0]):.4f}",
                        f"{float(cl[2]):.4f}", f"{float(cr[2]):.4f}",
                        f"{_f(info.get('prox_l')):.4f}", f"{_f(info.get('prox_r')):.4f}",
                        f"{_f(info.get('bias')):.4f}",
                        f"{float(np.linalg.norm(feats)):.4f}", f"{float(feats.mean()):.5f}",
                        f"{float(feats.std()):.5f}", f"{float(feats.min()):.5f}",
                        f"{float(feats.max()):.5f}",
                        f"{raw:.4f}", f"{clipped:.4f}", int(len(fired))]
                if odom:
                    try:
                        s, _a = odom.latest()
                        row += [f"{s.x:.4f}", f"{s.y:.4f}", f"{math.degrees(s.theta):.1f}"]
                    except OdomError:
                        row += ["", "", ""]
                writer.writerow(row)
                fh.flush()
                feats_buf.append(feats.copy())
                t_buf.append(time.monotonic() - t0)
                seq_buf.append(int(frame.seq))
                n_rows += 1

            next_t += brain.dt
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)
            else:
                next_t = time.monotonic()
    except TofError as exc:
        print(f"\n[static] *** {type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        print("\n[static] 中断")
    finally:
        fh.close()
        if odom:
            odom.stop()
        tof.stop()
        if feats_buf:
            try:
                np.savez_compressed(args.feats_npz, feats=np.stack(feats_buf),
                                    t=np.asarray(t_buf), tof_seq=np.asarray(seq_buf),
                                    scene=np.asarray(args.scene),
                                    meters_per_cell=np.asarray(args.meters_per_cell))
                print(f"[static] feats {np.stack(feats_buf).shape} -> {args.feats_npz}")
            except OSError as exc:
                print(f"[static] 存 npz 失败: {exc}")

    print(f"[static] rows={n_rows}  log={args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
