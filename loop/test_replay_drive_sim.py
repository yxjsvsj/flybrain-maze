"""P2d-9 任务一回归：时间正确回放（合成 CSV，fake brain，无硬件）。"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import sys
import tempfile

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.replay_drive_sim import load_frames, run_replay
from loop.replay_drive_sim import main as replay_main
from loop.run import make_setup

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def _write_csv(path, times, fr_mm=None):
    cols = ["t"] + [f"{c}_mm" for c in ORDER] + [f"{c}_status" for c in ORDER]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for t in times:
            st = ["NO_TARGET"] * len(ORDER)
            mm = [""] * len(ORDER)
            if fr_mm is not None:
                st[ORDER.index("FR")] = "VALID"
                mm[ORDER.index("FR")] = str(fr_mm)
            w.writerow([f"{t:.4f}"] + mm + st)


def _setup():
    cfg, maze, car, brain, groups, _e, _ = make_setup(
        brain_kind="fake", map_name="gen", device="cpu", make_decoder=False, verbose=False)
    cfg.decoder.mode = "dn"
    cfg.decoder.memory_gain = 0.0
    dn = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=0.40)
    return cfg, maze, car, brain, enc, dec


def main() -> int:
    print("=========== replay_drive_sim (P2d-9) regression ===========")
    cfg, maze, car, brain, enc, dec = _setup()
    dt = brain.dt
    with tempfile.TemporaryDirectory() as tmp:
        # 10Hz over 1.0s
        p = os.path.join(tmp, "ten.csv")
        _write_csv(p, [round(0.1 * i, 3) for i in range(11)])
        frames = load_frames(p)
        check("load_frames 11 frames", len(frames) == 11)
        traj, s = run_replay(cfg, maze, car, brain, enc, dec, frames, memory_on=False)
        check(f"tof_frames={s['tof_frames']} == 11", s["tof_frames"] == 11)
        check(f"brain_steps={s['brain_steps']} ~ recorded/dt+1",
              abs(s["brain_steps"] - (round(1.0 / dt) + 1)) <= 1)
        check(f"recorded~1.0s ({s['recorded_duration_s']:.2f})", abs(s["recorded_duration_s"] - 1.0) < 0.05)
        check(f"sim~brain_steps*dt ({s['sim_duration_s']:.2f})",
              abs(s["sim_duration_s"] - s["brain_steps"] * dt) < 1e-9)
        check("uses cells key (path_len_cells)", "path_len_cells" in s and "path_len_m" not in s)
        frame_updates = sum(1 for r in traj if r["new_frame"])
        check(f"only 11 frame updates (got {frame_updates})", frame_updates == 11)

        # 不规则间隔 + 乱序 + 重复时间
        p2 = os.path.join(tmp, "irr.csv")
        _write_csv(p2, [0.5, 0.0, 0.13, 0.13, 0.14, 1.0])
        traj2, s2 = run_replay(cfg, maze, car, brain, enc, dec, load_frames(p2), memory_on=False)
        check("irregular/unsorted no crash", s2["brain_steps"] > 0)
        check(f"irregular recorded=1.0 ({s2['recorded_duration_s']:.2f})",
              abs(s2["recorded_duration_s"] - 1.0) < 0.05)

        # legacy per-frame：brain_steps == tof_frames（旧行为）
        traj3, s3 = run_replay(cfg, maze, car, brain, enc, dec, frames, memory_on=False, per_frame=True)
        check("per_frame legacy: steps == frames", s3["brain_steps"] == s3["tof_frames"])

        # main prints recorded/sim
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = replay_main(["--csv", p, "--brain", "fake", "--device", "cpu", "--mode", "dn", "--memory", "0"])
        out = buf.getvalue()
        check("main rc 0 + prints recorded/sim", rc == 0 and "recorded=" in out and "sim=" in out)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
