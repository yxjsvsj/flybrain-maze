"""Capture real 5-ToF frames into a D2 frame pack. No brain, no motor.

Usage:
  python -m demo.brain_turn_d2.capture --scene B_FL_only --distance-mm 400 \
      --frames 100 --out demo/brain_turn_d2/D2_FRAMES/B_FL_only_400mm.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from demo.brain_turn_d2.frames import ORDER


def capture(port: int, frames: int, hz: float, scene: str, distance_mm: int,
            out: str | Path, wait_s: float = 10.0) -> str:
    from hardware.tof5_client import Tof5Client

    client = Tof5Client(port=port, warning_age=0.25, hard_stale=0.35, bind="0.0.0.0")
    client.start()
    t0 = time.monotonic()
    try:
        while client.age() is None:
            if time.monotonic() - t0 > wait_s:
                raise TimeoutError("no ToF frame")
            time.sleep(0.05)
        rows_r, rows_s, ts, seqs = [], [], [], []
        last_seq = None
        while len(ts) < frames:
            fr, _age = client.latest()
            if last_seq is not None and fr.seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = fr.seq
            rows_r.append([np.nan if fr.ranges.get(n) is None else float(fr.ranges[n])
                           for n in ORDER])
            rows_s.append([str(fr.status.get(n, "")) for n in ORDER])
            ts.append(float(fr.t))
            seqs.append(int(fr.seq))
            time.sleep(1.0 / max(0.1, hz))
    finally:
        client.stop()
    if len(ts) < frames:
        raise RuntimeError(f"captured {len(ts)} / {frames}")
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p, ranges_mm=np.asarray(rows_r, np.float64), status=np.asarray(rows_s),
        t=np.asarray(ts, np.float64), seq=np.asarray(seqs, np.int64),
        scene=np.asarray(scene), distance_mm=np.asarray(int(distance_mm)),
    )
    return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D2 real-ToF frame capture (no brain, no motor)")
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--scene", required=True,
                    choices=["A_open", "B_FL_only", "C_FR_only", "D_FL_FR_symmetric"])
    ap.add_argument("--distance-mm", type=int, choices=[400, 700], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait-s", type=float, default=10.0)
    args = ap.parse_args(argv)
    p = capture(args.port, args.frames, args.hz, args.scene, args.distance_mm,
                args.out, args.wait_s)
    print(f"[d2-capture] wrote {p}")
    print("[d2-capture] NO brain, NO motor, NO training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
