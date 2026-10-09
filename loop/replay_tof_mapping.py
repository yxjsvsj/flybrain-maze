"""OFF/ON 建图离线回放：同一份录制数据，旧（车中心射线） vs 新（逐原点射线）建图差异。

读一份含 `odom_x, odom_y, theta_deg` + `tof_{L,FL,F,FR,R}_mm` + `tof_*_st` 的 CSV
（例如 `loop/straight_run.py` 的 log），分别喂给：

  旧: 冻结 `OccupancyMemory.update(x, y, car_theta + sensor_angles, dists)`
      —— 所有射线从**车中心**、用 ToF 距离（cells）。
  新: `ToFMemory.update_rays(...)` + `loop.tof_geometry.frame_to_cell_rays`
      —— 每路从**真实传感器光学窗口**。

输出：
  * 两张占用图的差异（新图多出/少掉了哪些格）；
  * 指定传感器（默认 F）在两张图里各自落在哪个格 + 对应**世界坐标(米)**，
    便于和现场卷尺实测位置比对（差多少 cm）。

**纯离线，不连硬件、不动车。**
"""
from __future__ import annotations

import argparse
import csv
import math
import sys

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.tof_geometry import frame_to_cell_rays, load_geometry, rotate
from loop.memory_tof import ToFMemory
from nav.memory import OCCUPIED, OccupancyMemory

MIN_DIST_M = 0.05


class _Frame:
    def __init__(self, ranges, status, seq):
        self.ranges = dict(ranges)
        self.status = dict(status)
        self.seq = seq


def _mem(cls, w, h, max_range):
    return cls(w, h, max_range, unknown_cost=4.0, lookahead=0.9,
               arrive_dist=0.6, off_path_tol=1.5, free_conflict_threshold=3)


def replay(path, geometry, mpc, watch="F", w=60, h=60):
    geo = load_geometry(geometry)
    mem_old = _mem(OccupancyMemory, w, h, 6.0)
    mem_new = _mem(ToFMemory, w, h, 6.0)
    angles = {n: math.radians(DEFAULT_ANGLES_DEG[n]) for n in ORDER}
    n_rows = 0
    old_hit = new_hit = None
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                x = float(r["odom_x"]); y = float(r["odom_y"])
                th = math.radians(float(r["theta_deg"]))
            except (KeyError, ValueError):
                continue
            ranges, status = {}, {}
            for n in ORDER:
                mm = r.get(f"tof_{n}_mm", "")
                ranges[n] = float(mm) if mm not in ("", None) else None
                status[n] = r.get(f"tof_{n}_st", "")
            fr = _Frame(ranges, status, n_rows + 1)

            # --- OLD: 车中心射线 + ToF 距离(cells)，只取 VALID/TOO_NEAR ---
            idx, dists = [], []
            for i, n in enumerate(ORDER):
                st = status[n]
                if st == "VALID" and ranges[n] is not None:
                    idx.append(i); dists.append(max(MIN_DIST_M / mpc, (ranges[n] / 1000.0) / mpc))
                elif st == "TOO_NEAR":
                    idx.append(i); dists.append(MIN_DIST_M / mpc)
            if idx:
                import numpy as np
                a = np.array([th + angles[ORDER[i]] for i in idx], dtype=float)
                d = np.array(dists, dtype=float)
                mem_old.update(x, y, a, d)

            # --- NEW: 逐原点射线 ---
            rays = frame_to_cell_rays(fr, x, y, th, geo, mpc)
            mem_new.update_rays(x, y, rays, tof_seq=fr.seq)

            if watch in status and status[watch] == "VALID" and ranges[watch] is not None:
                rng_m = ranges[watch] / 1000.0
                ext = geo[watch]
                dx, dy = rotate(ext.x_m, ext.y_m, th)
                wa = th + ext.yaw_rad
                o_hx = x * mpc + rng_m * math.cos(wa)          # old: from car centre
                o_hy = y * mpc + rng_m * math.sin(wa)
                n_hx = (x + dx / mpc) * mpc + rng_m * math.cos(wa)
                n_hy = (y + dy / mpc) * mpc + rng_m * math.sin(wa)
                old_hit = (int(math.floor(o_hx / mpc)), int(math.floor(o_hy / mpc)), o_hx, o_hy)
                new_hit = (int(math.floor(n_hx / mpc)), int(math.floor(n_hy / mpc)), n_hx, n_hy)
            n_rows += 1

    old_occ = set(zip(*[a.tolist() for a in (mem_old.known == OCCUPIED).nonzero()]))
    new_occ = set(zip(*[a.tolist() for a in (mem_new.known == OCCUPIED).nonzero()]))
    only_new = sorted(new_occ - old_occ)
    only_old = sorted(old_occ - new_occ)
    print(f"[replay] rows={n_rows}  old_occupied={len(old_occ)}  new_occupied={len(new_occ)}")
    print(f"[replay] only-in-NEW (first 20): {only_new[:20]}")
    print(f"[replay] only-in-OLD (first 20): {only_old[:20]}")
    if old_hit and new_hit:
        print(f"[replay] {watch} last VALID hit:")
        print(f"    OLD cell=({old_hit[0]},{old_hit[1]})  world=({old_hit[2]:.3f},{old_hit[3]:.3f}) m")
        print(f"    NEW cell=({new_hit[0]},{new_hit[1]})  world=({new_hit[2]:.3f},{new_hit[3]:.3f}) m")
        dxm = new_hit[2] - old_hit[2]; dym = new_hit[3] - old_hit[3]
        print(f"    NEW-OLD delta = ({dxm*100:+.1f}, {dym*100:+.1f}) cm  "
              f"(|d|={math.hypot(dxm, dym)*100:.1f} cm)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OFF/ON 建图离线回放（不动车）")
    ap.add_argument("csv_path")
    ap.add_argument("--tof-geometry", default="hardware/tof_geometry.json")
    ap.add_argument("--meters-per-cell", type=float, default=0.40)
    ap.add_argument("--watch", default="F", choices=list(ORDER))
    args = ap.parse_args(argv)
    return replay(args.csv_path, args.tof_geometry, args.meters_per_cell, args.watch)


if __name__ == "__main__":
    sys.exit(main())
