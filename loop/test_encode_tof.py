"""PhysicalToFEncoder 独立 dry-run（只诊断，不改生产驱动/控制逻辑/冻结 P1）。

用 Tof5Client（UDP 8889）收 Pi 的 ToF 帧 -> 喂**同一个** PhysicalToFEncoder 实例，
逐项提示摆白板。CSV 每采样一行，含五路 mm/status/cell + 神经侧量 + growth。

流程：CLEAR / L / FL / F / FR / R / F_APPROACH / F_STATIC
- 静态阶段：稳定窗口（末 2s）median 汇总；F_STATIC 只取末 2s
- F_APPROACH：基线 1s -> MOVE NOW -> 3s 平稳靠近 -> 保持 1s；输出 F_cell start/end/min、
  growth/loom/threat 峰值、新 ToF 帧数（**不用 median**）
- F_APPROACH 与 F_STATIC 复用同一 encoder 实例，不重置 prev_size/cache

验收语义（±60° baseline）：left={FL,F} right={FR,F} front={F}；L/R(±70°) 只测距/建图，
不进神经掩码。NO_TARGET->max_range，TOO_NEAR->min distance，IO_ERROR/unhealthy->FAIL。
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from types import SimpleNamespace

from config import Config, apply_preset
from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER, RangeState
from hardware.tof5_client import Tof5Client, TofError
from loop.encode_tof import PhysicalToFEncoder

STATIC_STEPS = [
    ("CLEAR", "移开所有障碍，五路都空"),
    ("L", "白板挡在**车左侧** L 正前方 ~300-500mm"),
    ("FL", "白板挡在**左前** FL"),
    ("F", "白板挡在**正前** F"),
    ("FR", "白板挡在**右前** FR"),
    ("R", "白板挡在**车右侧** R"),
]

COLS = (["stage", "t", "tof_seq", "tof_age_ms", "tof_dt", "new_tof_frame"]
        + [f"{n}_{k}" for n in ORDER for k in ("mm", "status", "cell")]
        + ["dmin_L", "dmin_R", "dmin_F", "size_L", "size_R",
           "growth_L", "growth_R", "loom_L", "loom_R",
           "chase_L", "chase_R", "threat_L", "threat_R", "bias"])


def _groups():
    keys = ["loom_L", "loom_R", "chase_L", "chase_R", "threat_L", "threat_R", "fwd"]
    return {k: [i] for i, k in enumerate(keys)}


def _mkframe(seq, t, f_mm):
    st = {n: RangeState.NO_TARGET.value for n in ORDER}
    rg: dict = {n: None for n in ORDER}
    st["F"] = RangeState.VALID.value
    rg["F"] = float(f_mm)
    return SimpleNamespace(session_id="synth", seq=seq, t=t, healthy=True,
                           ranges=rg, status=st, ages_ms={n: 0 for n in ORDER},
                           init_count=1, read_errors=0, reinit_count=0)


def gating_selftest(cfg) -> bool:
    """同一 tof_seq 连续 5 次 sense/update，growth/prev_size 只应更新一次。"""
    enc = PhysicalToFEncoder(_groups(), cfg.encoder, brain_dt=0.02,
                             angles_deg=DEFAULT_ANGLES_DEG,
                             car_max_speed=cfg.car.max_speed, car_radius=cfg.car.radius)
    enc.update(_mkframe(1, 0.0, 1200), enc.sense(_mkframe(1, 0.0, 1200)))
    f2 = _mkframe(2, 0.1, 600)
    enc.update(f2, enc.sense(f2))
    c1 = tuple(enc._cache["L"])
    for _ in range(4):
        enc.update(f2, enc.sense(f2))
    c5 = tuple(enc._cache["L"])
    ok = c1 == c5
    print(f"[门控自测] 同帧5次 cache {tuple(round(x,4) for x in c1)} -> "
          f"{tuple(round(x,4) for x in c5)}  {'PASS' if ok else 'FAIL'}")
    return ok


class Sampler:
    def __init__(self, enc, client, m_per_cell):
        self.enc, self.client, self.mpc = enc, client, m_per_cell
        self.rows: list[dict] = []
        self._size_prev = {}
        self._t_prev = {}
        self._last_seq = None
        self._last_t = None
        self._new_count = 0
        self._last_growth = {"L": 0.0, "R": 0.0}

    def poll(self, stage, t_rel):
        try:
            frame, age = self.client.latest()
        except TofError as exc:
            return None, type(exc).__name__
        dists = self.enc.sense(frame)
        is_new = self.enc.update(frame, dists)
        _inject, info = self.enc.inject()
        cl = self.enc._cache.get("L", (0, 0, 0, 0, 0))
        cr = self.enc._cache.get("R", (0, 0, 0, 0, 0))
        tof_dt = None if self._last_t is None else frame.t - self._last_t
        if is_new:
            self._new_count += 1
            for side, idx, size in (("L", 0, cl[3]), ("R", 1, cr[3])):
                if side in self._size_prev and tof_dt:
                    self._last_growth[side] = (size - self._size_prev[side]) / max(1e-3, tof_dt)
                self._size_prev[side] = size
            self._last_t = frame.t
            self._last_seq = frame.seq
        row = {"stage": stage, "t": round(t_rel, 3), "tof_seq": frame.seq,
               "tof_age_ms": round(age * 1000, 1),
               "tof_dt": ("" if tof_dt is None else round(tof_dt, 4)),
               "new_tof_frame": int(bool(is_new))}
        for i, n in enumerate(ORDER):
            row[f"{n}_mm"] = ("" if frame.ranges.get(n) is None else frame.ranges[n])
            row[f"{n}_status"] = frame.status.get(n, "?")
            row[f"{n}_cell"] = round(dists[i] / self.mpc, 4)
        row.update({"dmin_L": info.get("dmin_L"), "dmin_R": info.get("dmin_R"),
                    "dmin_F": info.get("dmin_F"),
                    "size_L": cl[3], "size_R": cr[3],
                    "growth_L": round(self._last_growth["L"], 4),
                    "growth_R": round(self._last_growth["R"], 4),
                    "loom_L": cl[0], "loom_R": cr[0],
                    "chase_L": cl[1], "chase_R": cr[1],
                    "threat_L": cl[2], "threat_R": cr[2],
                    "bias": info.get("bias")})
        self.rows.append(row)
        return row, None


def med(rows, key):
    xs = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(xs) if xs else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PhysicalToFEncoder dry-run")
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--meters-per-cell", type=float, required=True)
    ap.add_argument("--provisional", action="store_true")
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--preset", default="real")
    ap.add_argument("--csv", default="enc_dry.csv")
    args = ap.parse_args(argv)

    cfg = Config()
    try:
        apply_preset(cfg, args.preset)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[warn] apply_preset({args.preset}): {exc}")

    enc = PhysicalToFEncoder(_groups(), cfg.encoder, brain_dt=0.02,
                             angles_deg=DEFAULT_ANGLES_DEG,
                             car_max_speed=cfg.car.max_speed, car_radius=cfg.car.radius)
    print(f"[dry] left={[enc.names[i] for i in range(len(enc.names)) if enc.left[i]]} "
          f"right={[enc.names[i] for i in range(len(enc.names)) if enc.right[i]]} "
          f"front={[enc.names[i] for i in range(len(enc.names)) if enc.front[i]]}")
    print(f"[dry] meters_per_cell={args.meters_per_cell:g} "
          f"{'(provisional)' if args.provisional else ''}  preset={args.preset}")
    if not gating_selftest(cfg):
        print("门控自测失败 -> 退出")
        return 1

    client = Tof5Client(args.port)
    client.start()
    t0 = time.time()
    while client.age() is None:
        if time.time() - t0 > 5:
            print("[dry] 等不到 ToF UDP；退出"); client.stop(); return 1
        time.sleep(0.05)
    print("[dry] ToF UDP ok\n")

    smp = Sampler(enc, client, args.meters_per_cell)
    fail = False
    print("=" * 90)

    # ---- 静态阶段（稳定窗口 = 末 2s）----
    for name, desc in STATIC_STEPS:
        input(f">>> {name}: {desc}\n    摆好后按 Enter ... ")
        stage_rows = []
        t_end = time.time() + args.secs
        while time.time() < t_end:
            r, err = smp.poll(name, args.secs - (t_end - time.time()))
            if err:
                print(f"    !! {err}"); fail = fail or (err == "TofUnhealthyError")
            elif r:
                stage_rows.append(r)
            time.sleep(0.02)
        win = stage_rows[-int(2.0 / 0.02):] if len(stage_rows) > 1 else stage_rows
        print(f"    median(末2s): dmin L/R/F = {med(win,'dmin_L'):.3f}/{med(win,'dmin_R'):.3f}/"
              f"{med(win,'dmin_F'):.3f}  chase L/R = {med(win,'chase_L'):.2f}/{med(win,'chase_R'):.2f}  "
              f"L_cell={med(win,'L_cell'):.3f} R_cell={med(win,'R_cell'):.3f} "
              f"F_cell={med(win,'F_cell'):.3f}  bias={med(win,'bias'):+.3f}")

    # ---- F_APPROACH（专用流程，不用 median）----
    input(">>> F_APPROACH: 把白板放正前 **80~100cm**\n    按 Enter 开始（随后 1s 基线 -> 提示 MOVE NOW）... ")
    base_rows = []
    t_end = time.time() + 1.0
    while time.time() < t_end:
        r, err = smp.poll("F_APPROACH", 0.0)
        if r:
            base_rows.append(r)
        time.sleep(0.02)
    print("    ** MOVE NOW ** —— 3s 内把白板平稳靠近到 25~30cm，然后保持 1s")
    app_rows = []
    t_end = time.time() + 3.0
    while time.time() < t_end:
        r, _e = smp.poll("F_APPROACH", 1.0 + (3.0 - (t_end - time.time())))
        if r:
            app_rows.append(r)
        time.sleep(0.02)
    hold_rows = []
    t_end = time.time() + 1.0
    while time.time() < t_end:
        r, _e = smp.poll("F_APPROACH", 4.0 + (1.0 - (t_end - time.time())))
        if r:
            hold_rows.append(r)
        time.sleep(0.02)
    allapp = base_rows + app_rows + hold_rows
    fc = [r["F_cell"] for r in allapp if isinstance(r["F_cell"], (int, float))]
    mx = lambda rs, k: (max((r[k] for r in rs if isinstance(r[k], (int, float))), default=float("nan")))
    print(f"\n    F_APPROACH: F_cell start={fc[0] if fc else '?'} end={fc[-1] if fc else '?'} "
          f"min={min(fc) if fc else '?'}")
    print(f"      新 ToF 帧数={smp._new_count}  growth L/R max="
          f"{mx(allapp,'growth_L'):.3f}/{mx(allapp,'growth_R'):.3f}  "
          f"loom L/R max={mx(allapp,'loom_L'):.3f}/{mx(allapp,'loom_R'):.3f}  "
          f"threat L/R max={mx(allapp,'threat_L'):.3f}/{mx(allapp,'threat_R'):.3f}")

    # ---- F_STATIC（复用同一 encoder；只汇总末 2s）----
    input("\n>>> F_STATIC: 白板**停在 F 前不动**\n    按 Enter 采集 ... ")
    st_rows = []
    t_end = time.time() + args.secs
    while time.time() < t_end:
        r, _e = smp.poll("F_STATIC", args.secs - (t_end - time.time()))
        if r:
            st_rows.append(r)
        time.sleep(0.02)
    win = st_rows[-int(2.0 / 0.02):] if len(st_rows) > 1 else st_rows
    print(f"    median(末2s): dmin_F={med(win,'dmin_F'):.3f}  "
          f"growth L/R={med(win,'growth_L'):.3f}/{med(win,'growth_R'):.3f}  "
          f"loom L/R={med(win,'loom_L'):.3f}/{med(win,'loom_R'):.3f}  "
          f"chase L/R={med(win,'chase_L'):.2f}/{med(win,'chase_R'):.2f}")

    client.stop()

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in smp.rows:
            w.writerow({k: r.get(k, "") for k in COLS})
    print(f"\n保存 {len(smp.rows)} 行 -> {args.csv}")
    print(f"dry-run {'FAIL' if fail else 'OK'}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
