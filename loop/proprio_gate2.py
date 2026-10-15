"""P2d-24 第二关：向第一关确认的自运动入口做**开环**神经实验（不改权重/不控车/无回正答案）。

第一关结论（见 PROPRIO_REVIEW.md）：连接组里
  - 无 haltere/前庭、无 chordotonal/campaniform/proprio 标签、无 efference/corollary 标签；
  - 有 **T4/T5 视觉运动(optic-flow)检测器**（13584，2跳→DN 927/motor 107），但**地面轮式车无相机/光流**；
  - 有 **vnc_sensory**（VNC 体/腿机械感觉，6370，2跳→DN 1314/motor 803）——**轮式车的轮编码器**是其自然类比。
→ 唯一"车上可获得 + 连接组可定位 + 可达运动回路"的入口 = vnc_sensory（其"本体感觉"身份为**推断**）。

本关只把**自运动(yaw)符号**输入 vnc_sensory 的左右两组，检查下游与 L50/R50 是否被真实传播。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.encode_tof import PhysicalToFEncoder
from loop.population_turn import L_IDX, R_IDX, TAU
from loop.run import make_setup

K_PROP = 0.5          # 注入幅度（人工约定，披露；不按迷宫成功率搜索）


def make_A_frame(seq, t):
    from loop.dn_dynamics_probe import STIMS, make_frame
    return make_frame(STIMS["A"], seq, t)


def run(seed, yaw_series, *, seconds=10.0, tof_hz=10.0, mpc=0.40, cfg, brain, groups, dn_idx,
        vL, vR, tau=TAU):
    from flybrain import Trace
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    dn_set = set(np.asarray(dn_idx).tolist())
    vL, vR = np.asarray(vL), np.asarray(vR)
    tr = Trace(brain, idx=np.asarray(dn_idx)[np.array(list(L_IDX) + list(R_IDX), int)], tau=tau)
    brain.reset(seed)
    dt = brain.dt
    period = 1.0 / tof_hz
    n = int(round(seconds / dt))
    frame, seq, last_t = None, 0, -1e9
    dn_spk, D = [], []
    for k in range(n):
        t = k * dt
        if frame is None or (t - last_t) >= period - 1e-9:
            seq += 1
            frame = make_A_frame(seq, t)
            last_t = t
        d = enc.sense(frame)
        enc.update(frame, d)
        inj, _info = enc.inject()
        y = float(yaw_series[k]) if k < len(yaw_series) else 0.0
        if y != 0.0 and len(vL):
            inj = list(inj) + [(vL, K_PROP * y), (vR, K_PROP * (-y))]
        fired = brain.step(inject=inj)
        dn_spk.append(int(np.isin(fired, np.asarray(dn_idx)).sum()) if len(fired) else 0)
        a = np.asarray(tr.observe(fired), np.float32)
        D.append(float(a[:len(L_IDX)].mean() - a[len(L_IDX):].mean()))
    return np.array(dn_spk, float), np.array(D, float), dt


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-seeds", default="64,101,202,303")
    ap.add_argument("--new-seeds", default="1601,2309")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args(argv)
    screen = [int(s) for s in args.screen_seeds.split(",")]
    new = [int(s) for s in args.new_seeds.split(",")]

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=args.device, seed=screen[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    vL = brain.cells(["vnc_sensory"], side="L")
    vR = brain.cells(["vnc_sensory"], side="R")
    print(f"[P2d-24 gate2] 入口=vnc_sensory  L={len(vL)} R={len(vR)}  注入 K={K_PROP}（人工符号约定，"
          "含义为 yaw 的左右腿本体感觉不对称；只查连接组是否传播，不判'正确'）")
    dt = brain.dt
    n = int(round(args.seconds / dt))
    print(f"\n{'seed':>6s} {'cond':>6s} {'dn_spk/步':>9s} {'<D>':>8s} {'|D|':>7s} {'vs0 的 dnΔ':>10s} {'vs0 的 DΔ':>9s}")
    rep = []
    for grp, seeds in (("scr", screen), ("new", new)):
        for sd in seeds:
            conds = {"zero": np.zeros(n), "yaw+": np.ones(n), "yaw-": -np.ones(n),
                     "alt": np.array([(1 if (k // int(1 / dt)) % 2 == 0 else -1) for k in range(n)], float)}
            base_dn = base_D = None
            for name, ys in conds.items():
                dns, D, _dt = run(sd, ys, seconds=args.seconds, cfg=cfg, brain=brain, groups=groups,
                                  dn_idx=dn_idx, vL=vL, vR=vR)
                if name == "zero":
                    base_dn, base_D = float(dns.mean()), float(D.mean())
                dd = float(dns.mean()) - (base_dn if base_dn is not None else 0.0)
                dD = float(D.mean()) - (base_D if base_D is not None else 0.0)
                print(f"{sd:6d} {name:>6s} {dns.mean():9.2f} {D.mean():+8.3f} {np.abs(D).mean():7.3f} "
                      f"{dd:+10.2f} {dD:+9.4f}")
                rep.append((grp, sd, name, dd, dD))
    # 一致性：yaw+ 与 yaw- 的 DΔ 是否反号且跨 seed 一致
    print("\n[一致性] 各 seed 的 (yaw+ DΔ, yaw- DΔ)（应反号=连接组传播了符号）与 dnΔ 同号性")
    for sd in screen + new:
        pp = [r for r in rep if r[1] == sd and r[2] == "yaw+"]
        pm = [r for r in rep if r[1] == sd and r[2] == "yaw-"]
        if pp and pm:
            ddp, ddm = pp[0][4], pm[0][4]
            dnp, dnm = pp[0][3], pm[0][3]
            print(f"  seed {sd:5d}: DΔ+={ddp:+.4f} DΔ-={ddm:+.4f} 反号={np.sign(ddp) == -np.sign(ddm)} | "
                  f"dnΔ+={dnp:+.2f} dnΔ-={dnm:+.2f}")
    print("\n[边界] 只证明符号是否经连接组传播；不控车、不含回正/地图/终点，不改权重。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
