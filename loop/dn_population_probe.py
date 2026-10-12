"""P2d-13 任务二：下行神经元群体对受控 5 路 ToF 刺激的响应分析（离线，不训练）。

固定 MaleCNS + 现有感觉编码 + 五路 ToF，离线注入五种静态刺激，记录下行神经元活动：
  A 无遮挡   B FR-only 近障   C FL-only 近障   D 正前 F-only   E 左右对称近障
在 brain seed 64/101/202/303 下检查：每种刺激相对 A 的变化、左右反应方向及跨 seed 一致性、
时间稳定性、是否有比单个 DNa02 更稳定的群体信号、对称前方是否有可重复方向证据。

**只报告相关性，不据此把某个神经元命名为"左转/右转神经元"。** 区分：
  - 生物学注释（FlyWire/MaleCNS 细胞类型：DNa02 转向、DNg100 前进、MDN 后退、DNp01 逃逸）
  - 本项目实验性推断（本脚本的差分/符号，均属**人工**映射，不冒充果蝇天然运动解码）

用法： python -m loop.dn_population_probe --seeds 64,101,202,303
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import TofFrame
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup

STIMULI = {
    "A_none": {},
    "B_FR":   {"FR": 400.0},
    "C_FL":   {"FL": 400.0},
    "D_F":    {"F": 400.0},
    "E_sym":  {"FL": 400.0, "FR": 400.0},
}
GROUP_NAMES = ("turn_L", "turn_R", "fwd", "back", "escape")


def make_frame(stim, seq, t):
    ranges = {n: None for n in ORDER}
    status = {n: "NO_TARGET" for n in ORDER}
    for n, mm in stim.items():
        ranges[n], status[n] = float(mm), "VALID"
    return TofFrame(session_id="probe", seq=seq, t=t, healthy=True, ranges=ranges, status=status,
                    ages_ms={n: 100.0 for n in ORDER}, init_count=0, read_errors=0, reinit_count=0)


def run_stim(cfg, brain, groups, enc, dec, trace, stim, *, seconds=3.0, settle=1.5):
    """稳态响应：组发放率(Hz) + 每个下行神经元的痕迹活动(时间均值)。"""
    frame = make_frame(stim, 1, 0.0)
    n = int(round(seconds / brain.dt))
    k0 = int(round(settle / brain.dt))
    g_rate = {name: [] for name in GROUP_NAMES}
    acts = []
    for k in range(n):
        d = enc.sense(frame)
        enc.update(frame, d)
        inj, info = enc.inject()
        fired = brain.step(inject=inj)
        dec.update(fired, info)
        acts.append(trace.observe(fired))
        if k >= k0:
            for name in GROUP_NAMES:
                g_rate[name].append(float(dec.rates[dec.index[name]]))
    act = np.mean(np.asarray(acts[k0:], np.float32), 0) if acts else np.zeros(0, np.float32)
    return {name: float(np.mean(v)) if v else 0.0 for name, v in g_rate.items()}, act


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="64,101,202,303")
    ap.add_argument("--brain", default="real")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--settle", type=float, default=1.5)
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]

    from flybrain import Trace
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=seeds[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    print(f"[probe] DNs={len(dn_idx)}  seeds={seeds}  seconds={args.seconds} settle={args.settle}")
    print("[annot] 生物学注释: turn=DNa02(L/R)  fwd=DNg100  back=MDN  escape=DNp01")
    print("[annot] 下方 diff=turn_L-turn_R 等均为**实验性推断**，非果蝇天然解码\n")

    gri = {}                                     # {seed: {stim: {group: rate}}}
    gact = {}                                    # {seed: {stim: act_vector}}
    for sd in seeds:
        enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                                 sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                                 car_radius=cfg.car.radius, meters_per_cell=0.40)
        dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
        trace = Trace(brain, idx=dn_idx, tau=0.2)
        brain.reset(sd)
        gri[sd], gact[sd] = {}, {}
        for name, stim in STIMULI.items():
            r, act = run_stim(cfg, brain, groups, enc, dec, trace, stim,
                              seconds=args.seconds, settle=args.settle)
            gri[sd][name], gact[sd][name] = r, act

    # ---- 表1：注释组的稳态发放率(Hz) ----
    print("表1 注释下行组稳态发放率(Hz)")
    print(f"{'seed':>5s} {'stim':7s} {'turn_L':>7s} {'turn_R':>7s} {'diff(L-R)':>9s} "
          f"{'fwd':>6s} {'back':>6s} {'escape':>7s}")
    for sd in seeds:
        for name in STIMULI:
            r = gri[sd][name]
            print(f"{sd:5d} {name:7s} {r['turn_L']:7.3f} {r['turn_R']:7.3f} "
                  f"{r['turn_L'] - r['turn_R']:9.3f} {r['fwd']:6.3f} {r['back']:6.3f} {r['escape']:7.3f}")

    # ---- 表2：跨 seed 方向一致性（diff = turn_L - turn_R）----
    print("\n表2 跨 seed 方向一致性  diff=turn_L-turn_R（DNa02）")
    print(f"{'stim':7s} " + " ".join(f"{sd:>8d}" for sd in seeds) + f" {'mean±std':>14s} {'sign一致':>8s}")
    for name in STIMULI:
        ds = [gri[sd][name]["turn_L"] - gri[sd][name]["turn_R"] for sd in seeds]
        signs = [np.sign(d) for d in ds]
        consistent = all(s == signs[0] for s in signs) and signs[0] != 0
        print(f"{name:7s} " + " ".join(f"{d:+8.3f}" for d in ds)
              + f" {np.mean(ds):+8.3f}±{np.std(ds):.3f} {str(consistent):>8s}")

    # ---- 表3：FR vs FL（DNa02 左右反应方向）----
    print("\n表3 DNa02 对单侧刺激：B(FR) 应使 turn_R↑、C(FL) 应使 turn_L↑？（差异方向）")
    for sd in seeds:
        b, c = gri[sd]["B_FR"], gri[sd]["C_FL"]
        print(f"  seed {sd:3d}: B_FR diff={b['turn_L'] - b['turn_R']:+.3f}  "
              f"C_FL diff={c['turn_L'] - c['turn_R']:+.3f}  "
              f"B/C 反号={np.sign(b['turn_L'] - b['turn_R']) != np.sign(c['turn_L'] - c['turn_R'])}")

    # ---- 表4：群体扫描（哪些下行神经元对 (C-B) 有跨 seed 一致符号）----
    print("\n表4 群体扫描：对左右刺激差 (C_FL - B_FR) 在全部 seed 同号的下行神经元")
    n_dn = len(dn_idx)
    dmat = np.stack([gact[sd]["C_FL"] - gact[sd]["B_FR"] for sd in seeds], 0)   # (nseed, n_dn)
    same = np.all(np.sign(dmat) == np.sign(dmat[0]), axis=0) & (np.abs(dmat[0]) > 1e-6)
    mag = np.abs(dmat.mean(0))
    order = np.argsort(-mag)
    n_same = int(same.sum())
    print(f"  同号且非零的下行神经元: {n_same}/{n_dn}")
    if n_same:
        print(f"  {'dn_i':>6s} {'mean(C-B)':>10s} {'std':>8s} {'mean|act|A':>11s}")
        shown = 0
        for i in order:
            if same[i] and shown < 10:
                a_act = np.mean([np.abs(gact[sd]["A_none"][i]) for sd in seeds])
                print(f"  {int(dn_idx[i]):6d} {dmat.mean(0)[i]:+10.4f} {dmat[:, i].std():8.4f} "
                      f"{a_act:11.4f}")
                shown += 1
    else:
        print("  （无：左右刺激差在不同 seed 间符号不一致）")

    # ---- 表5：对称前方 E / F-only D 是否出现可重复方向证据 ----
    print("\n表5 对称/正前刺激的方向证据（理想应为 0；方向证据 = 跨 seed 同号且非零）")
    for name in ("D_F", "E_sym"):
        ds = [gri[sd][name]["turn_L"] - gri[sd][name]["turn_R"] for sd in seeds]
        signs = [np.sign(d) for d in ds]
        rep = all(s == signs[0] for s in signs) and signs[0] != 0
        print(f"  {name:7s} diffs=" + " ".join(f"{d:+.3f}" for d in ds)
              + f"  可重复方向={rep}  均值{np.mean(ds):+.3f}±{np.std(ds):.3f}")

    # ---- 表6：时间稳定性（稳态窗内 diff 的变异系数，跨 seed 平均）----
    print("\n表6 时间稳定（本探针用稳态均值；下方为跨 seed 的 diff 标准差，越小越稳）")
    for name in STIMULI:
        ds = np.array([gri[sd][name]["turn_L"] - gri[sd][name]["turn_R"] for sd in seeds])
        print(f"  {name:7s} diff std 跨seed={ds.std():.4f}  |mean|={abs(ds.mean()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
