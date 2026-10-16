"""P2d-24a 第二关：**连续时序**开环自运动注入（不重置、逐段统计；只复现，不重挑细胞/不调幅度）。

固定入口 vnc_sensory(L/R)、K=0.5（人工双侧 ±电压偏置，**非已标定生理 yaw**）。
序列内 brain/Encoder/Trace **连续**，切换处**不重置**；另设同 seed/同脑年龄/同 BASE 感觉的**全零反馈**对照。
各段时长**运行前声明**（见 PLANS）。逐段报告：实际注入、输入池发放、DN 发放、D(t)、相对对照变化、
响应延迟、撤除后残余。1601/2309 标记为**历史已用**；新 seed 先声明（须不在历史表内）。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.encode_tof import PhysicalToFEncoder
from loop.population_turn import L_IDX, R_IDX, TAU
from loop.run import make_setup

K_PROP = 0.5
HISTORICAL = {64, 101, 202, 303, 512, 606, 777, 808, 999, 1001, 1103, 1201, 1301, 1409, 1511,
              1601, 1709, 1801, 1907, 2003, 2111, 2203, 2309}
NEW_SEEDS = (2701,)                       # 运行前声明（须∉HISTORICAL，见 main 校验）
# 各序列的 (状态, 时长s) —— 运行前声明
PLANS = {
    "zpz":   [("z", 1.5), ("+", 2.0), ("z", 3.0)],
    "znz":   [("z", 1.5), ("-", 2.0), ("z", 3.0)],
    "zpznz": [("z", 1.5), ("+", 2.0), ("z", 2.0), ("-", 2.0), ("z", 3.0)],
}
CTRL = {k: [(n, d) for n, d in v] for k, v in PLANS.items()}      # 同分段，全零反馈
YSIGN = {"z": 0.0, "+": 1.0, "-": -1.0}


def segments(plan, dt):
    out, acc = [], 0
    for name, dur in plan:
        m = int(round(dur / dt))
        out.append((name, acc, acc + m))
        acc += m
    return out


def prop_inj(name, feedback, vL, vR, k=K_PROP):
    """返回 (额外注入列表, 注入电压总量)。zero 状态或 feedback=False -> ([], 0)。"""
    y = YSIGN[name] if feedback else 0.0
    if y == 0.0 or len(vL) == 0:
        return [], 0.0
    return [(vL, k * y), (vR, k * (-y))], k * abs(y) * (len(vL) + len(vR))


def run_seq(seed, plan, feedback, *, cfg, brain, groups, dn_idx, vL, vR, tau=TAU, mpc=0.40):
    """连续运行整条序列（不重置）；记录注入、输入池发放、DN 发放、D。"""
    from flybrain import Trace
    from loop.dn_dynamics_probe import STIMS, make_frame
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    vL, vR = np.asarray(vL), np.asarray(vR)
    tr = Trace(brain, idx=np.asarray(dn_idx)[np.array(list(L_IDX) + list(R_IDX), int)], tau=tau)
    brain.reset(seed)
    dt = brain.dt
    period = 1.0 / 10.0
    frame, seq, last_t = None, 0, -1e9
    inj_v, spk_v, spk_dn, D, states = [], [], [], [], []
    for name, i0, i1 in segments(plan, dt):
        for k in range(i0, i1):
            t = k * dt
            if frame is None or (t - last_t) >= period - 1e-9:
                seq += 1
                frame = make_frame(STIMS["A"], seq, t)
                last_t = t
            d = enc.sense(frame)
            enc.update(frame, d)
            inj, _info = enc.inject()
            extra, iv = prop_inj(name, feedback, vL, vR)
            if extra:
                inj = list(inj) + extra
            fired = brain.step(inject=inj)
            inj_v.append(iv)
            spk_v.append(int(np.isin(fired, np.concatenate([vL, vR])).sum()) if len(fired) else 0)
            spk_dn.append(int(np.isin(fired, np.asarray(dn_idx)).sum()) if len(fired) else 0)
            a = np.asarray(tr.observe(fired), np.float32)
            D.append(float(a[:len(L_IDX)].mean() - a[len(L_IDX):].mean()))
            states.append(name)
    return (np.array(inj_v, float), np.array(spk_v, float), np.array(spk_dn, float), np.array(D, float),
            np.array(states), dt)


def seg_mean(x, i0, i1, dt, last=0.5):
    w = max(1, int(round(last / dt)))
    return float(np.mean(x[max(i0, i1 - w):i1]))


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-seeds", default="64,101,202,303")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    screen = [int(s) for s in args.screen_seeds.split(",")]
    for s in NEW_SEEDS:
        assert s not in HISTORICAL, f"新 seed {s} 已在历史表内，必须替换"
    print(f"[P2d-24a] 入口=vnc_sensory L/R, K={K_PROP}（人工双侧 ±电压偏置，非生理 yaw）")
    print(f"  序列(运行前声明): " + "; ".join(f"{k}={v}" for k, v in PLANS.items()))
    print(f"  screen={screen}  声明新 seed={NEW_SEEDS}（已核对∉历史表 {len(HISTORICAL)} 个）"
          f"  历史已用含 1601/2309")

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=args.device, seed=screen[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    vL = brain.cells(["vnc_sensory"], side="L")
    vR = brain.cells(["vnc_sensory"], side="R")

    for sd in screen + list(NEW_SEEDS):
        for key, plan in PLANS.items():
            iv, sv, sdn, D, st, dt = run_seq(sd, plan, True, cfg=cfg, brain=brain, groups=groups,
                                             dn_idx=dn_idx, vL=vL, vR=vR)
            _i2, _s2, _d2, Dc, _st2, _ = run_seq(sd, plan, False, cfg=cfg, brain=brain,
                                                 groups=groups, dn_idx=dn_idx, vL=vL, vR=vR)
            print(f"\n=== seed {sd} / 序列 {key} ({plan}) ===")
            print(f"  {'seg':>4s} {'injV/步':>9s} {'vncSpk':>7s} {'dnSpk':>6s} {'D(stim)':>8s} "
                  f"{'D(ctrl)':>8s} {'ΔD':>8s} {'lat_s':>6s}")
            prev = None
            for idx, (name, i0, i1) in enumerate(segments(plan, dt)):
                di, dv, dd = seg_mean(iv, i0, i1, dt), seg_mean(sv, i0, i1, dt), seg_mean(sdn, i0, i1, dt)
                ds, dc = seg_mean(D, i0, i1, dt), seg_mean(Dc, i0, i1, dt)
                # 响应延迟：切换后 |D-Dctrl| 首超 0.02
                lat = -1.0
                if idx > 0:
                    dev = np.abs(D[i0:i1] - Dc[i0:i1])
                    j = next((q for q in range(len(dev)) if dev[q] > 0.02), None)
                    lat = j * dt if j is not None else -1.0
                print(f"  {name:>4s} {di:9.1f} {dv:7.1f} {dd:6.1f} {ds:+8.3f} {dc:+8.3f} "
                      f"{ds - dc:+8.3f} {lat:6.2f}")
            # 撤除后残余 = 最后 z 段 ΔD
            last = segments(plan, dt)[-1]
            res = seg_mean(D, last[1], last[2], dt) - seg_mean(Dc, last[1], last[2], dt)
            print(f"  [撤除后残余 ΔD(末段 z)] = {res:+.4f}")
    print("\n[边界] 只复核连续时序下的动态响应与撤除行为；不接 P2Control/不控车/无回正/无 Dijkstra/无老师。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
