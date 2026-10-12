"""P2d-13d：正前方 F 感觉分组的单变量消融（BASE vs TEST，严格离线）。

BASE = 现有 PhysicalToFEncoder（左=FL+F，右=FR+F，front=F）。
TEST = P2 适配层 SideOnlyToFEncoder（左=FL，右=FR，front=F）——**只改 P2 感觉输入分组**。
其余全部一致：ToF 原始值、外参、神经元类型、脑权重、DNa02 读出、b0 标定、EMA tau=0.5s、
增益、速度、低通、front_safety；memory_gain=0；10Hz ToF / 50Hz 脑。

用法： python -m loop.f_grouping_ablation --seeds 64,101,202,303,512
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from loop.dn_population_probe import STIMULI, run_trial
from loop.dn_steering import calibrate_baseline
from loop.local_avoidance_bench import SCENARIOS
from loop.p2_encoder_variants import encoder_cls
from loop.run import make_setup


def neural_compare(cfg, brain, groups, dn, seeds, *, seconds=3.0, settle=1.5):
    """任务3：BASE/TEST 下各刺激的 DNa02 左右发放差（含相对 A 的差分）。"""
    from flybrain import Trace
    out = {}
    for var in ("BASE", "TEST"):
        ec = encoder_cls(var)
        out[var] = {}
        for sd in seeds:
            res = {name: run_trial(cfg, brain, groups, dn, Trace, stim, seed=sd,
                                   seconds=seconds, settle=settle, enc_cls=ec)
                   for name, stim in STIMULI.items()}
            out[var][sd] = {name: res[name]["rates"]["turn_L"] - res[name]["rates"]["turn_R"]
                            for name in STIMULI}
    return out


def dynamic(cfg, brain, groups, dn, seeds, scenarios, *, seconds=40.0, rig=None):
    """任务4：BASE/TEST 的动态配对（DN_readout；TEST 另做 DN_zero 依赖检查）。"""
    from loop.dn_steering import DNReadout
    from loop.dn_steering_diag import run_trace
    out = {}
    for var in ("BASE", "TEST"):
        ec = encoder_cls(var)
        out[var] = {}
        b0 = {sd: calibrate_baseline(cfg, brain, groups, dn, sd, enc_cls=ec) for sd in seeds}
        for sd in seeds:
            for sc in scenarios:
                for kind in ("DN_readout",) if var == "BASE" else ("DN_readout", "DN_zero"):
                    s, recs = run_trace(cfg, brain, groups, dn, sd, kind, SCENARIOS[sc],
                                        b0=b0[sd], seconds=seconds, rig=rig, enc_cls=ec)
                    d = next((i for i, r in enumerate(recs) if r["dmin_F"] < 1.2), None)
                    w = recs[d:d + int(round(1.5 / brain.dt))] if d is not None else []
                    turns = np.array([r["turn"] for r in w]) if w else np.zeros(0)
                    out[var][(sd, sc, kind)] = {
                        "reach": s["reach"], "t_goal": (recs[-1]["t"] if s["reach"] else -1.0),
                        "cev": s["cev"], "front_events": s["front_events"], "bestD": s["bestD"],
                        "minGap": s["minGap"], "sat%": s["sat%"], "stop%": s["stop%"],
                        "stuck": s["stuck"], "b0": b0[sd],
                        "win_meanturn": float(turns.mean()) if len(turns) else 0.0,
                        "win_sat%": float(np.mean(np.abs(turns) >= 0.999)) * 100 if len(turns) else 0.0}
    return out


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="64,101,202,303,512")
    ap.add_argument("--scenarios", default="must_left,must_right")
    ap.add_argument("--brain", default="real")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--skip-neural", action="store_true")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    scenarios = args.scenarios.split(",")

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=seeds[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn = np.asarray(brain.cells(["descending_neuron"]))

    # ---- 任务3：受控神经比较 ----
    if not args.skip_neural:
        print("任务3 DNa02 左右发放差 (turn_L-turn_R)，BASE vs TEST")
        nz = neural_compare(cfg, brain, groups, dn, seeds)
        print(f"{'seed':>5s} {'stim':7s} {'BASE':>9s} {'TEST':>9s} {'Δ(TEST-BASE)':>13s}")
        for sd in seeds:
            for name in STIMULI:
                b, t = nz["BASE"][sd][name], nz["TEST"][sd][name]
                print(f"{sd:5d} {name:7s} {b:+9.3f} {t:+9.3f} {t - b:+13.3f}")
        print("\n方向一致性（FR-only -> diff<0，FL-only -> diff>0；seed202 为已知例外）：")
        for var in ("BASE", "TEST"):
            ok = all((nz[var][sd]["B_FR"] < 0) and (nz[var][sd]["C_FL"] > 0) for sd in seeds)
            print(f"  {var:4s}: B>0 & C<0 全seed={ok}  "
                  + " ".join(f"{sd}:B{nz[var][sd]['B_FR']:+.2f}/C{nz[var][sd]['C_FL']:+.2f}"
                             for sd in seeds))

    # ---- 任务4：动态配对 ----
    print(f"\n任务4 动态配对（memory_gain=0, seconds={args.seconds:.0f}）")
    from loop.dynamic_tof_sim import load_rig
    dyn = dynamic(cfg, brain, groups, dn, seeds, scenarios, seconds=args.seconds, rig=load_rig())
    print(f"{'variant':5s} {'seed':>5s} {'scenario':10s} {'kind':10s} {'reach':>5s} {'t_goal':>7s} "
          f"{'cev':>4s} {'frontEv':>7s} {'bestD':>5s} {'minGap':>7s} {'winTurn':>8s} {'winSat%':>7s}")
    for var in ("BASE", "TEST"):
        for sd in seeds:
            for sc in scenarios:
                for kind in ("DN_readout",) if var == "BASE" else ("DN_readout", "DN_zero"):
                    r = dyn[var][(sd, sc, kind)]
                    print(f"{var:5s} {sd:5d} {sc:10s} {kind:10s} {str(r['reach'])[:1]:>5s} "
                          f"{r['t_goal']:7.2f} {r['cev']:4d} {r['front_events']:7d} {r['bestD']:5d} "
                          f"{r['minGap']:7.3f} {r['win_meanturn']:+8.3f} {r['win_sat%']:7.1f}")
    def reach_rate(var, kind):
        return sum(dyn[var][(sd, sc, kind)]["reach"] for sd in seeds for sc in scenarios)
    print(f"\n到达率: BASE readout {reach_rate('BASE','DN_readout')}/{len(seeds)*len(scenarios)}  "
          f"TEST readout {reach_rate('TEST','DN_readout')}/{len(seeds)*len(scenarios)}  "
          f"TEST zero {reach_rate('TEST','DN_zero')}/{len(seeds)*len(scenarios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
