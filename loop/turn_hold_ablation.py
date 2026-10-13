"""P2d-14：短时转向保持的公平对照（P2 离线）。

BASE 编码器 / 现有 b0 标定 / DNa02 / 前进速度 / 动力学全部固定，仅换 P2 转向来源：
  DN_ema  原来的 DNa02 + EMA
  DN_hold DNa02 + 短时保持（预注册 HOLD_THRESH/HOLD_S/HOLD_LEVEL）
  DN_zero 神经转向置零
  DN_swap 神经转向反号（左右交换）
在 must_left / must_right 上按 seed 配对；先 64/101/202/303/512，若改善再在声明的新 seed 确认。

用法： python -m loop.turn_hold_ablation --seeds 64,101,202,303,512
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from loop.dn_steering import calibrate_baseline
from loop.dn_steering_diag import run_trace
from loop.local_avoidance_bench import SCENARIOS
from loop.run import make_setup
from loop.turn_hold import HOLD_MAG_CAP, HOLD_S, HOLD_THRESH, make_source, required_omega_sign

VARIANTS = ("DN_ema", "DN_hold", "DN_zero", "DN_swap")


def audit():
    print("任务1 方向坐标审计（y 向下增大；正 omega -> theta 增大 -> 转向 +y）")
    print(f"  {'scenario':11s} {'start':>16s} {'goal':>12s} {'required_omega_sign':>20s}")
    signs = {}
    for name in ("must_left", "must_right"):
        rows = SCENARIOS[name]
        from world.maze import Maze
        m = Maze(rows)
        signs[name] = required_omega_sign(rows)
        print(f"  {name:11s} {str(m.start):>16s} {str(m.goal[:2]):>12s} "
              f"{signs[name]:>20.0f}")
    print("  结论：must_left 需负 omega（开口 -y）；must_right 需正 omega（开口 +y）。"
          " 场景 ID/地图保持不变。")
    return signs


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
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    scenarios = args.scenarios.split(",")

    signs = audit()
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=seeds[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn = np.asarray(brain.cells(["descending_neuron"]))
    from loop.dynamic_tof_sim import load_rig
    rig = load_rig()
    b0 = {sd: calibrate_baseline(cfg, brain, groups, dn, sd) for sd in seeds}

    print(f"\n任务3 公平对照（BASE 编码器, memory_gain=0, seconds={args.seconds:.0f}; "
          f"预注册 hold: thresh={HOLD_THRESH} hold={HOLD_S}s mag_cap={HOLD_MAG_CAP}）")
    print(f"{'variant':8s} {'seed':>5s} {'scenario':10s} {'reach':>5s} {'t_goal':>7s} {'cev':>4s} "
          f"{'stop%':>6s} {'firstTurn':>9s} {'cumTurn_deg':>11s} {'holds':>5s} {'fail':>5s}")
    rows_out = {}
    for sd in seeds:
        for var in VARIANTS:
            for sc in scenarios:
                src, obj = make_source(var, b0[sd], brain.dt)
                s, recs = run_trace(cfg, brain, groups, dn, sd, var, SCENARIOS[sc], b0=b0[sd],
                                    seconds=args.seconds, rig=rig, turn_source=src)
                turns = np.array([r["turn"] for r in recs]) if recs else np.zeros(0)
                first = next((float(np.sign(t)) for t in turns if abs(t) >= 0.3), 0.0)
                cum = float(np.degrees(np.sum([r["om"] for r in recs]) * brain.dt)) if recs else 0.0
                holds = getattr(obj, "activations", 0) if obj is not None else 0
                # 失败原因（方向对否基于固定几何）
                req = signs[sc]
                fail = "-"
                if not s["reach"]:
                    ft = [t for t in turns if abs(t) >= 0.3]
                    if ft and (np.sign(np.mean(ft)) != req):
                        fail = "A_dir"
                    elif s["stop%"] > 35:
                        fail = "E_stop"
                    else:
                        fail = "C_weak"
                rows_out[(var, sd, sc)] = s
                print(f"{var:8s} {sd:5d} {sc:10s} {str(s['reach'])[:1]:>5s} "
                      f"{(recs[-1]['t'] if s['reach'] else -1.0):7.2f} {s['cev']:4d} "
                      f"{s['stop%']:6.1f} {first:+9.0f} {cum:11.1f} {holds:5d} {fail:>5s}")
    print("\n到达率：")
    for var in VARIANTS:
        n = sum(rows_out[(var, sd, sc)]["reach"] for sd in seeds for sc in scenarios)
        print(f"  {var:8s} {n}/{len(seeds) * len(scenarios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
