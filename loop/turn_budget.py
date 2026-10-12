"""P2d-5 任务三：把实体运行的航向变化分解到「左右轮差」与「底盘偏置」。

只做算术复核（不改任何算法）。用标定值 D=0.067 m / L=0.194 m / CPR=1560。

用法：
  python -m loop.turn_budget --name bg0   --dL -3328 --dR 3415 --obs-deg -3.5
  python -m loop.turn_budget --name bg010 --dL -1607 --dR 1811 --obs-deg -8.1  --w-mean -0.02
  python -m loop.turn_budget --name bg015 --dL -1556 --dR 2037 --obs-deg -19.2 --w-mean -0.03
"""
from __future__ import annotations

import argparse
import math

D = 0.067
L = 0.194
CPR = 1560.0
MPC = math.pi * D / CPR          # m/count


def analyze(name, dL, dR, obs_deg, w_mean=None):
    dl = abs(dL) * MPC
    dr = abs(dR) * MPC
    mean = 0.5 * (dl + dr)
    diff = dr - dl               # >0 表示右轮走得更远（右转）
    dtheta_wheel_rad = diff / L
    dtheta_wheel_deg = math.degrees(dtheta_wheel_rad)
    ratio = (diff / mean * 100.0) if mean > 0 else float("nan")
    per_m = (dtheta_wheel_deg / mean) if mean > 0 else float("nan")
    print(f"[{name}] dist(L)={dl:.4f}m dist(R)={dr:.4f}m mean={mean:.4f}m "
          f"diff={diff:+.4f}m ({ratio:+.2f}%)")
    print(f"         wheel-dtheta={dtheta_wheel_deg:+.2f}deg  observed={obs_deg:+.2f}deg "
          f"(match={abs(dtheta_wheel_deg-obs_deg):.2f}deg)  turn-rate={per_m:+.1f} deg/m"
          + ("" if w_mean is None else f"  sent_W_mean={w_mean:+.4f}"))
    return {"dist_m": mean, "diff_m": diff, "ratio_pct": ratio,
            "wheel_dtheta_deg": dtheta_wheel_deg, "per_m": per_m}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dL", type=int, required=True, help="左轮 tick 变化（带符号，右转→负）")
    ap.add_argument("--dR", type=int, required=True, help="右轮 tick 变化（带符号）")
    ap.add_argument("--obs-deg", type=float, required=True, help="真实 odom 航向变化(度)")
    ap.add_argument("--w-mean", type=float, default=None, help="该运行的 sent_W 均值")
    a = ap.parse_args()
    analyze(a.name, a.dL, a.dR, a.obs_deg, a.w_mean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
