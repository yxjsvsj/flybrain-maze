"""P2d-6 任务一：用生产的通道符号 + swap_lr 逻辑复核实体运行的航向变化。

生产 `hardware/odometry.py` 的约定（`--swap-lr`）：
    ds_e1   = LEFT_SIGN  * dE1 * mpc      (LEFT_SIGN = -1)
    ds_e2   = RIGHT_SIGN * dE2 * mpc      (RIGHT_SIGN = +1)
    swap_lr -> physical_left, physical_right = ds_e2, ds_e1
    dtheta  = (physical_right - physical_left) / track
即 **UDP 的 left/right 字段是原始 E1/E2 counts，不是物理左右轮**。

用法：
  python -m loop.turn_budget --name bg010 --dE1 -1607 --dE2 1811 --obs-deg -8.1
"""
from __future__ import annotations

import argparse
import math

D_DEF = 0.067
L_DEF = 0.194
CPR_DEF = 1560.0


def physical_wheels(dE1, dE2, *, left_sign=-1.0, right_sign=+1.0, swap_lr=True,
                    wheel_diam=D_DEF, track=L_DEF, cpr=CPR_DEF):
    """原始 E1/E2 计数差 -> (physical_left_m, physical_right_m, ds_m, dtheta_rad)。"""
    mpc = math.pi * wheel_diam / cpr
    ds_e1 = left_sign * dE1 * mpc        # driver 把 E1 当“左”
    ds_e2 = right_sign * dE2 * mpc
    ds_left, ds_right = ds_e1, ds_e2
    if swap_lr:                          # 实测 E1/E2 与物理左右相反
        ds_left, ds_right = ds_right, ds_left
    ds = 0.5 * (ds_left + ds_right)
    dtheta = (ds_right - ds_left) / track
    return ds_left, ds_right, ds, dtheta


def analyze(name, dE1, dE2, obs_deg, *, left_sign, right_sign, swap_lr,
            wheel_diam, track, cpr, w_mean=None):
    dl, dr, ds, dtheta = physical_wheels(dE1, dE2, left_sign=left_sign, right_sign=right_sign,
                                         swap_lr=swap_lr, wheel_diam=wheel_diam,
                                         track=track, cpr=cpr)
    deg = math.degrees(dtheta)
    err = deg - obs_deg
    print(f"[{name}] raw E1={dE1} E2={dE2}  swap_lr={swap_lr} "
          f"left_sign={left_sign} right_sign={right_sign}")
    print(f"         physical L={dl:+.4f}m R={dr:+.4f}m ds={ds:+.4f}m  "
          f"dist={abs(ds):.4f}m")
    print(f"         wheel dtheta={deg:+.2f}deg  odom={obs_deg:+.2f}deg  "
          f"signed_err={err:+.2f}deg  |err|={abs(err):.2f}deg"
          + ("" if w_mean is None else f"  sent_W_mean={w_mean:+.4f}"))
    return {"ds_left": dl, "ds_right": dr, "ds": ds, "dtheta_deg": deg,
            "obs_deg": obs_deg, "signed_err_deg": err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dE1", type=int, required=True, help="原始 E1 计数差（UDP left）")
    ap.add_argument("--dE2", type=int, required=True, help="原始 E2 计数差（UDP right）")
    ap.add_argument("--obs-deg", type=float, required=True, help="真实 odom 航向变化(度, 带符号)")
    ap.add_argument("--wheel-diam", type=float, default=D_DEF)
    ap.add_argument("--track", type=float, default=L_DEF)
    ap.add_argument("--cpr", type=float, default=CPR_DEF)
    ap.add_argument("--left-sign", type=float, default=-1.0)
    ap.add_argument("--right-sign", type=float, default=+1.0)
    ap.add_argument("--no-swap-lr", action="store_true", help="关闭 swap_lr（默认开=生产）")
    ap.add_argument("--w-mean", type=float, default=None)
    a = ap.parse_args()
    analyze(a.name, a.dE1, a.dE2, a.obs_deg, left_sign=a.left_sign, right_sign=a.right_sign,
            swap_lr=not a.no_swap_lr, wheel_diam=a.wheel_diam, track=a.track, cpr=a.cpr,
            w_mean=a.w_mean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
