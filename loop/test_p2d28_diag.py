"""P2d-28 回归：p2d28_diag 的离线事件/指标/归因纯函数（合成反例）。**无模拟器**。"""
from __future__ import annotations

import sys

from loop.p2d28_diag import (ATTRIB, attribute, events, idx_first, metrics)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def rec(t, dminF, turn, sizeLR, theta, v=0.0, gd=99, Dmb=0.3, front=False, cev=0, turn_pre=None,
        tof_FL=1000.0, tof_FR=1000.0):
    return {"t": t, "dmin_F": dminF, "turn": turn, "size_L": sizeLR, "size_R": 0.0,
            "theta": theta, "v": v, "gd": gd, "Dmb": Dmb, "front": front, "cev": cev,
            "need_deg": 0.0, "D": Dmb, "om": turn, "tof_FL": tof_FL, "tof_FR": tof_FR,
            "tof_F": dminF * 1000, "turn_pre": (turn * 1.5 if turn_pre is None else turn_pre)}


def main() -> int:
    print("=========== p2d28_diag (synthetic) ===========")
    dt = 0.02
    # 合成：接近 -> 转向饱和 -> 侧障碍清 -> 命令回零 -> 恢复前进（成功型）
    recs = []
    tt = 0.0
    for i in range(50):
        recs.append(rec(tt, 2.0, 0.0, 0.0, 0.0, v=0.05)); tt += dt       # 接近前
    for i in range(200):
        recs.append(rec(tt, 0.5, -1.0, -0.6, -i * 0.01, v=0.01)); tt += dt  # 转向饱和
    for i in range(200):
        recs.append(rec(tt, 1.5, 0.0, 0.0, -2.0 - 0.0 * i, v=0.05, gd=5)); tt += dt  # 侧清+回零+前进
    e = events(recs, dt)
    check("approach detected", e["approach"] == 49 or (e["approach"] is not None and recs[e["approach"]]["dmin_F"] < 1.2))
    check("turn_start detected", e["turn_start"] == 50)
    check("lateral_diff_zero detected", e["lateral_diff_zero"] == 250)
    check("forward_resume detected", e["forward_resume"] is not None)
    m = metrics(recs, dt)
    check("saturated during turn phase", m["frac_sat_after_turn"] > 0.3)
    check("sign_turn negative", m["sign_turn"] < 0)
    check("no long rotation after turn zero", abs(m["heading_after_turn_zero_deg"]) < 5)

    # 失败型：持续饱和 + D 不回背景 + 侧障碍不清 + 命令趋零后仍旋转
    recs2 = []
    tt = 0.0
    for i in range(50):
        recs2.append(rec(tt, 2.0, 0.0, 0.0, 0.0)); tt += dt
    for i in range(500):
        recs2.append(rec(tt, 0.4, 1.0, 0.6, i * 0.02, v=0.0, Dmb=0.4, tof_FL=300, tof_FR=300)); tt += dt
    recs2.append(rec(tt, 0.4, 0.0, 0.6, 10.0, v=0.0, Dmb=0.4, tof_FL=300, tof_FR=300)); tt += dt
    for i in range(30):
        recs2.append(rec(tt, 0.4, 0.0, 0.6, 12.0 + i * 0.05, v=0.0, Dmb=0.4, tof_FL=300, tof_FR=300)); tt += dt
    m2 = metrics(recs2, dt)
    tags = attribute(m2, recs2)
    check("failure flagged saturated (M)", "M" in tags)
    check("failure flagged D not back (N)", "N" in tags)
    check("failure flagged side present (S)", "S" in tags)
    check("vehicle response NOT an independent fault (no V)", "V" not in tags)
    check("sign-flips-per-second field present", "turn_sign_flips_per_s" in m2)

    # 无证据 -> U
    recs3 = [rec(i * dt, 3.0, 0.0, 0.0, 0.0) for i in range(10)]
    check("no evidence -> U", attribute(metrics(recs3, dt), recs3) == ["U"])

    check("ATTRIB has 5 categories", set(ATTRIB) == {"S", "N", "M", "V", "U"})
    check("idx_first none-sentinel", idx_first(recs3, lambda r: False) is None)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
