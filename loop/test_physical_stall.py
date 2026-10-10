"""P2 windowed stall detector 离线验证（不连硬件、不动车）。

场景：odom 50/44/30/20Hz × 实速比 1.0/0.5/0.3/0.2/0.1；真零位移堵转；
启动延迟；间歇编码器；原地转向；STOP；front_safety；stale。

运行： python -m loop.test_physical_stall
"""
from __future__ import annotations

import math
import sys

import numpy as np

from loop.physical_stall import WindowedStallDetector

MPC = 0.40
M_PER_COUNT = math.pi * 0.070 / 1560.0
_fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _fails.append(name)


def simulate(odom_hz, r, *, v_cmd=0.66, max_speed=0.9, T=20.0, startup_s=0.0,
             zero_move=False, intermittent=0, front_blocked=False, turn=False,
             stop=False, stale_after=None):
    brain_dt = 0.02
    n = int(T / brain_dt)
    v_real = 0.0 if zero_move else v_cmd * r
    w_real = 0.5 if turn else 0.0
    det = WindowedStallDetector(m_per_count=M_PER_COUNT, meters_per_cell=MPC)
    ot = np.arange(0.0, T, 1.0 / odom_hz)
    move_cells = v_real * ot
    turn_rad = w_real * ot
    if turn:                                    # 原地转：轮子反向，各走 L/2
        left_cum = -turn_rad * 0.08 / M_PER_COUNT
        right_cum = +turn_rad * 0.08 / M_PER_COUNT
        pos_cells = 0.0 * ot
    else:
        pos_cells = move_cells
        left_cum = -(pos_cells / MPC) / (M_PER_COUNT / MPC)
        right_cum = +(pos_cells / MPC) / (M_PER_COUNT / MPC)
    if intermittent:
        for k in range(0, len(ot), intermittent):
            left_cum[k] = left_cum[k - 1] if k > 0 else 0.0
            right_cum[k] = right_cum[k - 1] if k > 0 else 0.0

    j = 0
    stalled_any = 0
    reasons = set()
    for i in range(n):
        t = i * brain_dt
        # consume all odom frames up to t
        while j < len(ot) and ot[j] <= t:
            if stale_after is None or ot[j] <= stale_after:
                det.observe(ot[j], int(left_cum[j]), int(right_cum[j]), seq=j)
            j += 1
        det.note_command(max(0.0, t - 0.0) * 0 + (startup_s if startup_s > 0 else 0.0), 0.0, 0.0) if False else None
        # command: at t<startup_s it's 0 (ramp), else v_cmd
        vc = 0.0 if (stop or t < startup_s) else v_cmd
        wc = 0.0 if (stop or t < startup_s) else (0.5 if turn else 0.0)
        det.note_command(t, vc, wc)
        st, reason = det.stalled(t, vc, wc, front_blocked=front_blocked)
        if st:
            stalled_any += 1
        if reason:
            reasons.add(reason)
    return stalled_any / n, reasons


print("=== 目标：慢/不同步/间歇 不触发；真堵转 触发 ===")
for oh in (50.0, 44.0, 30.0, 20.0):
    for r in (1.0, 0.5, 0.3, 0.2, 0.1):
        frac, reasons = simulate(oh, r)
        ok = frac == 0.0
        check(f"odom={oh:g} r={r} -> no stall", ok, f"stalled={frac*100:.0f}% reasons={reasons}")

print("\n=== 真堵转 / 特殊场景 ===")
f, _ = simulate(50.0, 1.0, zero_move=True)
check("true zero-move stall fires", f > 0, f"stalled={f*100:.0f}%")
f, _ = simulate(44.0, 0.3, startup_s=0.3)
check("startup 0.3s grace -> no stall", f == 0, f"stalled={f*100:.0f}%")
f, _ = simulate(44.0, 1.0, intermittent=5)
check("intermittent encoder -> no stall", f == 0, f"stalled={f*100:.0f}%")
f, rs = simulate(50.0, 1.0, turn=True)
check("in-place turn -> no stall", f == 0, f"stalled={f*100:.0f}% reasons={rs}")
f, rs = simulate(50.0, 1.0, stop=True)
check("STOP -> reason 'stop', no stall", f == 0 and "stop" in rs, f"reasons={rs}")
f, rs = simulate(50.0, 1.0, front_blocked=True)
check("front_blocked -> reason 'front_safety', no stall", f == 0 and "front_safety" in rs, f"reasons={rs}")
f, rs = simulate(50.0, 1.0, stale_after=2.0)
check("odom stale -> no stall (external FAILSAFE)", f == 0, f"stalled={f*100:.0f}%")

print(f"\n  {'ALL PASS' if not _fails else 'FAILED: ' + ', '.join(_fails)}")
sys.exit(1 if _fails else 0)
