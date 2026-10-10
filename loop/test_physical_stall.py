"""P2 windowed stall detector 离线验证 v2（不连硬件、不动车）。

场景：odom 50/44/30/20Hz × 实速比 1.0/0.5/0.3/0.2/0.1；真零位移堵转；启动延迟；
间歇编码器；原地转向；波动命令；突然 STOP；front_safety；stale。

运行： python -m loop.test_physical_stall
"""
from __future__ import annotations

import math
import sys

import numpy as np

from loop.physical_stall import WindowedStallDetector

MPC = 0.40
D, L, CPR = 0.067, 0.194, 1560.0
K = (math.pi * D / CPR) / MPC          # cells / count
_fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _fails.append(name)


def simulate(odom_hz, r, *, v_cmd=0.66, T=20.0, startup_s=0.0, zero_move=False,
             intermittent=0, front_blocked=False, turn=False, stop=False,
             stale_after=None, jitter=0.0):
    brain_dt = 0.02
    n = int(T / brain_dt)
    det = WindowedStallDetector(wheel_diam=D, track=L, cpr=CPR, meters_per_cell=MPC,
                                max_speed=0.9, max_omega=3.0)
    ot = np.arange(0.0, T, 1.0 / odom_hz)
    v_real = 0.0 if zero_move else v_cmd * r
    w_real = 0.6 if turn else 0.0
    moved = v_real * ot
    turnr = w_real * ot
    eL = moved - turnr * (L / MPC) / 2.0     # 左轮 cells
    eR = moved + turnr * (L / MPC) / 2.0     # 右轮 cells
    cntL = np.round(-eL / K).astype(int)     # left_sign = -1
    cntR = np.round(+eR / K).astype(int)
    if intermittent:
        for k in range(0, len(ot), intermittent):
            cntL[k] = cntL[k - 1] if k > 0 else 0
            cntR[k] = cntR[k - 1] if k > 0 else 0

    j = 0
    stalled_any = 0
    reasons = set()
    for i in range(n):
        t = i * brain_dt
        while j < len(ot) and ot[j] <= t:
            if stale_after is None or ot[j] <= stale_after:
                det.observe(ot[j], int(cntL[j]), int(cntR[j]), seq=j)
            j += 1
        vc = 0.0 if (stop or t < startup_s) else v_cmd
        wc = 0.0 if (stop or t < startup_s) else (0.6 if turn else 0.0)
        if jitter:
            vc += jitter * math.sin(6.0 * t)   # 脑逐周期抖动
            wc += jitter * math.sin(6.0 * t + 1.0)
        det.note_command(t, vc, wc)
        st, reason = det.stalled(t, vc, wc, front_blocked=front_blocked)
        if st:
            stalled_any += 1
        if reason:
            reasons.add(reason)
    return stalled_any / n, reasons


print("=== 慢/不同步/间歇 -> 不触发 ===")
for oh in (50.0, 44.0, 30.0, 20.0):
    for r in (1.0, 0.5, 0.3, 0.2, 0.1):
        f, rs = simulate(oh, r)
        check(f"odom={oh:g} r={r} no stall", f == 0, f"stalled={f*100:.0f}%")

print("\n=== 波动命令 / 特殊场景 ===")
f, rs = simulate(44.0, 0.5, jitter=0.05)
check("fluctuating cmd (brain jitter) no stall", f == 0, f"stalled={f*100:.0f}%")
f, _ = simulate(50.0, 1.0, zero_move=True)
check("true zero-move fires", f > 0, f"stalled={f*100:.0f}%")
f, _ = simulate(44.0, 0.3, startup_s=0.3)
check("startup 0.3s grace no stall", f == 0, f"stalled={f*100:.0f}%")
f, _ = simulate(44.0, 1.0, intermittent=5)
check("intermittent encoder no stall", f == 0, f"stalled={f*100:.0f}%")
f, _ = simulate(50.0, 1.0, turn=True)
check("in-place turn no stall", f == 0, f"stalled={f*100:.0f}%")
f, rs = simulate(50.0, 1.0, stop=True)
check("STOP reason 'stop'", f == 0 and "stop" in rs, f"reasons={rs}")
f, rs = simulate(50.0, 1.0, front_blocked=True)
check("front_blocked reason 'front_safety'", f == 0 and "front_safety" in rs, f"reasons={rs}")
f, _ = simulate(50.0, 1.0, stale_after=2.0)
check("stale no stall (FAILSAFE external)", f == 0, f"stalled={f*100:.0f}%")

print("\n=== sustain 计时 ===")
# 检测器自身 sustain=0.35s；加上冻结 Decoder stall_after=0.30s -> 预计 ~0.65s
f, _ = simulate(50.0, 1.0, zero_move=True, T=3.0)
check("detector sustain reaches stall in [0.3,0.8]s of sustained zero-move",
      f > 0.15, f"fraction-with-zero-move {f*100:.0f}% (0.35s of 3s ~ 12%, 0.65s ~ 22%)")

print(f"\n  {'ALL PASS' if not _fails else 'FAILED: ' + ', '.join(_fails)}")
sys.exit(1 if _fails else 0)
