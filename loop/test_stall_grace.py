"""复现/回归：WindowedStallDetector 的 startup grace 必须由 STOP/idle->active 转换触发。

bug：note_command 只在 |Δcmd|>cmd_change_thresh(0.35) 时更新 _cmd_change_t；而 readout
smooth 启动是 0.236/0.390/0.490/0.555…（相邻 Δ<0.35）→ 宽限从未开始 → grace 失效 →
零位移时 sustain(0.50s) 一到就误判 mechanical。

运行：  python -m loop.test_stall_grace
"""
from __future__ import annotations

import sys

from loop.physical_stall import WindowedStallDetector

DT = 0.02
MAX_SPEED = 0.9
MAX_OMEGA = 2.6
MPC = 0.40
RAMP = [0.236, 0.390, 0.490, 0.555, 0.620, 0.660, 0.675]   # readout smooth ~0.675

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def mk():
    return WindowedStallDetector(meters_per_cell=MPC, max_speed=MAX_SPEED, max_omega=MAX_OMEGA)


def _drive(det, *, t0, dur, v_of_i, w_of_i, ticks_of_i, seq0, theta_of_i=None):
    """50Hz：新 odom seq + note_command + stalled。返回 [(t, stalled, reason, grace, sustain)]。"""
    out = []
    n = int(dur / DT)
    seq = seq0
    for i in range(n):
        t = t0 + i * DT
        seq += 1
        det.observe(t, *ticks_of_i(i), seq=seq,
                    theta=(0.0 if theta_of_i is None else theta_of_i(i)))
        v, w = v_of_i(i), (0.0 if w_of_i is None else w_of_i(i))
        det.note_command(t, v, w)
        stalled, reason = det.stalled(t, v, w, front_blocked=False)
        out.append((t, stalled, reason, det.grace_active, det.sustain_duration))
    return out, seq


def _first_mech(res):
    for t, stalled, reason, _g, _s in res:
        if stalled and reason == "mechanical":
            return t
    return None


# --------------------------------------------------------------------------
def t_grace_activates_on_start():
    print("\n[T1] 启动运动命令后 grace 立即 True；grace 期内不得 mechanical；之后仍能触发")
    det = mk()
    res, _ = _drive(det, t0=0.0, dur=1.5,
                    v_of_i=lambda i: RAMP[min(i, len(RAMP) - 1)],
                    w_of_i=None, ticks_of_i=lambda i: (0, 0), seq0=0)
    check("grace_active True on first active command", res[0][3] is True)
    early = [r for r in res if r[0] < 0.70]
    check("no mechanical within first 0.70s grace",
          all(not (s and r == "mechanical") for _t, s, r, _g, _su in early))
    mt = _first_mech(res)
    check(f"mechanical still fires after grace+sustain (t={mt})",
          mt is not None and mt >= 0.70 + 0.50 - 1e-6)


def t_jitter_no_infinite_reset():
    print("\n[T2] 小幅 jitter 不得无限重置 grace（应 ~grace+sustain 触发）")
    det = mk()
    # 稳态 v≈0.675 + 小抖动（相邻 Δ 远小于 0.35），零位移
    def v_of_i(i):
        return 0.675 + 0.03 * (1 if i % 2 else -1)
    res, _ = _drive(det, t0=0.0, dur=1.6, v_of_i=v_of_i, w_of_i=None,
                    ticks_of_i=lambda i: (0, 0), seq0=100)
    mt = _first_mech(res)
    check(f"mechanical fires ~1.2s despite jitter (t={mt})",
          mt is not None and mt <= 0.70 + 0.50 + 0.12)


def t_stop_restart_new_grace():
    print("\n[T3] STOP -> 重新启动运动得到新的 grace")
    det = mk()
    res, seq = _drive(det, t0=0.0, dur=0.8, v_of_i=lambda i: 0.675, w_of_i=None,
                      ticks_of_i=lambda i: (0, 0), seq0=200)
    check("still within grace at t=0.8 (no stall yet)", _first_mech(res) is None)
    # STOP 0.3s
    res2, seq = _drive(det, t0=0.8, dur=0.3, v_of_i=lambda i: 0.0, w_of_i=None,
                       ticks_of_i=lambda i: (0, 0), seq0=seq)
    check("STOP classified as 'stop'", any(r == "stop" for _t, _s, r, _g, _su in res2))
    # restart motion
    res3, seq = _drive(det, t0=1.1, dur=0.1, v_of_i=lambda i: 0.675, w_of_i=None,
                       ticks_of_i=lambda i: (0, 0), seq0=seq)
    check("grace_active True right after restart", res3[0][3] is True)


def t_front_safety_and_stale():
    print("\n[T4] front_safety / stale 不得退化")
    det = mk()
    t = 0.0
    for i in range(60):
        t = i * DT
        det.observe(t, 0, 0, seq=i, theta=0.0)
        det.note_command(t, MAX_SPEED, 0.0)
    s, r = det.stalled(t, MAX_SPEED, 0.0, front_blocked=True)
    check("front_blocked -> (not stalled,'front_safety')", s is False and r == "front_safety")
    det2 = mk()
    det2.observe(0.0, 0, 0, seq=0, theta=0.0)
    det2.note_command(0.0, MAX_SPEED, 0.0)
    s, r = det2.stalled(1.0, MAX_SPEED, 0.0)
    check("stale -> (not stalled,'')", s is False and r == "")


def main() -> int:
    print("========== WindowedStallDetector startup-grace regression ==========")
    t_grace_activates_on_start()
    t_jitter_no_infinite_reset()
    t_stop_restart_new_grace()
    t_front_safety_and_stale()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
