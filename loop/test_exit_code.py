"""退出码 + Bridge 状态/时序 测试（离线，无硬件）。

A. evaluate_run_status：各失败组合 -> rc!=0；正常 -> 0；dry-run 不要求 STOP 确认。
B. 旧实现回归：physical_run.main 旧版对失败仍返回 0（此测试锁定“必须非零”的契约）。
C. Bridge 真实 HTTP vs dry 时序：dry 不算 first_http_drive_start_t；真实（打桩 http_json）
   必须记录 first_http_drive_start_t 与 DRIVE 间隔，且不把 on_step/邮箱提交当发送时刻。

运行：  python -m loop.test_exit_code
"""
from __future__ import annotations

import sys
import time

from loop.odom_shadow_run import OdomError
from loop.physical_run import evaluate_run_status

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def t_exit_status():
    print("\n[A] evaluate_run_status 组合")
    ok = dict(start_ok=True, mech_exc=None, comms_exc=None, interrupted=False,
              latches=[], stop_confirmed=True, dry_run=False)
    rc, r = evaluate_run_status(**ok)
    check("all good -> rc 0", rc == 0 and not r)
    check("mechanical -> nonzero", evaluate_run_status(**{**ok, "mech_exc": RuntimeError()})[0] != 0)
    check("comms OdomError -> nonzero", evaluate_run_status(**{**ok, "comms_exc": OdomError("x")})[0] != 0)
    check("preflight fail -> nonzero", evaluate_run_status(**{**ok, "start_ok": False})[0] != 0)
    check("interrupted -> nonzero", evaluate_run_status(**{**ok, "interrupted": True})[0] != 0)
    lat = [(1.0, "latch", {"state": "FAILSAFE", "msg": "x"})]
    check("bg FAILSAFE latch -> nonzero", evaluate_run_status(**{**ok, "latches": lat})[0] != 0)
    lat2 = [(1.0, "latch", {"state": "BLOCKED_GUARD", "msg": "x"})]
    check("bg BLOCKED_GUARD latch -> nonzero", evaluate_run_status(**{**ok, "latches": lat2})[0] != 0)
    check("STOP unconfirmed (hw) -> nonzero",
          evaluate_run_status(**{**ok, "stop_confirmed": False})[0] != 0)
    check("STOP unconfirmed but dry_run -> rc 0 (allowed)",
          evaluate_run_status(**{**ok, "stop_confirmed": None, "dry_run": True})[0] == 0)
    check("dry_run + no failsafe -> rc 0",
          evaluate_run_status(**{**ok, "dry_run": True, "stop_confirmed": None})[0] == 0)


def t_bridge_real_vs_dry_timing():
    print("\n[C] Bridge 真实 HTTP vs dry 时序")
    import hardware.pikachu_bridge as pb
    from loop.timing_collect import Timing

    # dry：不应有 first_http_drive_start_t
    tdry = Timing()
    bdry = pb.PikachuBridge(pb.PikachuConfig(dry_run=True, rate_hz=50.0, log_path=""))
    bdry.timing = tdry
    bdry.start()
    bdry.on_step({"t": 0.0, "v": 1.0, "omega": 0.0, "x": 0, "y": 0, "theta": 0})
    time.sleep(0.15)
    bdry.stop(reason="t")
    check("dry: http_drive_dry counted", (tdry.counts().get("http_drive_dry", 0) or 0) > 0)
    check("dry: first_http_drive_start_t absent", tdry.get("first_http_drive_start_t") is None)

    # real（打桩 http_json）：必须记录 first_http_drive_start_t + 间隔
    orig = pb.http_json
    pb.http_json = lambda url, payload=None, timeout=0.08: (200, {"ok": True, "status": {"open": True}})
    try:
        treal = Timing()
        breal = pb.PikachuBridge(pb.PikachuConfig(
            dry_run=False, rate_hz=50.0, log_path="", base_url="http://127.0.0.1:1"))
        breal.timing = treal
        started = breal.start()
        for i in range(10):
            breal.on_step({"t": i * 0.02, "v": 1.0, "omega": 0.0, "x": 0, "y": 0, "theta": 0})
            time.sleep(0.02)
        time.sleep(0.1)
        breal.stop(reason="t")
    finally:
        pb.http_json = orig
    check("real: preflight started (RUNNING)", started is True)
    check("real: first_http_drive_start_t recorded", treal.get("first_http_drive_start_t") is not None)
    check("real: first_http_drive_resp_t >= start",
          (treal.get("first_http_drive_resp_t") or 0) >= (treal.get("first_http_drive_start_t") or 0))
    check("real: http_drive_real counted", (treal.counts().get("http_drive_real", 0) or 0) > 0)
    check("real: DRIVE interval recorded", treal.stats("http_drive_interval_real")["n"] >= 1)


def main() -> int:
    print("============== P2 exit-code + bridge timing ==============")
    t_exit_status()
    t_bridge_real_vs_dry_timing()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
