"""Bridge 发送路径竞态测试（真实 PikachuBridge + 真实邮箱/发送线程，dry_run=True）。

聚焦两件事：
  A. FAILSAFE latch 恰好发生在 mailbox.take() 与 _send(DRIVE) 之间时，过期非零
     DRIVE 是否可能通过发送出口。
  B. 已经开始的在途 DRIVE 发送与 latch 的序列化：最终 STOP 不得被过期 DRIVE 覆盖。

使用确定性线程同步（threading.Event），不依赖 sleep 判定；dry_run=True 不连 Pi/电机。
stop() 的线程退出与 STOP 未确认报告也一并检查。

运行：  python -m loop.test_bridge_race
"""
from __future__ import annotations

import sys
import threading

from hardware.pikachu_bridge import BridgeState, PikachuBridge, PikachuConfig

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def _mk_bridge(rate_hz=200.0):
    cfg = PikachuConfig(dry_run=True, rate_hz=rate_hz, log_path="",
                        sim_max_speed=0.9, sim_max_omega=2.6, min_cmd=0.0)
    bridge = PikachuBridge(cfg)
    sends = []
    ev = {"stop": threading.Event(), "drive": threading.Event()}
    orig_send = bridge._send

    def send(V, W, kind, item):
        sends.append((kind, round(float(V), 4), round(float(W), 4)))
        if kind == "STOP":
            ev["stop"].set()
        if kind == "DRIVE" and (abs(V) > 1e-6 or abs(W) > 1e-6):
            ev["drive"].set()
        return orig_send(V, W, kind, item)

    bridge._send = send
    return bridge, sends, ev


def _drive_rec(v=0.9, t=0.0):
    return {"t": t, "v": v, "omega": 0.0, "x": 0.0, "y": 0.0, "theta": 0.0}


def _nonzero_drives(sends):
    return [s for s in sends if s[0] == "DRIVE" and (abs(s[1]) > 1e-6 or abs(s[2]) > 1e-6)]


def t_failsafe_at_take():
    print("\n[T1] FAILSAFE 恰在 take 与 _send(DRIVE) 之间 latch -> 不发过期 DRIVE")
    bridge, sends, ev = _mk_bridge()
    orig_take = bridge._mailbox.take
    injected = {"done": False}

    def take():
        item = orig_take()
        if item is not None and not injected["done"]:
            injected["done"] = True               # 同发送线程重入：模拟 latch 命中 take 那一瞬
            bridge.enter_failsafe("race: injected exactly at take")
        return item

    bridge._mailbox.take = take
    bridge.start()
    bridge.on_step(_drive_rec(0.9))               # 推一条非零 DRIVE 进邮箱
    got_stop = ev["stop"].wait(2.0)               # 确定性：等到出口发出 STOP
    state_at_stop = bridge.state                  # stop() 之前的状态
    bridge.stop(reason="test")
    check("FAILSAFE latched", state_at_stop is BridgeState.FAILSAFE)
    check("a STOP reached the outlet", got_stop)
    check("NO non-zero DRIVE ever reached the outlet", len(_nonzero_drives(sends)) == 0)


def t_inflight_serialized():
    print("\n[T2] 在途 DRIVE 与 latch 串行：最终 STOP 不被过期 DRIVE 覆盖")
    bridge, sends, ev = _mk_bridge()
    drive_started = threading.Event()
    release_drive = threading.Event()
    orig_send = bridge._send
    held = {"n": 0}

    def send(V, W, kind, item):
        if kind == "DRIVE" and (abs(V) > 1e-6 or abs(W) > 1e-6) and held["n"] == 0:
            held["n"] += 1
            drive_started.set()                    # 通知主线程：在途 DRIVE 已开始
            release_drive.wait(2.0)                # 停在 _send 内部（持有 _send_lock）
        return orig_send(V, W, kind, item)

    bridge._send = send
    bridge.start()
    bridge.on_step(_drive_rec(0.9))

    assert drive_started.wait(2.0), "sender never started the DRIVE"
    latch_thread = threading.Thread(
        target=bridge.enter_failsafe, args=("race: during in-flight DRIVE",), daemon=True)
    latch_thread.start()                           # 该线程会阻塞在 _send_lock 上
    latch_thread.join(0.2)                         # 给它时间进入“等待”，但不应完成
    blocking = latch_thread.is_alive()
    release_drive.set()                            # 放行在途 DRIVE
    got_stop = ev["stop"].wait(2.0)
    latch_thread.join(2.0)
    bridge.stop(reason="test")

    nd = _nonzero_drives(sends)
    check("latch blocks until in-flight DRIVE done (serialized)", blocking)
    check("DRIVE went out before latch completed", len(nd) >= 1)
    check("STOP went out after the DRIVE", got_stop)
    # 关键：latch 完成后不再有非零 DRIVE；最后一条运动命令是 STOP
    last_motion = [s for s in sends if s[0] in ("DRIVE", "STOP")]
    check("last motion command is STOP (not overwritten by stale DRIVE)",
          bool(last_motion) and last_motion[-1][0] == "STOP")
    check("no non-zero DRIVE after the STOP",
          not any(s[0] == "DRIVE" and s[1] != 0 for s in sends[sends.index(last_motion[-1]):]))


def t_onstep_after_latch_dropped():
    print("\n[T3] latch 之后 on_step 提交的非零不得通过出口")
    bridge, sends, ev = _mk_bridge()
    bridge.start()
    bridge.enter_failsafe("race: latch before further on_step")
    bridge.on_step(_drive_rec(0.9))               # 应被 on_step 直接丢弃（state != RUNNING）
    got_stop = ev["stop"].wait(2.0)
    bridge.stop(reason="test")
    check("a STOP reached the outlet", got_stop)
    check("no non-zero DRIVE after latch", len(_nonzero_drives(sends)) == 0)


def t_stop_thread_exit_and_report():
    print("\n[T4] stop() 线程退出 + STOP 未确认报告")
    bridge, sends, ev = _mk_bridge()
    bridge.start()
    bridge.on_step(_drive_rec(0.9))
    ev["drive"].wait(2.0)
    bridge.stop(reason="test")
    check("sender thread exited", bridge._thread is None or not bridge._thread.is_alive())
    check("state == STOPPED", bridge.state is BridgeState.STOPPED)
    # dry_run：没有真实 STOP -> 明确标为未确认为“已确认”，并给出说明
    check("dry_run stop reported as NOT confirmed (None)", bridge.last_stop_confirmed is None)
    check("dry_run note present", "dry_run" in (bridge.last_stop_note or ""))


def main() -> int:
    print("============== PikachuBridge send-path race test (dry_run) ==============")
    t_failsafe_at_take()
    t_inflight_serialized()
    t_onstep_after_latch_dropped()
    t_stop_thread_exit_and_report()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
