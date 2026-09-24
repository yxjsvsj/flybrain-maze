"""无硬件自检：用本地 mock Pikachu HTTP 服务把 Bridge 的每条安全路径打一遍。

    python -m hardware.selftest

不需要任何硬件。退出码非 0 表示有失败。
"""
from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hardware.pikachu_bridge import (BridgeState, PikachuBridge, PikachuConfig,
                                     http_json)
from loop.shadow_run import RealTimePacer

RESULTS: list[tuple[str, bool, str]] = []
TESTS: list[tuple[str, object]] = []


def check(name):
    """注册测试（**不在导入时执行**，由 main() 统一跑）。"""
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


# --------------------------------------------------------------------------- mock
class MockState:
    def __init__(self) -> None:
        self.guard_mode = False
        self.serial_open = True
        self.reconnect_ok = True
        self.reconnect_opens_port = True  # reconnect 成功后端口是否真的打开
        self.reconnect_delay_s = 0.0     # 模拟 CH340 打开串口时的 DTR 复位耗时
        # 模拟"设备掉过 USB，pyserial 仍认为 is_open=True，真正 write 才失败"
        self.stale_fd = False
        self.reconnect_fixes_stale = True  # reconnect 能否治好 stale fd
        self.drive_ok = True
        self.drive_http = 200
        self.latency_s = 0.0
        self.requests: list[tuple[str, dict]] = []
        self.timeline: list[tuple[str, float, float]] = []   # (path, 到达, 响应完成)
        self.lock = threading.Lock()


class MockServer:
    def __init__(self) -> None:
        self.state = MockState()
        state = self.state

        class H(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # type: ignore[override]  静音
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    return json.loads(raw) if raw else {}
                except Exception:                                  # noqa: BLE001
                    return {}

            def do_GET(self):
                if self.path == "/api/status":
                    self._send(200, {"port": "/dev/ttyUSB0", "baud": 115200,
                                     "open": state.serial_open, "last_error": None,
                                     "guard_mode": state.guard_mode,
                                     "guard_ready": False})
                else:
                    self._send(404, {})

            def do_POST(self):
                body = self._read()
                t_arrive = time.perf_counter()
                with state.lock:
                    state.requests.append((self.path, body))
                if state.latency_s:
                    time.sleep(state.latency_s)
                if self.path == "/api/reconnect":
                    if state.reconnect_delay_s:
                        time.sleep(state.reconnect_delay_s)
                    # 真实行为：reconnect 成功会把串口打开，之后 status.open 变 true
                    if state.reconnect_ok and state.reconnect_opens_port:
                        state.serial_open = True
                    if state.reconnect_fixes_stale:
                        state.stale_fd = False
                    self._send(200, {"ok": state.reconnect_ok,
                                     "status": {"open": state.serial_open,
                                                "port": "/dev/ttyUSB0",
                                                "baud": 115200,
                                                "last_error": None}})
                elif self.path == "/api/drive":
                    if state.guard_mode:
                        self._send(409, {"ok": False, "error": "Guard mode enabled",
                                         "status": {"open": state.serial_open}})
                    elif state.stale_fd:
                        # HTTP 200 但 ok=false：正是 SerialController.write 失败的样子
                        self._send(200, {"ok": False,
                                         "status": {"open": True,
                                                    "last_error":
                                                        "write failed: [Errno 19] No such device"}})
                    else:
                        self._send(state.drive_http,
                                   {"ok": state.drive_ok,
                                    "status": {"open": state.serial_open,
                                               "last_error": None}})
                else:
                    self._send(404, {})
                with state.lock:
                    state.timeline.append((self.path, t_arrive, time.perf_counter()))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def drives(self):
        return [b for p, b in self.state.requests if p == "/api/drive"]


def base_cfg(url: str, **kw: Any) -> PikachuConfig:
    d: dict[str, Any] = dict(
        base_url=url, rate_hz=20.0, timeout_s=0.2, max_failures=3,
        preflight_timeout_s=2.0, serial_settle_s=0.15,
        sim_max_speed=0.9, sim_max_omega=2.6,
        max_v=0.30, max_w=0.30, max_motor_mix=0.30, min_cmd=0.0, log_path="",
    )
    d.update(kw)
    return PikachuConfig(**d)


def rec(v=0.0, omega=0.0, t=0.0):
    return {"v": v, "omega": omega, "t": t, "x": 0.0, "y": 0.0, "theta": 0.0}


# --------------------------------------------------------------------------- 1 缩放
@check("T1 缩放/钳位/motor-mix")
def t1():
    b = PikachuBridge(base_cfg("http://127.0.0.1:1"))
    # 归一化：满速满角 -> 先被 max_v/max_w 钳，再被 mix 钳
    V, W, rv, rw, cl = b.scale(0.9, 0.0)          # 直行满速
    assert abs(rv - 1.0) < 1e-9 and abs(V - 0.30) < 1e-9, (V, rv)
    assert not cl
    V, W, rv, rw, cl = b.scale(0.9, 2.6)          # 满速+满角
    m = max(abs(V + W), abs(V - W))
    assert m <= 0.30 + 1e-9, f"mix 超限: {m}"
    assert cl, "应记录 mix 钳位事件"
    # 曲率比例保持
    assert abs(V / W - rv / rw) < 1e-6, (V / W, rv / rw)
    # 零输入 -> 零输出
    V, W, *_ = b.scale(0.0, 0.0)
    assert V == 0.0 and W == 0.0
    # omega_sign
    b2 = PikachuBridge(base_cfg("http://127.0.0.1:1", omega_sign=-1.0))
    _, W2, *_ = b2.scale(0.0, 1.3)
    assert W2 < 0, f"omega_sign=-1 应翻号，得到 {W2}"
    # v_gain：用足够小的输入，避免被 max_v 钳掉而观测不到 gain
    # 注意 scale() 返回的 raw 已经乘过 gain（是"钳位前的目标值"）
    b3 = PikachuBridge(base_cfg("http://127.0.0.1:1", v_gain=0.5))
    b4 = PikachuBridge(base_cfg("http://127.0.0.1:1", v_gain=1.0))
    V3, _, rv3, *_ = b3.scale(0.18, 0.0)          # 归一化 0.2 * 0.5 -> 0.1
    V4, _, rv4, *_ = b4.scale(0.18, 0.0)          # 归一化 0.2 * 1.0 -> 0.2
    assert abs(rv3 - 0.10) < 1e-9, f"v_gain=0.5 应得 0.10，得到 {rv3}"
    assert abs(rv4 - 0.20) < 1e-9, f"v_gain=1.0 应得 0.20，得到 {rv4}"
    assert abs(rv4 / rv3 - 2.0) < 1e-9, "gain 的作用应可见"
    assert abs(V3 - 0.10) < 1e-9 and abs(V4 - 0.20) < 1e-9
    # 大输入时 gain 后的值仍受 max_v 约束
    V5, *_ = b3.scale(0.9, 0.0)                   # 归一化 1.0 * 0.5 = 0.5 -> 钳到 0.3
    assert abs(V5 - 0.30) < 1e-9, f"应被 max_v 钳到 0.30，得到 {V5}"


# --------------------------------------------------------------------------- 2 邮箱
@check("T2 邮箱 latest-wins")
def t2():
    from hardware.pikachu_bridge import _Mailbox
    mb = _Mailbox()
    mb.put({"V": 1})
    mb.put({"V": 2})
    mb.put({"V": 3})
    assert mb.take()["V"] == 3, "应取到最新"
    assert mb.take() is None, "取走后应为空"
    mb.put({"V": 4})
    mb.clear()
    assert mb.take() is None, "clear 后应为空"


# --------------------------------------------------------------------------- 3 预检
@check("T3 预检成功 -> RUNNING")
def t3():
    srv = MockServer()
    try:
        b = PikachuBridge(base_cfg(srv.url))
        assert b.start(), "预检应通过"
        assert b.state is BridgeState.RUNNING
        b.stop()
        assert b.state is BridgeState.STOPPED
    finally:
        srv.close()


@check("T4 预检失败：guard_mode=true")
def t4():
    srv = MockServer()
    try:
        srv.state.guard_mode = True
        b = PikachuBridge(base_cfg(srv.url))
        assert not b.start(), "guard_mode=true 应拒绝启动"
        assert b.state is BridgeState.INIT
    finally:
        srv.close()


@check("T5 预检失败：reconnect ok=false")
def t5():
    srv = MockServer()
    try:
        srv.state.serial_open = False      # 只有串口没开时才会走 reconnect 分支
        srv.state.reconnect_ok = False
        b = PikachuBridge(base_cfg(srv.url))
        assert not b.start(), "reconnect 失败应拒绝启动"
    finally:
        srv.close()


@check("T6 预检失败：reconnect 报 ok 但串口仍没打开")
def t6():
    srv = MockServer()
    try:
        srv.state.serial_open = False
        srv.state.reconnect_opens_port = False   # 假成功：ok=true 但 open 仍是 false
        b = PikachuBridge(base_cfg(srv.url))
        assert not b.start(), "串口未真正打开应拒绝启动"
    finally:
        srv.close()


# --------------------------------------------------------------------------- 7 每帧
@check("T7 每帧 json ok=false 也算失败（不能只看 HTTP 200）")
def t7():
    srv = MockServer()
    try:
        srv.state.drive_ok = False
        b = PikachuBridge(base_cfg(srv.url, max_failures=2))
        assert not b.start(), "预检的 STOP 也会失败，应拒绝启动"
    finally:
        srv.close()


@check("T8 连续失败 -> FAILSAFE latch，且之后只发 STOP")
def t8():
    srv = MockServer()
    try:
        b = PikachuBridge(base_cfg(srv.url, max_failures=3))
        assert b.start()
        b.on_step(rec(v=0.5, omega=0.0))
        time.sleep(0.15)
        n_before = len(srv.drives())
        srv.state.drive_ok = False                 # 之后全部失败
        for i in range(20):
            b.on_step(rec(v=0.5, omega=0.2, t=i * 0.05))
            time.sleep(0.02)
        time.sleep(0.4)
        assert b.state is BridgeState.FAILSAFE, f"应 latch 到 FAILSAFE，得到 {b.state}"
        # FAILSAFE 之后必须只发 STOP
        after = srv.drives()[n_before + 5:]
        assert after, "FAILSAFE 后应仍在发 STOP"
        bad = [x for x in after if float(x.get("v", 0)) != 0.0 or float(x.get("w", 0)) != 0.0]
        assert not bad, f"FAILSAFE 后发了非零指令: {bad[:3]}"
        # latch：不会自动恢复
        srv.state.drive_ok = True
        for i in range(10):
            b.on_step(rec(v=0.9, omega=0.0, t=i * 0.05))
            time.sleep(0.02)
        time.sleep(0.2)
        assert b.state is BridgeState.FAILSAFE, "FAILSAFE 不应自动恢复"
        b.stop()
    finally:
        srv.close()


@check("T9 HTTP 409 -> BLOCKED_GUARD latch")
def t9():
    srv = MockServer()
    try:
        b = PikachuBridge(base_cfg(srv.url, max_failures=99))
        assert b.start()
        srv.state.guard_mode = True                # 运行中被打开 guard
        b.on_step(rec(v=0.5, omega=0.0))
        time.sleep(0.4)
        assert b.state is BridgeState.BLOCKED_GUARD, f"得到 {b.state}"
        b.stop()
    finally:
        srv.close()


# --------------------------------------------------------------------------- 10 退出
@check("T10 stop() 补发 STOP 且幂等")
def t10():
    srv = MockServer()
    try:
        b = PikachuBridge(base_cfg(srv.url))
        assert b.start()
        b.on_step(rec(v=0.5, omega=0.3))
        time.sleep(0.15)
        b.stop()
        stops = [x for x in srv.drives()
                 if float(x.get("v", 1)) == 0.0 and float(x.get("w", 1)) == 0.0]
        assert len(stops) >= 3, f"退出应补发至少 3 次 STOP，实际 {len(stops)}"
        n = len(srv.state.requests)
        b.stop()                                   # 幂等：不应再发任何请求
        b.stop()
        assert len(srv.state.requests) == n, "重复 stop() 不应再发请求"
    finally:
        srv.close()


@check("T11 服务不可达时 on_step 不抛异常，最终 latch")
def t11():
    b = PikachuBridge(base_cfg("http://127.0.0.1:9", timeout_s=0.05, max_failures=3))
    assert not b.start(), "不可达时预检应失败"
    # 即使没启动，on_step 也不能抛
    for i in range(50):
        b.on_step(rec(v=0.9, omega=0.5, t=i * 0.02))
    b.stop()
    b.stop()
    assert b.state is BridgeState.STOPPED


@check("T12 dry-run 全程不发 HTTP，on_step 正常计数")
def t12():
    b = PikachuBridge(base_cfg("http://127.0.0.1:9", dry_run=True, log_path=""))
    assert b.start()
    assert b.state is BridgeState.RUNNING
    for i in range(30):
        b.on_step(rec(v=0.5, omega=0.1, t=i * 0.02))
    time.sleep(0.3)
    st = b.stats()
    assert st["on_step_calls"] == 30, st
    assert st["frames_ok"] > 0, "dry-run 应记录成功的发送帧"
    assert st["frames_failed"] == 0, st
    b.stop()


# --------------------------------------------------------------------------- 13 节拍
@check("T13 RealTimePacer：sim 1s ≈ wall 1s 且不累计漂移")
def t13():
    p = RealTimePacer(max_lag_s=0.5, enabled=True)
    t0 = time.perf_counter()
    for k in range(51):                            # 0.00 .. 1.00s，步长 0.02
        p.wait(k * 0.02)
    elapsed = time.perf_counter() - t0
    assert abs(elapsed - 1.0) < 0.08, f"1s 仿真应约等 1s 墙钟，实际 {elapsed:.3f}s"
    assert p.resyncs == 0, f"不应触发重对齐，得到 {p.resyncs}"
    # 关闭后不阻塞
    p2 = RealTimePacer(enabled=False)
    t1 = time.perf_counter()
    for k in range(51):
        p2.wait(k * 0.02)
    assert time.perf_counter() - t1 < 0.01, "enabled=False 不应阻塞"


@check("T14 RealTimePacer：落后过多时重对齐而不是追赶")
def t14():
    p = RealTimePacer(max_lag_s=0.05, enabled=True)
    time.sleep(0.5)                                # 人为落后 0.5s
    p.wait(0.0)
    assert p.resyncs == 1, f"应重对齐一次，得到 {p.resyncs}"
    # 重对齐后必须**正常按节拍**（睡满剩余时间），而不是立刻返回去追赶
    t0 = time.perf_counter()
    p.wait(0.1)
    elapsed = time.perf_counter() - t0
    assert 0.07 < elapsed < 0.15, \
        f"重对齐后 wait(0.1) 应睡约 0.1s（不追赶），实际 {elapsed:.3f}s"
    # 再连跑一段，确认没有残留的追赶行为
    t1 = time.perf_counter()
    for k in range(11, 21):
        p.wait(k * 0.01)
    elapsed2 = time.perf_counter() - t1
    assert 0.07 < elapsed2 < 0.15, f"后续 0.1s 仿真应约等 0.1s 墙钟，实际 {elapsed2:.3f}s"


@check("T15 RealTimePacer：实体模式(allow_resync=False) 不重对齐，原样返回 lag")
def t15():
    p = RealTimePacer(max_lag_s=0.05, enabled=True, allow_resync=False)
    time.sleep(0.4)                                # 人为落后 0.4s
    lag = p.wait(0.0)
    assert lag > 0.05, f"实体模式应把 lag 返回给调用方，得到 {lag}"
    assert p.resyncs == 0, f"实体模式不该静默重对齐，resyncs={p.resyncs}"
    assert p.max_lag_seen >= 0.39, f"应记录 max_lag_seen，得到 {p.max_lag_seen}"
    # 未超阈值时仍正常按节拍
    p2 = RealTimePacer(max_lag_s=0.25, enabled=True, allow_resync=False)
    t0 = time.perf_counter()
    for k in range(11):
        p2.wait(k * 0.01)
    el = time.perf_counter() - t0
    assert 0.07 < el < 0.16, f"未超阈值应正常节拍，实际 {el:.3f}s"
    assert p2.resyncs == 0
    # dry-run 模式（allow_resync=True）才会重对齐
    p3 = RealTimePacer(max_lag_s=0.05, enabled=True, allow_resync=True)
    time.sleep(0.3)
    p3.wait(0.0)
    assert p3.resyncs == 1, "dry-run 模式应重对齐"


@check("T16 pacer 落后超限 -> enter_failsafe，之后只发 STOP 且不自动恢复")
def t16():
    srv = MockServer()
    try:
        b = PikachuBridge(base_cfg(srv.url, max_failures=99))
        assert b.start()
        b.on_step(rec(v=0.5, omega=0.0))
        time.sleep(0.15)
        n0 = len(srv.drives())
        b.note_lag(0.31)
        b.enter_failsafe("pacer 落后 310ms > 250ms")
        assert b.state is BridgeState.FAILSAFE, f"得到 {b.state}"
        st = b.stats()
        assert abs(st["max_lag_ms"] - 310.0) < 1e-6, st
        time.sleep(0.4)
        after = srv.drives()[n0:]
        assert after, "FAILSAFE 后应仍在发 STOP"
        bad = [x for x in after
               if float(x.get("v", 0)) != 0.0 or float(x.get("w", 0)) != 0.0]
        assert not bad, f"FAILSAFE 后发了非零指令: {bad[:3]}"
        # 即使一切恢复正常也不自动恢复运动
        for i in range(10):
            b.on_step(rec(v=0.9, omega=0.0, t=i * 0.02))
            time.sleep(0.02)
        time.sleep(0.15)
        assert b.state is BridgeState.FAILSAFE, "FAILSAFE 不应自动恢复"
        b.stop()
    finally:
        srv.close()


@check("T17 status.open=true 时不得调用 /api/reconnect")
def t17():
    srv = MockServer()
    try:
        assert srv.state.serial_open is True
        b = PikachuBridge(base_cfg(srv.url, preflight_timeout_s=2.0))
        assert b.start(), "串口已开时预检应直接通过"
        paths = [p for p, _ in srv.state.requests]
        assert "/api/reconnect" not in paths, \
            f"串口已打开却调用了 reconnect: {paths}"
        assert "/api/drive" in paths, "仍应发一次 STOP 做验证"
        b.stop()
    finally:
        srv.close()


@check("T18 status.open=false -> reconnect 用 preflight 超时，且成功后要 settle")
def t18():
    srv = MockServer()
    try:
        srv.state.serial_open = False
        # 关键回归：reconnect 耗时 0.4s，远大于控制帧超时 0.08s，但小于 preflight 超时 2.0s。
        # 旧实现用 0.08s 会直接 TimeoutError（就是实机 smoke test 失败的原因）。
        srv.state.reconnect_delay_s = 0.4
        settle = 0.5
        b = PikachuBridge(base_cfg(srv.url, timeout_s=0.08,
                                   preflight_timeout_s=2.0, serial_settle_s=settle))
        assert b.start(), "用 preflight 超时应当成功（0.4s 延迟 > 0.08s 控制超时）"
        paths = [p for p, _, _ in srv.state.timeline]
        assert "/api/reconnect" in paths, "串口没开时必须 reconnect"
        t_rec_done = next(t1 for p, _, t1 in srv.state.timeline
                          if p == "/api/reconnect")
        t_stop = next(t0 for p, t0, _ in srv.state.timeline
                      if p == "/api/drive" and t0 > t_rec_done)
        gap = t_stop - t_rec_done
        assert gap >= settle * 0.8, \
            f"reconnect 成功后应等 {settle}s 再发 STOP，实际只等了 {gap:.3f}s"
        b.stop()
    finally:
        srv.close()


@check("T19 reconnect 超时/失败 -> 不动车，返回失败")
def t19():
    srv = MockServer()
    try:
        srv.state.serial_open = False
        srv.state.reconnect_delay_s = 2.0        # 超过下面的 preflight 超时
        b = PikachuBridge(base_cfg(srv.url, preflight_timeout_s=0.4))
        assert not b.start(), "reconnect 超时应拒绝启动"
        assert b.state is BridgeState.INIT, f"不应进入 RUNNING，得到 {b.state}"
        drives = [p for p, _, _ in srv.state.timeline if p == "/api/drive"]
        assert not drives, f"reconnect 失败后不该发任何 drive 指令，实际发了 {len(drives)} 次"
        b.stop()
    finally:
        srv.close()


@check("T20 stale fd：open=true 但写不进去 -> 自动补一次 reconnect 后恢复")
def t20():
    srv = MockServer()
    try:
        assert srv.state.serial_open is True
        srv.state.stale_fd = True        # 设备掉过 USB，但 /api/status 仍说 open=true
        b = PikachuBridge(base_cfg(srv.url, preflight_timeout_s=2.0,
                                   serial_settle_s=0.15))
        assert b.start(), "应通过补一次 reconnect 自我恢复"
        paths = [p for p, _ in srv.state.requests]
        assert paths.count("/api/reconnect") == 1, \
            f"应恰好补一次 reconnect，实际 {paths.count('/api/reconnect')} 次：{paths}"
        assert srv.state.stale_fd is False, "reconnect 后 stale 应被清除"
        b.stop()
    finally:
        srv.close()


@check("T21 stale fd 且 reconnect 也治不好 -> 拒绝启动，不发运动指令")
def t21():
    srv = MockServer()
    try:
        srv.state.stale_fd = True
        srv.state.reconnect_fixes_stale = False
        b = PikachuBridge(base_cfg(srv.url, preflight_timeout_s=2.0,
                                   serial_settle_s=0.1))
        assert not b.start(), "补 reconnect 也无效时应拒绝启动"
        assert b.state is BridgeState.INIT
        # 只能有 STOP(0,0)，不能有任何非零运动指令
        bad = [x for x in srv.drives()
               if float(x.get("v", 0)) != 0.0 or float(x.get("w", 0)) != 0.0]
        assert not bad, f"不该发运动指令，实际 {bad}"
    finally:
        srv.close()


@check("T22 死区抬升 min_cmd：非零命令幅度不低于它、比例不变、0 仍为 0")
def t22():
    b = PikachuBridge(base_cfg("http://127.0.0.1:1", min_cmd=0.6,
                               max_v=1.0, max_w=1.0, max_motor_mix=1.0))
    # 前进：小命令被抬到 0.6
    V, W, *_ = b.scale(0.09, 0.0)                 # 归一化 0.1
    assert abs(max(abs(V), abs(W)) - 0.6) < 1e-9, (V, W)
    assert V > 0 and abs(W) < 1e-12, (V, W)
    # 纯转：同样抬到 0.6，符号保持
    V, W, *_ = b.scale(0.0, -0.26)                # 归一化 -0.1
    assert abs(abs(W) - 0.6) < 1e-9 and W < 0, (V, W)
    # 方向比例不变：v:omega = 0.1:0.05 -> 抬升后仍 2:1
    V, W, *_ = b.scale(0.09, 0.13)
    assert abs(V / W - 2.0) < 1e-9, (V, W)
    assert abs(max(abs(V), abs(W)) - 0.6) < 1e-9, (V, W)
    assert max(abs(V + W), abs(V - W)) >= 0.6 - 1e-9, \
        "主导侧电机量应 >= min_cmd（不保证差速时两侧都超过起转门槛）"
    # 零输入 -> 零输出（抬升不能把 0 抬起来）
    V, W, *_ = b.scale(0.0, 0.0)
    assert V == 0.0 and W == 0.0
    # 已高于 min_cmd 的命令不被改动
    V, W, *_ = b.scale(0.9, 0.0)                  # 归一化 1.0
    assert abs(V - 1.0) < 1e-9, V
    # min_cmd=0 关闭抬升
    b0 = PikachuBridge(base_cfg("http://127.0.0.1:1", min_cmd=0.0))
    V, W, *_ = b0.scale(0.09, 0.0)
    assert abs(V - 0.1) < 1e-9, V


def main() -> int:
    for name, fn in TESTS:
        try:
            fn()
            RESULTS.append((name, True, ""))
        except Exception as exc:                                   # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
            import traceback
            traceback.print_exc()
    print("\n" + "=" * 70)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, msg in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"         {msg}")
    print("-" * 70)
    print(f"  {n_pass}/{len(RESULTS)} passed")
    print("=" * 70)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
