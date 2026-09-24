"""Pikachu Bot 影子执行器桥接。

把冻结仿真（tag P1-mapfix）输出的 (v, omega) 缩放成归一化 (V, W)，以固定频率
POST 到 Pikachu 的 Flask `/api/drive`。

**不修改任何冻结 P1 文件**——挂点是 loop.run() 里已有的 trace_fn（它每步给出
dec.v / dec.omega，即这一步真正下发给仿真车的指令）。

安全机制分层：
  L1 启动预检   /api/status(guard_mode=false) -> /api/reconnect(ok) -> /api/drive(0,0)
  L2 节流       固定 rate_hz + 覆盖式单槽邮箱（latest-wins）
  L3 超时       timeout_s < 1/rate_hz
  L4 熔断       连续失败 >= max_failures -> FAILSAFE（**latch**，只发 STOP，需人工重启）
  L5 钳位       max_v / max_w / max_motor_mix 三重 + 死区抬升 min_cmd（抬主导幅度、保比例）
  L6 斜率       slew_rate（第一阶段 0；STOP 永远绕过）
  L7 退出       幂等 stop()：清邮箱 -> 停线程 -> join -> 同步补发 2~3 次 STOP
  L8 guard      /api/drive 返回 409 -> BLOCKED_GUARD（**latch**）
  L9 日志       每发送帧一行 CSV（raw 与 sent 都记，便于量化保真损失）

`on_step()` 永不抛异常、永不阻塞——网络故障不会影响冻结的仿真器。
桥接彻底挂掉时，Nano 自己的 400ms command timeout 会让车停下（L0 冗余）。
"""
from __future__ import annotations

import csv
import json
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum


class BridgeState(str, Enum):
    INIT = "INIT"
    RUNNING = "RUNNING"
    FAILSAFE = "FAILSAFE"              # latch：只发 STOP
    BLOCKED_GUARD = "BLOCKED_GUARD"    # latch：Pikachu 处于 guard 模式，拒绝驱动
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


@dataclass
class PikachuConfig:
    base_url: str = "http://127.0.0.1:8000"
    endpoint: str = "/api/drive"
    rate_hz: float = 10.0          # 第一阶段 10Hz；改 15Hz 时把 timeout_s 降到 0.04~0.05
    # 三种超时严格分开：
    timeout_s: float = 0.08            # **只用于实时控制帧**，必须 < 1/rate_hz
    preflight_timeout_s: float = 2.0   # /api/status、/api/reconnect、验证用 STOP
    serial_settle_s: float = 1.2       # 真正 reconnect 之后的等待（DTR 会复位 Nano）
    max_failures: int = 5          # 连续失败熔断阈值（10Hz 下约 0.5s）

    # 缩放：先按仿真量程归一化，再乘 gain
    sim_max_speed: float = 0.9     # 从冻结 Config.car.max_speed 取
    sim_max_omega: float = 2.6     # 从冻结 Config.car.max_omega 取
    v_gain: float = 1.0
    w_gain: float = 1.0
    omega_sign: float = 1.0        # 实物左右接反时改 -1

    # 钳位
    max_v: float = 1.0
    max_w: float = 1.0
    max_motor_mix: float = 1.0     # Nano: motorA=V+W, motorB=V-W；限制单电机不超过它

    # 死区抬升：wheels-up 实测实体在归一化幅度 <0.60 时堵转（drive PWM 42/255）。
    # 任何**非零**命令的 max(|V|,|W|) 会被抬到 min_cmd，方向比例不变；0 仍为 0。
    # 注意：这只保证"主导侧"的混合电机量达到 min_cmd，差速时另一侧可以更低、
    # 仍可能低于单轮起转门槛。设 0 关闭。不要设 >max_v，否则抬升后被 max_v 削回又落回死区。
    # 0.60 是 wheels-up（空载）门槛；装到地面带负载后的最低启动值需单独重新标定，
    # 不要把 0.65 当成最终实体参数。
    min_cmd: float = 0.65

    slew_rate: float = 0.0         # 归一化单位/秒，0 = 关闭（第一阶段关闭）
    dry_run: bool = False
    log_path: str = ""
    require_serial_open: bool = True

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------- HTTP
def http_json(url: str, payload: dict | None = None, timeout: float = 0.08):
    """返回 (status, body)。HTTP 4xx/5xx 也返回 body，不抛。"""
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "Connection": "close"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw, status = resp.read(), resp.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    body = json.loads(raw) if raw else {}
    return status, body


# --------------------------------------------------------------------------- 邮箱
class _Mailbox:
    """覆盖式单槽。影子模式下过期指令没有价值，队列只会积压。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item = None

    def put(self, item) -> None:
        with self._lock:
            self._item = item

    def take(self):
        with self._lock:
            item, self._item = self._item, None
            return item

    def clear(self) -> None:
        with self._lock:
            self._item = None


LOG_FIELDS = ["t", "sim_x", "sim_y", "sim_theta", "sim_v", "sim_omega",
              "raw_V", "raw_W", "sent_V", "sent_W", "kind",
              "http_status", "ok", "latency_ms", "note", "state",
              "max_lag_ms", "lag_failures"]


# --------------------------------------------------------------------------- 预检
def _do_reconnect(base: str, cfg: "PikachuConfig", log) -> tuple[bool, str]:
    """POST /api/reconnect 并用 preflight 超时等它。成功则等 serial_settle_s。"""
    try:
        _, body = http_json(base + "/api/reconnect", {}, cfg.preflight_timeout_s)
    except Exception as exc:                                   # noqa: BLE001
        return False, f"POST /api/reconnect 失败: {type(exc).__name__}: {exc}"
    s = body.get("status") or {}
    if not body.get("ok") or not s.get("open"):
        return False, (f"reconnect 未成功: ok={body.get('ok')} "
                       f"open={s.get('open')} err={s.get('last_error')}")
    log(f"[preflight] reconnect ok  port={s.get('port')}  "
        f"等串口稳定 {cfg.serial_settle_s:.1f}s（DTR 可能已复位 Nano）")
    time.sleep(cfg.serial_settle_s)
    return True, ""


def _try_stop(base: str, cfg: "PikachuConfig") -> tuple[bool, str]:
    """发 STOP(0,0) 做端到端验证。返回 (ok, 最后一次失败原因)。"""
    last = "n/a"
    for _ in range(3):
        try:
            status, body = http_json(base + cfg.endpoint, {"v": 0.0, "w": 0.0},
                                     cfg.preflight_timeout_s)
        except Exception as exc:                               # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(0.2)
            continue
        s = body.get("status") or {}
        if status == 409:
            return False, "HTTP 409 guard mode"
        if not body.get("ok"):
            last = f"json ok=false (http {status}) err={s.get('last_error')!r}"
            time.sleep(0.2)
            continue
        if cfg.require_serial_open and not s.get("open"):
            last = f"serial not open ({s.get('last_error')})"
            time.sleep(0.2)
            continue
        return True, ""
    return False, last


def preflight(base_url: str, cfg: PikachuConfig, log=print):
    """启动预检（Bridge.start() 和 smoke_test 共用，避免两处逻辑跑偏）。

    返回 (ok: bool, message: str)。除了验证用的 STOP(0,0) 之外不发任何运动指令。

    流程：
      1. GET /api/status（preflight_timeout_s）-> guard_mode 必须 false
      2. status.open == true  -> 先**不** reconnect（串口已开着，reconnect 既没必要、
            又可能因 DTR 复位把 CH340 卡住）
         status.open == false -> POST /api/reconnect，成功后等 serial_settle_s
      3. 发一次 STOP 做端到端验证
      4. **stale fd 兜底**：若第 2 步跳过了 reconnect、但第 3 步写不进去，说明设备掉过
         USB 而 pyserial 仍认为 is_open=True（实测 CH340 会反复掉线重枚举）。此时补一次
         reconnect 再验；仍失败才拒绝启动。

    实时发送帧仍然用 command_timeout_s（默认 0.08s）；这里不做全局放宽。
    """
    base = base_url.rstrip("/")

    # 1) status
    try:
        _, body = http_json(base + "/api/status", None, cfg.preflight_timeout_s)
    except Exception as exc:                                   # noqa: BLE001
        return False, f"GET /api/status 失败: {type(exc).__name__}: {exc}"
    if body.get("guard_mode"):
        return False, "guard_mode=true，先在网页上关闭 guard 模式"
    already_open = bool(body.get("open"))
    log(f"[preflight] /api/status ok  guard_mode=false  serial_open={already_open}")

    # 2) 只在串口没开时才 reconnect
    if already_open:
        log("[preflight] 串口已打开 -> 跳过 /api/reconnect")
    else:
        ok, msg = _do_reconnect(base, cfg, log)
        if not ok:
            return False, msg

    # 3) STOP 端到端验证
    ok, last = _try_stop(base, cfg)
    if ok:
        log("[preflight] STOP confirmed")
        return True, ""

    # 4) stale fd 兜底：跳过过 reconnect 才需要
    if already_open:
        log(f"[preflight] STOP 失败（{last}）且之前跳过了 reconnect -> 补一次 reconnect")
        ok, msg = _do_reconnect(base, cfg, log)
        if not ok:
            return False, f"STOP 失败（{last}）；补 reconnect 也失败: {msg}"
        ok, last = _try_stop(base, cfg)
        if ok:
            log("[preflight] reconnect 后 STOP confirmed（stale fd 已恢复）")
            return True, ""
        return False, f"reconnect 后 STOP 仍失败: {last}"

    return False, f"STOP 验证失败: {last}"


class PikachuBridge:
    def __init__(self, cfg: PikachuConfig):
        self.cfg = cfg
        self.state = BridgeState.INIT
        self._mailbox = _Mailbox()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._slew_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._log_fh = None
        self._log_w = None
        self._last_out = (0.0, 0.0)
        self._last_out_t = 0.0

        # 统计
        self.frames_ok = 0
        self.frames_failed = 0
        self.fails_now = 0
        self.stops_ok = 0
        self.last_latency_ms = 0.0
        self.last_error = ""
        self.last_raw = (0.0, 0.0)
        self.last_sent = (0.0, 0.0)
        self.mix_clamp_events = 0
        self.slew_events = 0
        self.mailbox_overwrites = 0
        self._put_count = 0
        # pacer 落后统计（实体运动期间落后是危险信号：实体已经比仿真多执行了旧命令）
        self.max_lag_ms = 0.0
        self.lag_failures = 0

    # ---- 缩放 + 钳位 --------------------------------------------------------
    def scale(self, sim_v: float, sim_omega: float):
        """返回 (V, W, raw_V, raw_W, mix_clamped)。

        raw_* 是**归一化并乘 gain 之后、任何钳位之前**的目标值——日志里记它，
        就能看出 max_v/max_w/mix 到底削掉了多少。
        """
        cfg = self.cfg
        raw_v = sim_v / cfg.sim_max_speed * cfg.v_gain
        raw_w = sim_omega / cfg.sim_max_omega * cfg.w_gain * cfg.omega_sign
        V = min(max(raw_v, -cfg.max_v), cfg.max_v)
        W = min(max(raw_w, -cfg.max_w), cfg.max_w)
        # 死区抬升（放在 max_v/max_w 钳位之后，避免先抬后削）。只改幅度、不改方向比例。
        # 边界：max(|V+W|,|V-W|) >= max(|V|,|W|) 只保证"较主导的一侧"混合电机量 >= min_cmd，
        # 不保证差速时两个轮子都超过单轮起转门槛（另一侧可以更低）。
        # 抬升后 max(|V|,|W|)==min_cmd<=max_v，不会越界。
        if cfg.min_cmd > 0.0:
            mag = max(abs(V), abs(W))
            if 0.0 < mag < cfg.min_cmd:
                k = cfg.min_cmd / mag
                V *= k
                W *= k
        # Nano 内部 motorA = V + W, motorB = V - W，所以要单独限制电机量
        m = max(abs(V + W), abs(V - W))
        clamped = False
        if cfg.max_motor_mix > 0 and m > cfg.max_motor_mix:
            s = cfg.max_motor_mix / m
            V *= s
            W *= s
            clamped = True
        return V, W, raw_v, raw_w, clamped

    def _apply_slew(self, V: float, W: float):
        cfg = self.cfg
        now = time.perf_counter()
        with self._slew_lock:
            dt = max(1e-3, now - self._last_out_t) if self._last_out_t else 1.0 / cfg.rate_hz
            step = cfg.slew_rate * dt
            pV, pW = self._last_out
            nV = min(max(V, pV - step), pV + step)
            nW = min(max(W, pW - step), pW + step)
            if abs(nV - V) > 1e-9 or abs(nW - W) > 1e-9:
                self.slew_events += 1
            self._last_out = (nV, nW)
            self._last_out_t = now
            return nV, nW

    # ---- 状态 ---------------------------------------------------------------
    def _set_state(self, st: BridgeState) -> bool:
        with self._state_lock:
            if self.state in (BridgeState.STOPPED, BridgeState.STOPPING):
                return False
            self.state = st
            return True

    def _latch(self, st: BridgeState, msg: str) -> None:
        with self._state_lock:
            if self.state != BridgeState.RUNNING:
                return
            self.state = st
        self._mailbox.clear()
        if st is BridgeState.FAILSAFE:
            print(f"\n[bridge] *** FAILSAFE (latched) *** {msg}")
            print("[bridge] 只发送 STOP，不再接受运动指令。"
                  "排查网络/服务后需重新启动 shadow_run。")
        else:
            print(f"\n[bridge] *** BLOCKED_GUARD (latched) *** {msg}")
            print("[bridge] Pikachu 处于 guard 模式，/api/drive 被拒绝。"
                  "先关闭 guard 再重新启动 shadow_run。")

    def enter_failsafe(self, reason: str) -> None:
        """从外部（例如 pacer 落后超限）把 bridge 打进 FAILSAFE。幂等，只从 RUNNING 生效。"""
        self._latch(BridgeState.FAILSAFE, reason)

    def note_lag(self, lag_s: float) -> None:
        """记录 pacer 的墙钟落后量（秒）。正数=仿真落后于墙钟。"""
        ms = max(0.0, lag_s) * 1e3
        if ms > self.max_lag_ms:
            self.max_lag_ms = ms

    # ---- 启动 ---------------------------------------------------------------
    def start(self) -> bool:
        cfg = self.cfg
        if cfg.log_path:
            self._open_log()
        if cfg.dry_run:
            print("[bridge] dry-run：不发送任何 HTTP 请求（只记录将要发送的 V/W）")
            self._set_state(BridgeState.RUNNING)
            self._spawn()
            return True

        ok, msg = preflight(cfg.base_url, cfg,
                            log=lambda s: print(f"[bridge] {s}"))
        if not ok:
            print(f"[bridge] 预检失败：{msg}")
            return False
        self.stops_ok += 1
        print("[bridge] -> RUNNING")
        self._set_state(BridgeState.RUNNING)
        self._spawn()
        return True

    def _spawn(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sender_loop, name="pikachu-sender",
                                        daemon=True)
        self._thread.start()

    # ---- 每步回调（仿真线程） -----------------------------------------------
    def on_step(self, rec: dict) -> None:
        """给 frozen run() 的 trace_fn 用。**永不抛异常、永不阻塞**。"""
        try:
            with self._state_lock:
                if self.state is not BridgeState.RUNNING:
                    return
            V, W, raw_v, raw_w, clamped = self.scale(
                float(rec.get("v", 0.0)), float(rec.get("omega", 0.0)))
            if clamped:
                self.mix_clamp_events += 1
            self.last_raw = (raw_v, raw_w)
            self._put_count += 1
            self._mailbox.put({
                "V": V, "W": W, "raw": (raw_v, raw_w),
                "t": float(rec.get("t", 0.0)),
                "x": float(rec.get("x", math.nan)),
                "y": float(rec.get("y", math.nan)),
                "theta": float(rec.get("theta", math.nan)),
                "sim_v": float(rec.get("v", 0.0)),
                "sim_omega": float(rec.get("omega", 0.0)),
            })
        except Exception as exc:                                   # noqa: BLE001
            self.last_error = f"on_step: {type(exc).__name__}: {exc}"

    # ---- 发送线程 -----------------------------------------------------------
    def _sender_loop(self) -> None:
        cfg = self.cfg
        period = 1.0 / cfg.rate_hz
        next_t = time.perf_counter()
        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_t:
                self._stop_event.wait(min(next_t - now, 0.02))
                continue
            next_t += period
            if next_t < now:                    # 落后了，重新对齐，避免突发追赶
                next_t = now + period

            with self._state_lock:
                st = self.state
            if st in (BridgeState.FAILSAFE, BridgeState.BLOCKED_GUARD):
                self._send(0.0, 0.0, kind="STOP", item=None)
                continue
            if st is not BridgeState.RUNNING:
                continue
            item = self._mailbox.take()
            if item is None:
                continue
            self._send(item["V"], item["W"], kind="DRIVE", item=item)

    def _send(self, V: float, W: float, kind: str, item: dict | None) -> None:
        cfg = self.cfg
        if kind == "DRIVE" and cfg.slew_rate > 0:
            V, W = self._apply_slew(V, W)
        elif kind == "STOP":
            # STOP 永远绕过 slew，立即置 0
            with self._slew_lock:
                self._last_out = (0.0, 0.0)
                self._last_out_t = time.perf_counter()

        payload = {"v": round(float(V), 2), "w": round(float(W), 2)}
        t0 = time.perf_counter()
        status, body, err = 0, {}, ""
        if cfg.dry_run:
            status, body = 200, {"ok": True, "status": {"open": True}}
        else:
            try:
                status, body = http_json(cfg.base_url.rstrip("/") + cfg.endpoint,
                                         payload, cfg.timeout_s)
            except Exception as exc:                               # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
        latency = (time.perf_counter() - t0) * 1e3
        self.last_latency_ms = latency

        ok, note = False, err
        if err:
            pass
        elif status == 409:
            note = "HTTP 409 guard mode"
            self._latch(BridgeState.BLOCKED_GUARD, note)
        else:
            st = body.get("status") or {}
            ok = bool(body.get("ok"))
            if not ok:
                note = f"json ok=false (http {status})"
            elif cfg.require_serial_open and not st.get("open"):
                ok = False
                note = f"serial not open (http {status}) err={st.get('last_error')}"

        if ok:
            self.frames_ok += 1
            self.fails_now = 0
            if kind == "STOP":
                self.stops_ok += 1
            else:
                self.last_sent = (V, W)
        else:
            self.frames_failed += 1
            self.fails_now += 1
            self.last_error = note
            if self.fails_now >= cfg.max_failures:
                self._latch(BridgeState.FAILSAFE,
                            f"连续 {self.fails_now} 次发送失败：{note}")

        self._log_row(item, V, W, kind, status, ok, latency, note)

    # ---- 退出 ---------------------------------------------------------------
    def stop(self, reason: str = "") -> None:
        """幂等。顺序：STOPPING -> 清邮箱 -> 停线程 -> join -> 补发 STOP -> STOPPED。"""
        with self._state_lock:
            if self.state in (BridgeState.STOPPING, BridgeState.STOPPED):
                return
            self.state = BridgeState.STOPPING
        if reason:
            print(f"[bridge] stopping: {reason}")
        self._mailbox.clear()
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if not self.cfg.dry_run:
            base = self.cfg.base_url.rstrip("/")
            for _ in range(3):
                try:
                    _, body = http_json(base + self.cfg.endpoint,
                                        {"v": 0.0, "w": 0.0}, self.cfg.timeout_s)
                    if body.get("ok"):
                        self.stops_ok += 1
                except Exception:                                  # noqa: BLE001
                    pass
                time.sleep(0.05)
        with self._state_lock:
            self.state = BridgeState.STOPPED
        self._close_log()
        print(f"[bridge] stopped  state={self.state.value}  ok={self.frames_ok} "
              f"failed={self.frames_failed} stops={self.stops_ok} "
              f"mix_clamp={self.mix_clamp_events} last_latency={self.last_latency_ms:.1f}ms")

    def __enter__(self) -> "PikachuBridge":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop(reason=f"exit ({exc_type.__name__})" if exc_type else "exit")
        return False

    # ---- 日志 ---------------------------------------------------------------
    def _open_log(self) -> None:
        self._log_fh = open(self.cfg.log_path, "w", newline="", encoding="utf-8")
        self._log_w = csv.DictWriter(self._log_fh, fieldnames=LOG_FIELDS,
                                     extrasaction="ignore")
        self._log_w.writeheader()

    def _close_log(self) -> None:
        with self._log_lock:
            if self._log_fh is not None:
                try:
                    self._log_fh.flush()
                    self._log_fh.close()
                except Exception:                                  # noqa: BLE001
                    pass
                self._log_fh = None
                self._log_w = None

    def _log_row(self, item, V, W, kind, status, ok, latency, note) -> None:
        w, fh = self._log_w, self._log_fh
        if w is None or fh is None:
            return
        it = item or {}
        row = {
            "t": it.get("t", ""),
            "sim_x": it.get("x", ""), "sim_y": it.get("y", ""),
            "sim_theta": it.get("theta", ""),
            "sim_v": it.get("sim_v", ""), "sim_omega": it.get("sim_omega", ""),
            "raw_V": it.get("raw", ("", ""))[0] if item else 0.0,
            "raw_W": it.get("raw", ("", ""))[1] if item else 0.0,
            "sent_V": round(V, 4), "sent_W": round(W, 4),
            "kind": kind, "http_status": status, "ok": int(bool(ok)),
            "latency_ms": round(latency, 2), "note": note,
            "state": self.state.value,
            "max_lag_ms": round(self.max_lag_ms, 1),
            "lag_failures": self.lag_failures,
        }
        with self._log_lock:
            try:
                w.writerow(row)
                fh.flush()
            except Exception:                                      # noqa: BLE001
                pass

    # ---- 诊断 ---------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "state": self.state.value,
            "frames_ok": self.frames_ok,
            "frames_failed": self.frames_failed,
            "stops_ok": self.stops_ok,
            "fails_now": self.fails_now,
            "mix_clamp_events": self.mix_clamp_events,
            "slew_events": self.slew_events,
            "on_step_calls": self._put_count,
            "last_latency_ms": round(self.last_latency_ms, 2),
            "max_lag_ms": round(self.max_lag_ms, 1),
            "lag_failures": self.lag_failures,
            "last_raw": self.last_raw,
            "last_sent": self.last_sent,
            "last_error": self.last_error,
        }
