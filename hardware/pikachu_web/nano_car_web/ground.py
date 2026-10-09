"""P2c-0 Ground Control：网页人工调试/监视（独立于 frozen P1）。

权限模型
--------
  MANUAL  网页方向键可用；/api/drive（AUTO/Windows 链路）被 403 联锁挡死。
  AUTO    physical_run / Windows 链路可用；网页方向键禁用。
  E-STOP  latch，优先级最高：一旦触发，MANUAL 与 /api/drive 全部挡死，只能手动清除。

状态来源
--------
  读 Pi 本机 sender 原子写的共享 JSON（odometry.py / tof5.py 的 --state-file）：
    /run/pikachu/odom.json   位姿 + ticks + session_id + wall 时间戳
    /run/pikachu/tof.json    五路 mm/status + ages + healthy + wall
  本模块**不碰 GPIO/I2C、不碰串口驱动进程**。

关于 /api/drive
--------------
  原语义（normalized V/W、返回 status）一字未改；仅新增一个**默认放行的安全联锁**：
  只有在 MANUAL 模式或 E-STOP latch 时才 403。默认 AUTO 下行为与以前完全一致。
"""
from __future__ import annotations

import json
import os
import threading
import time

from flask import jsonify, render_template, request

MANUAL = "MANUAL"
AUTO = "AUTO"

ODOM_FILE = os.environ.get("GROUND_ODOM_STATE", "/run/pikachu/odom.json")
TOF_FILE = os.environ.get("GROUND_TOF_STATE", "/run/pikachu/tof.json")
ODOM_STALE_S = float(os.environ.get("GROUND_ODOM_STALE", "0.35"))
TOF_STALE_S = float(os.environ.get("GROUND_TOF_STALE", "0.45"))
HEARTBEAT_TIMEOUT_S = float(os.environ.get("GROUND_HB_TIMEOUT", "6.0"))
WALL_RATE_HZ = 20.0


def _read_json(path):
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


class GroundControl:
    """人工调试的权限/状态/单墙安全。线程安全。"""

    def __init__(self, controller) -> None:
        self.ctrl = controller
        self._lock = threading.Lock()
        self._mode = AUTO
        self._estop = False
        self._estop_reason = ""
        self._hb = 0.0
        self._wall_stop = None
        self._wall_state = {"running": False, "reason": "", "stop_mm": None}
        threading.Thread(target=self._watch, daemon=True).start()

    # ------------------------------------------------------------------ 权限
    def authority(self) -> dict:
        with self._lock:
            age = (time.monotonic() - self._hb) if self._hb else None
            return {"mode": self._mode, "estop": self._estop,
                    "estop_reason": self._estop_reason, "manual_heartbeat_age": age}

    def set_mode(self, mode) -> bool:
        mode = (mode or "").upper()
        if mode not in (MANUAL, AUTO):
            return False
        with self._lock:
            self._mode = mode
            self._hb = time.monotonic()
        return True

    def heartbeat(self) -> None:
        with self._lock:
            self._hb = time.monotonic()

    def estop(self, reason: str = "manual") -> None:
        with self._lock:
            self._estop = True
            self._estop_reason = reason
        self.abort_wall()
        for _ in range(6):
            try:
                self.ctrl.send_drive(0.0, 0.0)
            except Exception:                                      # noqa: BLE001
                pass
            time.sleep(0.03)

    def clear_estop(self) -> None:
        with self._lock:
            self._estop = False
            self._estop_reason = ""

    def drive_blocked(self) -> bool:
        """给 /api/drive 的联锁：MANUAL 或 E-STOP 时挡。"""
        with self._lock:
            return self._estop or self._mode == MANUAL

    def allow_manual(self) -> bool:
        with self._lock:
            return self._mode == MANUAL and not self._estop

    def _watch(self) -> None:
        while True:
            time.sleep(0.5)
            with self._lock:
                if (self._mode == MANUAL and self._hb
                        and (time.monotonic() - self._hb) > HEARTBEAT_TIMEOUT_S):
                    self._mode = AUTO

    # ------------------------------------------------------------------ 状态
    def snapshot(self) -> dict:
        now = time.time()
        od = _read_json(ODOM_FILE)
        tf = _read_json(TOF_FILE)
        o_age = None if not od or "wall" not in od else round(now - od["wall"], 3)
        t_age = None if not tf or "wall" not in tf else round(now - tf["wall"], 3)
        o_ok = o_age is not None and o_age < ODOM_STALE_S
        t_status = (tf or {}).get("status") or {}
        t_io = any(s == "IO_ERROR" for s in t_status.values())
        t_ok = (t_age is not None and t_age < TOF_STALE_S
                and bool((tf or {}).get("healthy")) and not t_io)
        return {
            "authority": self.authority(),
            "serial": self.ctrl.status(),
            "odom": {"ok": o_ok, "age": o_age, **(od or {})},
            "tof": {"ok": t_ok, "age": t_age, "io_error": t_io, **(tf or {})},
            "streams_ok": bool(o_ok and t_ok),
            "wall_run": dict(self._wall_state),
        }

    # ------------------------------------------------------------------ 驱动
    def manual_drive(self, v: float, w: float):
        with self._lock:
            if self._estop:
                return False, "estop_latched"
            if self._mode != MANUAL:
                return False, "manual_not_active"
        if not self.snapshot()["streams_ok"]:
            return False, "streams_stale"
        ok = self.ctrl.send_drive(v, w)
        return ok, ("" if ok else "serial_send_failed")

    def stop(self) -> bool:
        try:
            return self.ctrl.send_drive(0.0, 0.0)
        except Exception:                                          # noqa: BLE001
            return False

    # -------------------------------------------------------------- 单墙安全
    def start_wall(self, mag: float, stop_mm: float, max_s: float):
        with self._lock:
            if self._estop:
                return False, "estop_latched"
            if self._mode != MANUAL:
                return False, "manual_not_active"
        self.abort_wall()
        ev = threading.Event()
        with self._lock:
            self._wall_stop = ev
            self._wall_state = {"running": True, "reason": "", "stop_mm": stop_mm,
                                "mag": mag, "started": time.time()}
        threading.Thread(target=self._wall_loop,
                         args=(ev, float(mag), float(stop_mm), float(max_s)),
                         daemon=True).start()
        return True, ""

    def abort_wall(self) -> None:
        with self._lock:
            ev = self._wall_stop
        if ev is not None:
            ev.set()

    def _wall_loop(self, ev: threading.Event, mag: float, stop_mm: float, max_s: float) -> None:
        period = 1.0 / WALL_RATE_HZ
        t_end = time.monotonic() + max_s
        reason = "timeout"
        try:
            while time.monotonic() < t_end:
                if ev.is_set():
                    reason = "aborted"
                    break
                with self._lock:
                    if self._estop:
                        reason = "estop"
                        break
                snap = self.snapshot()
                if not snap["streams_ok"]:
                    reason = "unsafe: streams stale/unhealthy"
                    break
                fmm = (snap["tof"].get("ranges") or {}).get("F")
                fst = (snap["tof"].get("status") or {}).get("F")
                if fst == "TOO_NEAR" or (fst == "VALID" and fmm is not None and fmm <= stop_mm):
                    reason = (f"F {fmm:.0f}mm <= {stop_mm:.0f}" if fmm is not None
                              else "F TOO_NEAR")
                    break
                self.ctrl.send_drive(mag, 0.0)
                time.sleep(period)
        finally:
            self.stop()
            with self._lock:
                self._wall_state = {"running": False, "reason": reason,
                                    "stop_mm": stop_mm, "finished": time.time()}


def register_ground_routes(app, gc: GroundControl) -> None:
    @app.route("/ground")
    def ground_page():
        return render_template("ground.html")

    @app.route("/api/authority", methods=["GET", "POST"])
    def authority():
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            if "mode" in data and not gc.set_mode(data.get("mode")):
                return jsonify({"ok": False, "error": "mode must be MANUAL|AUTO",
                                **gc.authority()}), 400
        return jsonify({"ok": True, **gc.authority()})

    @app.route("/api/manual/state")
    def manual_state():
        return jsonify(gc.snapshot())

    @app.route("/api/manual/mode", methods=["POST"])
    def manual_mode():
        data = request.get_json(silent=True) or {}
        if not gc.set_mode(data.get("mode")):
            return jsonify({"ok": False, "error": "mode must be MANUAL|AUTO",
                            **gc.authority()}), 400
        return jsonify({"ok": True, **gc.authority()})

    @app.route("/api/manual/heartbeat", methods=["POST"])
    def manual_heartbeat():
        gc.heartbeat()
        return jsonify({"ok": True, **gc.authority()})

    @app.route("/api/manual/release", methods=["POST"])
    def manual_release():
        gc.stop()
        gc.set_mode(AUTO)
        return jsonify({"ok": True, **gc.authority()})

    @app.route("/api/manual/drive", methods=["POST"])
    def manual_drive():
        data = request.get_json(silent=True) or {}
        try:
            v = float(data.get("v", 0)); w = float(data.get("w", 0))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "bad v/w"}), 400
        ok, err = gc.manual_drive(v, w)
        return jsonify({"ok": ok, "error": err, "serial": gc.ctrl.status()}), (200 if ok else 403)

    @app.route("/api/manual/stop", methods=["POST"])
    def manual_stop():
        ok = gc.stop()
        return jsonify({"ok": ok, "serial": gc.ctrl.status()})

    @app.route("/api/manual/estop", methods=["POST"])
    def manual_estop():
        data = request.get_json(silent=True) or {}
        gc.estop(reason=str(data.get("reason") or "manual"))
        return jsonify({"ok": True, **gc.authority()})

    @app.route("/api/manual/clear-estop", methods=["POST"])
    def manual_clear_estop():
        gc.clear_estop()
        return jsonify({"ok": True, **gc.authority()})

    @app.route("/api/manual/wall", methods=["POST"])
    def manual_wall():
        data = request.get_json(silent=True) or {}
        try:
            mag = float(data.get("mag", 0.5))
            stop_mm = float(data.get("stop_mm", 450))
            max_s = float(data.get("max_s", 2.0))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "bad params"}), 400
        ok, err = gc.start_wall(mag, stop_mm, max_s)
        return jsonify({"ok": ok, "error": err, **gc.authority()}), (200 if ok else 403)

    @app.route("/api/manual/wall-stop", methods=["POST"])
    def manual_wall_stop():
        gc.abort_wall()
        gc.stop()
        return jsonify({"ok": True})

    @app.before_request
    def _drive_interlock():
        if request.method == "POST" and request.path == "/api/drive" and gc.drive_blocked():
            a = gc.authority()
            err = "estop_latched" if a["estop"] else "manual_mode_active"
            return jsonify({"ok": False, "error": err, "mode": a["mode"],
                            "estop": a["estop"], "status": gc.ctrl.status()}), 403
        return None
