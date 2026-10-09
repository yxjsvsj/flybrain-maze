"""P2c-0 地面调试工具（独立；不改 frozen P1、不跑 physical_run）。

模式
----
  pulse  短脉冲手动标定 forward/left/right：
         configurable magnitude（默认 0.80），action 默认 0.5s，STOP 默认 2s，repeat 次；
         Ctrl+C / 异常 / finally 一律补发多次 STOP。
  wall   单墙安全：低速 forward；F <= stop_mm（首测 450mm）立即 STOP；
         最大运行 2s；odom/ToF stale/unhealthy/source-reset 立即 FAILSAFE latch（只发 STOP）。

日志列（每采样一行）：
  t, V, W, F_mm, F_status, left, right, x, y, theta, stop_reason, age_odom_ms, age_tof_ms

用法：
  python -m hardware.ground_commission --url http://192.168.50.57:8000 \
      --mode pulse --only forward --mag 0.80 --repeat 5 --duration 0.5 --stop-dur 2
  python -m hardware.ground_commission --url http://192.168.50.57:8000 \
      --mode wall --wall-v 0.5 --stop-mm 450 --max-run-s 2
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time

from hardware.odometry_client import OdomClient, OdomError
from hardware.pikachu_bridge import http_json
from hardware.tof5_client import Tof5Client, TofError

DIRS = {"forward": (1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
LOG_COLS = ["t", "V", "W", "F_mm", "F_status", "left", "right", "x", "y", "theta",
            "stop_reason", "age_odom_ms", "age_tof_ms"]


class Failsafe(Exception):
    """安全层触发：立即 STOP + latch（不自动恢复）。"""


def post_drive(base, v, w, timeout):
    try:
        st, body = http_json(base + "/api/drive", {"v": float(v), "w": float(w)}, timeout)
    except Exception as exc:                                   # noqa: BLE001
        return False, f"{type(exc).__name__}:{exc}"
    s = body.get("status") or {}
    if not body.get("ok") or not s.get("open"):
        return False, f"http{st} ok={body.get('ok')} open={s.get('open')} err={s.get('last_error')}"
    return True, ""


class Runner:
    def __init__(self, args):
        self.a = args
        self.base = args.url.rstrip("/")
        self.odom = OdomClient(args.odom_port, max_age=args.max_odom_age)
        self.tof = Tof5Client(args.tof_port, warning_age=0.25, hard_stale=args.max_tof_age)
        self.odom.start()
        self.tof.start()
        self.fh = open(args.log, "w", newline="", encoding="utf-8") if args.log else None
        self.w = csv.DictWriter(self.fh, fieldnames=LOG_COLS) if self.fh else None
        if self.w:
            self.w.writeheader()
        self.t0 = time.time()
        self.fail = False
        self.sends_fail = 0

    # ---- 基础 ----
    def streams(self):
        """读两路，返回 (odom_sample, age_o, tof_frame, age_t)。stale/unhealthy/reset -> Failsafe。"""
        try:
            o, ao = self.odom.latest()
        except OdomError as exc:
            raise Failsafe(f"odom {type(exc).__name__}: {exc}") from exc
        try:
            t, at = self.tof.latest()
        except TofError as exc:
            raise Failsafe(f"tof {type(exc).__name__}: {exc}") from exc
        return o, ao, t, at

    def log(self, v, w, of, ao, tf, at, reason=""):
        if self.w is None or self.fh is None:
            return
        self.w.writerow({
            "t": round(time.time() - self.t0, 3), "V": v, "W": w,
            "F_mm": ("" if tf.ranges.get("F") is None else tf.ranges["F"]),
            "F_status": tf.status.get("F", ""),
            "left": of.left, "right": of.right,
            "x": round(of.x, 5), "y": round(of.y, 5), "theta": round(of.theta, 6),
            "stop_reason": reason,
            "age_odom_ms": round(ao * 1000, 1), "age_tof_ms": round(at * 1000, 1)})
        self.fh.flush()

    def send(self, v, w):
        ok, note = post_drive(self.base, v, w, self.a.timeout)
        if not ok:
            self.sends_fail += 1
            if self.sends_fail <= 3:
                print(f"    !! send fail: {note}")
        else:
            self.sends_fail = 0
        return ok

    def stop(self, n=4):
        for _ in range(n):
            post_drive(self.base, 0.0, 0.0, self.a.preflight_timeout)
            time.sleep(0.05)

    def tick(self, v, w, reason=""):
        o, ao, tf, at = self.streams()          # 可能抛 Failsafe
        self.log(v, w, o, ao, tf, at, reason)
        return tf

    def close(self):
        if self.fh:
            self.fh.close()
        self.odom.stop()
        self.tof.stop()


def run_pulse(r):
    names = list(DIRS) if r.a.only == "all" else [r.a.only]
    period = 1.0 / r.a.rate
    reason = ""
    for name in names:
        vs, ws = DIRS[name]
        for k in range(r.a.repeat):
            print(f">>> {name}#{k+1}  V={vs*r.a.mag:+.2f} W={ws*r.a.mag:+.2f}  {r.a.duration}s")
            t_end = time.time() + r.a.duration
            while time.time() < t_end:
                r.send(vs * r.a.mag, ws * r.a.mag)
                r.tick(vs * r.a.mag, ws * r.a.mag)
                time.sleep(period)
            t_end = time.time() + r.a.stop_dur
            while time.time() < t_end:
                r.send(0.0, 0.0)
                r.tick(0.0, 0.0)
                time.sleep(period)


def run_wall(r):
    print(f">>> wall: forward V={r.a.wall_v:.2f}, STOP when F<={r.a.stop_mm}mm "
          f"(max {r.a.max_run_s}s)")
    reason = "timeout"
    t_end = time.time() + r.a.max_run_s
    period = 1.0 / r.a.rate
    while time.time() < t_end:
        tf = r.tick(r.a.wall_v, 0.0)            # 读流 + 记录（可能 Failsafe）
        fmm = tf.ranges.get("F")
        fstatus = tf.status.get("F")
        if fstatus == "TOO_NEAR":
            reason = "F TOO_NEAR"
            break
        if fstatus == "VALID" and fmm is not None and fmm <= r.a.stop_mm:
            reason = f"F {fmm:.0f}mm <= {r.a.stop_mm}"
            break
        r.send(r.a.wall_v, 0.0)
        time.sleep(period)
    print(f"    STOP reason: {reason}")
    try:                                        # 记一行 stop_reason（尽力）
        r.log(0.0, 0.0, *r.streams(), reason)
    except (Failsafe, Exception):              # noqa: BLE001
        pass
    return reason


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2c-0 地面调试工具")
    ap.add_argument("--url", required=True, help="Pikachu Flask，如 http://192.168.50.57:8000")
    ap.add_argument("--mode", choices=["pulse", "wall"], required=True)
    ap.add_argument("--only", default="all", choices=["all", "forward", "left", "right"])
    ap.add_argument("--mag", type=float, default=0.80)
    ap.add_argument("--duration", type=float, default=0.5)
    ap.add_argument("--stop-dur", type=float, default=2.0)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--wall-v", type=float, default=0.5)
    ap.add_argument("--stop-mm", type=float, default=450.0)
    ap.add_argument("--max-run-s", type=float, default=2.0)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--timeout", type=float, default=0.08)
    ap.add_argument("--preflight-timeout", type=float, default=2.0)
    ap.add_argument("--odom-port", type=int, default=8888)
    ap.add_argument("--tof-port", type=int, default=8889)
    ap.add_argument("--max-odom-age", type=float, default=0.25)
    ap.add_argument("--max-tof-age", type=float, default=0.35)
    ap.add_argument("--log", default="ground_commission.csv")
    args = ap.parse_args(argv)

    r = Runner(args)
    stop_all = False

    def _sig(signum, _f):
        nonlocal stop_all
        stop_all = True
        print(f"\n[ground] 信号 {signum} -> STOP + 退出")
        r.stop(5)
        r.close()
        sys.exit(1)

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig)
        except (ValueError, OSError):
            pass

    # 等两路第一个包
    for name, cli in (("odom", r.odom), ("tof", r.tof)):
        t0 = time.time()
        while cli.age() is None:
            if time.time() - t0 > 5:
                print(f"[ground] 等不到 {name}；退出"); r.close(); return 1
            time.sleep(0.05)
    print("[ground] odom + tof ok\n")

    try:
        if args.mode == "pulse":
            run_pulse(r)
        else:
            run_wall(r)
    except Failsafe as exc:
        r.fail = True
        print(f"\n[ground] *** FAILSAFE: {exc} -> STOP 并 latch（不自动恢复）***")
    except KeyboardInterrupt:
        print("\n[ground] 中断")
    finally:
        r.stop(5)
        print(f"[ground] final STOP sent. sends_fail={r.sends_fail}")
        r.close()
    print(f"[ground] {'FAIL' if r.fail else 'done'}  log={args.log}")
    return 1 if r.fail else 0


if __name__ == "__main__":
    sys.exit(main())
