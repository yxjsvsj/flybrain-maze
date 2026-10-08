"""P2b：ToF 遥测客户端（Windows 侧）——接收 Pi 的 10Hz UDP compact JSON。

端口约定：odom 8888，ToF 8889（互相独立）。UDP latest-wins，不换 TCP、不加缓冲。

一帧一个 datagram：
  {"ver":2,"session_id":"...","seq":..,"t":..,"healthy":true,
   "ranges":{...}, "status":{...}, "ages_ms":{...},
   "init_count":..,"read_errors":..,"reinit_count":..}

语义
----
- age 用 Windows 本地 time.monotonic() 的接收时刻算，不用 Pi 的 t。
- 同一 session 内 seq 必须递增：seq==last -> duplicate；seq<last -> out_of_order；
  seq 大跳 -> missing_by_gap（缺的个数）。**session_id 变化 -> 立即 SOURCE_RESET**。
- 两级 stale：
    age > warning_age(0.25s)            -> 只记 warning，不停车
    age > hard_stale(0.35s)             -> TofStaleError（上层 FAILSAFE latch）
    healthy == false / session 变化      -> 立即 TofUnhealthyError / TofSourceResetError
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass


class TofError(RuntimeError):
    pass


class TofStaleError(TofError):
    pass


class TofUnhealthyError(TofError):
    pass


class TofSourceResetError(TofError):
    pass


@dataclass(frozen=True)
class TofFrame:
    session_id: str
    seq: int
    t: float
    healthy: bool
    ranges: dict
    status: dict
    ages_ms: dict
    init_count: int
    read_errors: int
    reinit_count: int


class Tof5Client:
    def __init__(self, port: int = 8889, warning_age: float = 0.25,
                 hard_stale: float = 0.35, bind: str = "0.0.0.0",
                 recv_timeout: float = 0.2):
        self.warning_age = float(warning_age)
        self.hard_stale = float(hard_stale)
        # 兼容旧调用
        self.max_age = self.hard_stale
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind, int(port)))
        self._sock.settimeout(float(recv_timeout))
        self._lock = threading.Lock()
        self._frame: TofFrame | None = None
        self._recv_t: float | None = None
        self._session: str | None = None
        self._last_seq: int | None = None
        self._reset: tuple[str, str] | None = None
        self._last_pi_t: float | None = None
        self._last_rx_t: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # ---- UDP 统计（拆分口径）----
        self.rx_total = 0
        self.accepted = 0
        self.duplicate_seq = 0
        self.out_of_order_seq = 0
        self.missing_by_gap = 0
        self.bad_json = 0
        self.session_reset = 0
        # ---- 接收侧间隔（diagnostic）----
        self.ia_pi: list = []      # Pi 帧间隔（t 差）
        self.ia_rx: list = []      # Windows 接收间隔
        # ---- 两级 stale 计数 ----
        self.warning_count = 0     # age 落入 warning 带的次数（每次 latest() 调用）
        self.hard_stale_count = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._rx, name="tof5-rx", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "Tof5Client":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _rx(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break
            with self._lock:
                self.rx_total += 1
            try:
                o = json.loads(data)
                fr = TofFrame(
                    session_id=str(o["session_id"]), seq=int(o["seq"]),
                    t=float(o["t"]), healthy=bool(o["healthy"]),
                    ranges=dict(o["ranges"]), status=dict(o["status"]),
                    ages_ms=dict(o.get("ages_ms", {})),
                    init_count=int(o.get("init_count", 0)),
                    read_errors=int(o.get("read_errors", 0)),
                    reinit_count=int(o.get("reinit_count", 0)))
            except (ValueError, KeyError, TypeError):
                with self._lock:
                    self.bad_json += 1
                continue
            now = time.monotonic()
            with self._lock:
                if self._session is not None and fr.session_id != self._session:
                    if self._reset is None:
                        self._reset = (self._session, fr.session_id)
                    self.session_reset += 1
                    continue
                if self._session is None:
                    self._session = fr.session_id
                if self._last_seq is not None:
                    if fr.seq == self._last_seq:
                        self.duplicate_seq += 1
                        continue
                    if fr.seq < self._last_seq:
                        self.out_of_order_seq += 1
                        continue
                    if fr.seq > self._last_seq + 1:
                        self.missing_by_gap += fr.seq - self._last_seq - 1
                if self._last_rx_t is not None:
                    self.ia_rx.append(round(now - self._last_rx_t, 4))
                    if self._last_pi_t is not None:
                        self.ia_pi.append(round(fr.t - self._last_pi_t, 4))
                self._last_pi_t = fr.t
                self._last_rx_t = now
                self._last_seq = fr.seq
                self._frame = fr
                self._recv_t = now
                self.accepted += 1

    def latest(self) -> tuple[TofFrame, float]:
        with self._lock:
            fr, rt, reset = self._frame, self._recv_t, self._reset
        if reset is not None:
            raise TofSourceResetError(f"session 变化 {reset[0]} -> {reset[1]}（producer 重启）")
        if fr is None or rt is None:
            raise TofStaleError("还没收到任何 ToF 帧")
        if not fr.healthy:
            raise TofUnhealthyError("ToF healthy=false（地址丢失/自愈失败）")
        age = time.monotonic() - rt
        if age > self.hard_stale:
            self.hard_stale_count += 1
            raise TofStaleError(f"ToF age {age*1000:.0f}ms > hard_stale "
                                f"{self.hard_stale*1000:.0f}ms")
        if age > self.warning_age:
            self.warning_count += 1
        return fr, age

    def age(self) -> float | None:
        with self._lock:
            rt = self._recv_t
        return None if rt is None else time.monotonic() - rt

    def metrics(self) -> dict:
        with self._lock:
            return {
                "rx_total": self.rx_total, "accepted_new": self.accepted,
                "duplicate_seq": self.duplicate_seq,
                "out_of_order_seq": self.out_of_order_seq,
                "missing_by_gap": self.missing_by_gap, "bad_json": self.bad_json,
                "session_reset": self.session_reset,
                "last_seq": self._last_seq, "session": self._session,
                "warning_count": self.warning_count,
                "hard_stale_count": self.hard_stale_count,
                "ia_pi": list(self.ia_pi), "ia_rx": list(self.ia_rx),
            }

    def stats(self) -> dict:
        with self._lock:
            return {"received": self.accepted, "dropped": self.duplicate_seq + self.out_of_order_seq,
                    "bad": self.bad_json, "last_seq": self._last_seq, "reset": self._reset}


def _monitor(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="ToF UDP 监听/测速")
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--warning-age", type=float, default=0.25)
    ap.add_argument("--hard-stale", type=float, default=0.35)
    args = ap.parse_args(argv)

    c = Tof5Client(args.port, warning_age=args.warning_age,
                   hard_stale=args.hard_stale, bind=args.bind)
    c.start()
    print(f"listening ToF udp :{args.port} for {args.seconds:g}s ...", flush=True)
    t0 = time.time()
    last = 0
    try:
        while time.time() - t0 < args.seconds:
            time.sleep(1.0)
            m = c.metrics()
            rate = m["accepted_new"] - last
            last = m["accepted_new"]
            print(f"t={time.time()-t0:4.1f}s acc={m['accepted_new']} rate={rate:.1f}/s "
                  f"dup={m['duplicate_seq']} ooo={m['out_of_order_seq']} "
                  f"gap={m['missing_by_gap']} warn={m['warning_count']} "
                  f"hard={m['hard_stale_count']}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        c.stop()
    print("final metrics:", c.metrics())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_monitor())
