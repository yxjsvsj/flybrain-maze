"""P2a.5/2b：里程计遥测客户端（Windows 侧）——接收 Pi 的 50Hz UDP compact JSON。

端口约定：odom 8888，ToF 8889。UDP latest-wins。

一帧一个 datagram：
  {"ver":2,"session_id":"<uuid>","seq":..,"t":..,"left":..,"right":..,
   "x":..,"y":..,"theta":..}
- 同 session 内只接受 seq 递增；seq==last -> duplicate；seq<last -> out_of_order；
  seq 大跳 -> missing_by_gap。
- **session_id 变化 -> 立即 ODOM_SOURCE_RESET**（不再用 seq backstep 猜测）。
- age 用 Windows 本地 time.monotonic() 接收时刻算。latest() 上 stale/reset 抛异常。
- **端口独占**：不设 SO_REUSEADDR；Windows 上设 SO_EXCLUSIVEADDRUSE，重复 listener
  直接报端口占用，禁止静默分流 UDP 包。
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass


class OdomError(RuntimeError):
    pass


class OdomStaleError(OdomError):
    pass


class OdomSourceResetError(OdomError):
    pass


@dataclass(frozen=True)
class OdomSample:
    session_id: str
    seq: int
    t: float
    left: int
    right: int
    x: float
    y: float
    theta: float


def _exclusive(sock: socket.socket) -> None:
    """Windows 独占端口（Linux 无此选项，忽略）。**
    不设 SO_REUSEADDR —— 重复绑定必须失败，而不是静默分走包。
    """
    opt = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
    if opt is not None:
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, 1)
        except OSError:
            pass


class OdomClient:
    def __init__(self, port: int, max_age: float = 0.25, bind: str = "0.0.0.0",
                 recv_timeout: float = 0.2):
        self.max_age = float(max_age)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _exclusive(self._sock)
        self._sock.bind((bind, int(port)))
        self._sock.settimeout(float(recv_timeout))
        self._lock = threading.Lock()
        self._sample: OdomSample | None = None
        self._recv_t: float | None = None
        self._session: str | None = None
        self._last_seq: int | None = None
        self._reset: tuple[str, str] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rx_total = 0
        self.accepted = 0
        self.duplicate_seq = 0
        self.out_of_order_seq = 0
        self.missing_by_gap = 0
        self.bad_json = 0
        self.source_ip = None
        self.source_port = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._rx, name="odom-rx", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "OdomClient":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _rx(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            with self._lock:
                self.rx_total += 1
                self.source_ip, self.source_port = addr[0], addr[1]
            try:
                o = json.loads(data)
                s = OdomSample(
                    session_id=str(o["session_id"]), seq=int(o["seq"]),
                    t=float(o["t"]), left=int(o["left"]), right=int(o["right"]),
                    x=float(o["x"]), y=float(o["y"]), theta=float(o["theta"]))
            except (ValueError, KeyError, TypeError):
                with self._lock:
                    self.bad_json += 1
                continue
            now = time.monotonic()
            with self._lock:
                if self._session is not None and s.session_id != self._session:
                    if self._reset is None:
                        self._reset = (self._session, s.session_id)
                    continue
                if self._session is None:
                    self._session = s.session_id
                if self._last_seq is not None:
                    if s.seq == self._last_seq:
                        self.duplicate_seq += 1
                        continue
                    if s.seq < self._last_seq:
                        self.out_of_order_seq += 1
                        continue
                    if s.seq > self._last_seq + 1:
                        self.missing_by_gap += s.seq - self._last_seq - 1
                self._last_seq = s.seq
                self._sample = s
                self._recv_t = now
                self.accepted += 1

    def latest(self) -> tuple[OdomSample, float]:
        with self._lock:
            s, rt, reset = self._sample, self._recv_t, self._reset
        if reset is not None:
            raise OdomSourceResetError(f"session 变化 {reset[0]} -> {reset[1]}")
        if s is None or rt is None:
            raise OdomStaleError("还没收到任何 odom 包")
        age = time.monotonic() - rt
        if age > self.max_age:
            raise OdomStaleError(f"odom age {age*1000:.0f}ms > {self.max_age*1000:.0f}ms")
        return s, age

    def age(self) -> float | None:
        with self._lock:
            rt = self._recv_t
        return None if rt is None else time.monotonic() - rt

    def metrics(self) -> dict:
        with self._lock:
            return {"rx_total": self.rx_total, "accepted": self.accepted,
                    "duplicate_seq": self.duplicate_seq,
                    "out_of_order_seq": self.out_of_order_seq,
                    "missing_by_gap": self.missing_by_gap, "bad_json": self.bad_json,
                    "last_seq": self._last_seq, "session": self._session,
                    "source": (self.source_ip, self.source_port), "reset": self._reset}

    def stats(self) -> dict:
        with self._lock:
            return {"received": self.accepted,
                    "dropped": self.duplicate_seq + self.out_of_order_seq,
                    "bad": self.bad_json, "last_seq": self._last_seq,
                    "reset": self._reset}
