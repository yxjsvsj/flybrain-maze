"""P2b：ToF 遥测客户端（Windows 侧）——接收 Pi 的 10Hz UDP compact JSON。

端口约定：odom 8888，ToF 8889（互相独立）。

一帧一个 datagram：
  {"ver":1,"seq":..,"t":..,"healthy":true,
   "ranges":{"L":..,"FL":..,"F":..,"FR":..,"R":..},   # mm 或 null
   "status":{"L":"VALID",...},                          # VALID/NO_TARGET/TOO_NEAR/IO_ERROR
   "init_count":..,"read_errors":..,"reinit_count":..}

语义
----
- **age 用 Windows 本地 time.monotonic() 的接收时刻** 计算，不用 Pi 的 t。
- **只接受 seq > last_seq**；重复/乱序丢弃；**大回退 = producer 重启 -> source reset**。
- `latest()` 在 stale / unhealthy / reset 时**抛异常**（让上层 FAILSAFE latch）。
  - age > max_age      -> TofStaleError
  - healthy == false   -> TofUnhealthyError
  - seq 大回退         -> TofSourceResetError
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass

RESET_SEQ_BACKSTEP = 100


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
    seq: int
    t: float
    healthy: bool
    ranges: dict          # name -> mm(float) 或 None
    status: dict          # name -> str
    init_count: int
    read_errors: int
    reinit_count: int


class Tof5Client:
    def __init__(self, port: int = 8889, max_age: float = 0.25, bind: str = "0.0.0.0",
                 recv_timeout: float = 0.2):
        self.max_age = float(max_age)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind, int(port)))
        self._sock.settimeout(float(recv_timeout))
        self._lock = threading.Lock()
        self._frame: TofFrame | None = None
        self._recv_t: float | None = None
        self._last_seq: int | None = None
        self._reset: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.received = 0
        self.dropped = 0
        self.bad = 0

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
            try:
                o = json.loads(data)
                fr = TofFrame(
                    seq=int(o["seq"]), t=float(o["t"]), healthy=bool(o["healthy"]),
                    ranges=dict(o["ranges"]), status=dict(o["status"]),
                    init_count=int(o.get("init_count", 0)),
                    read_errors=int(o.get("read_errors", 0)),
                    reinit_count=int(o.get("reinit_count", 0)))
            except (ValueError, KeyError, TypeError):
                with self._lock:
                    self.bad += 1
                continue
            now = time.monotonic()
            with self._lock:
                if self._last_seq is not None and fr.seq <= self._last_seq:
                    if (self._last_seq - fr.seq) > RESET_SEQ_BACKSTEP and self._reset is None:
                        self._reset = (self._last_seq, fr.seq)
                    self.dropped += 1
                    continue
                self._last_seq = fr.seq
                self._frame = fr
                self._recv_t = now
                self.received += 1

    def latest(self) -> tuple[TofFrame, float]:
        """返回 (frame, age_s)。stale / unhealthy / reset 时抛异常。"""
        with self._lock:
            fr, rt, reset = self._frame, self._recv_t, self._reset
        if reset is not None:
            raise TofSourceResetError(f"seq 大幅回落 {reset[0]} -> {reset[1]}（producer 重启？）")
        if fr is None or rt is None:
            raise TofStaleError("还没收到任何 ToF 帧")
        age = time.monotonic() - rt
        if age > self.max_age:
            raise TofStaleError(f"ToF age {age*1000:.0f}ms > {self.max_age*1000:.0f}ms")
        if not fr.healthy:
            raise TofUnhealthyError("ToF healthy=false（地址丢失/自愈失败）")
        return fr, age

    def age(self) -> float | None:
        with self._lock:
            rt = self._recv_t
        return None if rt is None else time.monotonic() - rt

    def stats(self) -> dict:
        with self._lock:
            return {"received": self.received, "dropped": self.dropped, "bad": self.bad,
                    "last_seq": self._last_seq, "reset": self._reset}


def _monitor(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="ToF UDP 监听/测速")
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--max-age", type=float, default=0.25)
    args = ap.parse_args(argv)

    c = Tof5Client(args.port, max_age=args.max_age, bind=args.bind)
    c.start()
    print(f"listening ToF udp :{args.port} for {args.seconds:g}s ...", flush=True)
    t0 = time.time()
    last = 0
    try:
        while time.time() - t0 < args.seconds:
            time.sleep(1.0)
            st = c.stats()
            rate = st["received"] - last
            last = st["received"]
            try:
                fr, age = c.latest()
                extra = (f"age={age*1000:.0f}ms healthy={fr.healthy} "
                         f"L={fr.ranges.get('L')} F={fr.ranges.get('F')} R={fr.ranges.get('R')}")
            except TofError as exc:
                extra = f"<{type(exc).__name__}>"
            print(f"t={time.time()-t0:4.1f}s recv={st['received']} rate={rate:.1f}/s "
                  f"drop={st['dropped']} bad={st['bad']} last_seq={st['last_seq']}  {extra}",
                  flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        c.stop()
    print("final:", c.stats())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_monitor())
