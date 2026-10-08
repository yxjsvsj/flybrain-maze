"""P2a.5 里程计客户端（Windows 侧）：接收 Pi 的 50Hz UDP JSON 遥测。

语义（按设计约定）
------------------
- 一个 UDP datagram = 一个 JSON object，无需换行。
  形如 {"ver":1,"seq":123,"t":18.42,"left":-7280,"right":7132,"x":1.014,"y":-0.047,"theta":-0.131}
- **age 一律用 Windows 本地 time.monotonic() 的"接收时刻"算**，绝不用 Pi 的 t 减 Windows 时钟。
- 丢包可忽略；**只接受 seq > last_seq**。重复(==)或乱序(略小)直接丢弃。
- **producer 重启导致 seq 大幅回落** → ODOM_SOURCE_RESET，不自动重新对齐。
- `latest()` 在 stale / reset 时**抛异常**，绝不返回旧位姿——让上层硬失败。

不碰已经稳定的 Pikachu Flask 马达路径。
"""
from __future__ import annotations

import json
import math
import socket
import threading
import time
from dataclasses import dataclass

# seq 回退超过这个量，判定为 producer 重启（而不是单个乱序包）
RESET_SEQ_BACKSTEP = 100


class OdomError(RuntimeError):
    pass


class OdomStaleError(OdomError):
    """超过 max_age 没收到新包。"""


class OdomSourceResetError(OdomError):
    """seq 大幅回落：producer 重启。"""


@dataclass(frozen=True)
class OdomSample:
    seq: int
    t: float          # Pi 侧时钟，仅供参考
    left: int
    right: int
    x: float
    y: float
    theta: float


class OdomClient:
    def __init__(self, port: int, max_age: float = 0.25, bind: str = "0.0.0.0",
                 recv_timeout: float = 0.2):
        self.max_age = float(max_age)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind, int(port)))
        self._sock.settimeout(float(recv_timeout))
        self._lock = threading.Lock()
        self._sample: OdomSample | None = None
        self._recv_t: float | None = None
        self._last_seq: int | None = None
        self._reset: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.received = 0
        self.dropped = 0
        self.bad = 0

    # ---- 生命周期 ----
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

    # ---- 接收线程 ----
    def _rx(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                obj = json.loads(data)
                s = OdomSample(
                    seq=int(obj["seq"]), t=float(obj["t"]),
                    left=int(obj["left"]), right=int(obj["right"]),
                    x=float(obj["x"]), y=float(obj["y"]), theta=float(obj["theta"]))
            except (ValueError, KeyError, TypeError):
                with self._lock:
                    self.bad += 1
                continue
            now = time.monotonic()
            with self._lock:
                if self._last_seq is not None:
                    if s.seq <= self._last_seq:
                        if (self._last_seq - s.seq) > RESET_SEQ_BACKSTEP and self._reset is None:
                            self._reset = (self._last_seq, s.seq)
                        self.dropped += 1
                        continue
                self._last_seq = s.seq
                self._sample = s
                self._recv_t = now
                self.received += 1

    # ---- 读取 ----
    def latest(self) -> tuple[OdomSample, float]:
        """返回 (sample, age_s)。stale / reset 时抛异常，不返回旧值。"""
        with self._lock:
            s, rt, reset = self._sample, self._recv_t, self._reset
        if reset is not None:
            raise OdomSourceResetError(f"seq 大幅回落 {reset[0]} -> {reset[1]}（producer 重启？）")
        if s is None or rt is None:
            raise OdomStaleError("还没有收到任何 odom 包")
        age = time.monotonic() - rt
        if age > self.max_age:
            raise OdomStaleError(f"odom age {age*1000:.0f}ms > {self.max_age*1000:.0f}ms")
        return s, age

    def age(self) -> float | None:
        with self._lock:
            rt = self._recv_t
        return None if rt is None else time.monotonic() - rt

    def stats(self) -> dict:
        with self._lock:
            return {"received": self.received, "dropped": self.dropped, "bad": self.bad,
                    "last_seq": self._last_seq, "reset": self._reset}


def _monitor(argv=None) -> int:
    """诊断：监听 UDP 一段时间，打印速率/seq/age（T0 用）。"""
    import argparse

    ap = argparse.ArgumentParser(description="odom UDP 监听/测速")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--max-age", type=float, default=0.25)
    args = ap.parse_args(argv)

    c = OdomClient(args.port, max_age=args.max_age, bind=args.bind)
    c.start()
    print(f"listening udp :{args.port} for {args.seconds:g}s ...", flush=True)
    t0 = time.time()
    last = 0
    try:
        while time.time() - t0 < args.seconds:
            time.sleep(1.0)
            st = c.stats()
            rate = st["received"] - last
            last = st["received"]
            age = c.age()
            ages = "n/a" if age is None else f"{age*1000:.0f}ms"
            try:
                s, _a = c.latest()
                pose = (f"x={s.x:+.4f} y={s.y:+.4f} th={math.degrees(s.theta):+6.1f}deg "
                        f"L={s.left} R={s.right}")
            except Exception as exc:                       # noqa: BLE001
                pose = f"<{type(exc).__name__}>"
            print(f"t={time.time()-t0:4.1f}s  recv={st['received']:5d}  rate={rate:5.1f}/s  "
                  f"drop={st['dropped']} bad={st['bad']} last_seq={st['last_seq']} age={ages}\n"
                  f"          {pose}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        c.stop()
    print("final stats:", c.stats())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_monitor())
