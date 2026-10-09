"""P2b：5x VL53L0X 驱动（自愈：地址丢失自动重新分配）。

接线（BCM）
-----------
  3V3 -> 全部 VIN ; GND -> 全部 GND
  GPIO2 (SDA) / GPIO3 (SCL)
  XSHUT: GPIO5->L  GPIO6->FL  GPIO13->F  GPIO19->FR  GPIO26->R   (GPIO1 = NC)

地址（VL53L0X 断电/复位后回 0x29）：
  L 0x30   FL 0x31   F 0x32   FR 0x33   R 0x34

方位角（相对车头，**逆时针/车体左侧为正**）：**参数化**，默认 v1.3 支架：
  L +70°   FL +30°   F 0°   FR -30°   R -70°

距离状态（**显式**，不再用 None 混表"无目标"和"太近"）
------------------------------------------------------
  VALID     有效读数（mm）
  NO_TARGET 8191/超量程；neural 可按 max_range，mapping 通常跳过该 ray
  TOO_NEAR  太近(<min_mm)；映射为 min distance，触发 front safety/stop
  IO_ERROR  I2C 读取异常；上层应 unhealthy -> FAILSAFE

关键：**任何一次传感器复位都会丢地址**，所以本驱动不只启动时分配一次——
读到 I2C 错误、或周期性健康检查发现地址丢失时，会自动重跑 XSHUT 地址分配。
仍然失败则 healthy=False，由上层处理。

无硬件时（Windows）也能 import——blinka 只在真正 start() 时导入。
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from enum import Enum

ORDER = ("L", "FL", "F", "FR", "R")
XSHUT_BCM = {"L": 5, "FL": 6, "F": 13, "FR": 19, "R": 26}
I2C_ADDR = {"L": 0x30, "FL": 0x31, "F": 0x32, "FR": 0x33, "R": 0x34}

# v1.3 支架的真实角度（度，逆时针为正）。仅作默认值；请用 angles_deg 传入。
DEFAULT_ANGLES_DEG = {"L": 70.0, "FL": 30.0, "F": 0.0, "FR": -30.0, "R": -70.0}

INVALID_MM = 8191        # VL53L0X "无目标/超量程" sentinel
NEAR_MIN_MM = 30         # 最小量程：比这更近不测


class RangeState(str, Enum):
    VALID = "VALID"
    NO_TARGET = "NO_TARGET"
    TOO_NEAR = "TOO_NEAR"
    IO_ERROR = "IO_ERROR"


@dataclass(frozen=True)
class RangeReading:
    state: RangeState
    mm: float | None = None      # VALID -> 距离；TOO_NEAR -> min_mm；其它 -> None

    @property
    def valid(self) -> bool:
        return self.state is RangeState.VALID


def classify_range(raw, *, min_mm: int = NEAR_MIN_MM, invalid_mm: int = INVALID_MM,
                   range_bias_mm: float = 0.0) -> RangeReading:
    """把原始读数分类成 (state, mm)。异常请由调用方包成 IO_ERROR。"""
    try:
        r = int(raw)
    except (TypeError, ValueError):
        return RangeReading(RangeState.IO_ERROR, None)
    if r <= 0 or r >= invalid_mm - 1:
        return RangeReading(RangeState.NO_TARGET, None)
    if r < min_mm:
        return RangeReading(RangeState.TOO_NEAR, float(min_mm))
    return RangeReading(RangeState.VALID, float(r) - range_bias_mm)


class Tof5:
    """5x VL53L0X。start() 分配地址；read() 读；丢地址会自愈。"""

    def __init__(self, *, angles_deg: dict | None = None,
                 sensor_origin_m: dict | None = None,
                 range_bias_mm: float = 0.0, min_mm: int = NEAR_MIN_MM,
                 invalid_mm: int = INVALID_MM, health_every: int = 50,
                 reinit_min_interval_s: float = 2.0):
        self.angles_deg = dict(angles_deg or DEFAULT_ANGLES_DEG)
        self.angles_rad = {k: math.radians(v) for k, v in self.angles_deg.items()}
        # 传感器相对车体原点的安装位移（米）。初版全部 (0,0)——只做注入/方向验证；
        # 正式 P2c 前必须实测每个传感器的 (x, y)，不要永久假设共点。
        self.sensor_origin_m = dict(sensor_origin_m or {n: (0.0, 0.0) for n in ORDER})
        self.range_bias_mm = range_bias_mm
        self.min_mm = min_mm
        self.invalid_mm = invalid_mm
        self.health_every = health_every
        self.reinit_min_interval_s = reinit_min_interval_s

        self._i2c = None
        self._pins = {}
        self._sensors = {}
        self._read_n = 0
        self._seq = 0
        self._t0 = time.monotonic()
        self._last_reinit_try = 0.0
        self._got_io_error = False
        # continuous 模式：每路的最新原始读数与"最后一次真实更新"时刻
        self._last_raw: dict[str, int | None] = {n: None for n in ORDER}
        self._last_upd: dict[str, float | None] = {n: None for n in ORDER}
        self.session_id = f"{int(time.time())}-{os.getpid()}"

        self.init_count = 0
        self.read_errors = 0
        self.reinit_events = 0
        self.healthy = False

    # ---- 生命周期 ----
    def start(self) -> None:
        import board
        import busio
        self._i2c = busio.I2C(board.SCL, board.SDA)
        self._claim_pins()
        if not self._init_addresses():
            raise RuntimeError("ToF 地址初始化失败（检查 VIN/GND/SDA/SCL/XSHUT）")

    def stop(self) -> None:
        for s in self._sensors.values():
            try:
                s.stop_continuous()
            except Exception:                                  # noqa: BLE001
                pass
        for p in self._pins.values():
            try:
                p.deinit()
            except Exception:                                  # noqa: BLE001
                pass
        self._pins = {}
        self._sensors = {}
        self.healthy = False

    def __enter__(self) -> "Tof5":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- 内部 ----
    def _claim_pins(self) -> None:
        import board
        import digitalio
        for n in ORDER:
            pin = getattr(board, f"D{XSHUT_BCM[n]}")   # board.D5 == BCM5
            p = digitalio.DigitalInOut(pin)
            p.direction = digitalio.Direction.OUTPUT
            p.value = False
            self._pins[n] = p
        time.sleep(0.1)

    def _init_addresses(self) -> bool:
        import adafruit_vl53l0x
        for n in ORDER:
            self._pins[n].value = False        # 全部进 reset
        time.sleep(0.1)
        self._sensors = {}
        for n in ORDER:
            self._pins[n].value = True         # 只放这一个
            time.sleep(0.02)
            try:
                s = adafruit_vl53l0x.VL53L0X(self._i2c)
                s.set_address(I2C_ADDR[n])
                self._sensors[n] = s
            except Exception:                                  # noqa: BLE001
                self.healthy = False
                return False
        # 地址分配完成后，逐路进入 continuous ranging（之后主循环只读最新结果）
        for n in ORDER:
            try:
                self._sensors[n].start_continuous()
            except Exception:                                  # noqa: BLE001
                self.healthy = False
                return False
        for n in ORDER:
            self._last_raw[n] = None
            self._last_upd[n] = None
        self.init_count += 1
        self.healthy = True
        return True

    def _scan(self):
        i2c = self._i2c
        if i2c is None:
            return set()
        while not i2c.try_lock():
            pass
        try:
            return set(i2c.scan())
        finally:
            i2c.unlock()

    def health_ok(self) -> bool:
        want = set(I2C_ADDR.values())
        return want.issubset(self._scan())

    def _recover(self) -> bool:
        now = time.monotonic()
        if now - self._last_reinit_try < self.reinit_min_interval_s:
            return False
        self._last_reinit_try = now
        self.reinit_events += 1
        return self._init_addresses()

    # ---- 读取 ----
    def read(self) -> dict:
        """返回 {L,FL,F,FR,R} -> RangeReading。

        continuous 模式下**只在 data_ready 时读**（非阻塞），否则沿用上一次的
        **真实**测量值；首帧前该路视为 NO_TARGET。绝不把旧值当新测量。
        """
        out = {}
        io_err = False
        now = time.monotonic()
        for n in ORDER:
            try:
                if self._sensors[n].data_ready:
                    self._last_raw[n] = int(self._sensors[n].read_range())   # 读+清中断
                    self._last_upd[n] = now
                raw = self._last_raw[n]
                if raw is None:
                    out[n] = RangeReading(RangeState.NO_TARGET, None)
                else:
                    out[n] = classify_range(raw, min_mm=self.min_mm,
                                            invalid_mm=self.invalid_mm,
                                            range_bias_mm=self.range_bias_mm)
            except Exception:                                  # noqa: BLE001
                out[n] = RangeReading(RangeState.IO_ERROR, None)
                io_err = True
                self.read_errors += 1
        self._read_n += 1
        if io_err:
            self._got_io_error = True
        if io_err or (self.health_every and self._read_n % self.health_every == 0
                      and not self.health_ok()):
            self._recover()
        return out

    def frame(self) -> dict:
        """读一帧并打包成标准遥测 frame（供 UDP 发送 / 环路消费）。"""
        d = self.read()
        self._seq += 1
        now = time.monotonic()
        ages = {}
        for n in ORDER:
            u = self._last_upd[n]
            ages[n] = None if u is None else round((now - u) * 1000.0, 1)
        return {
            "ver": 2,
            "session_id": self.session_id,
            "seq": self._seq,
            "t": round(now - self._t0, 4),
            "healthy": bool(self.healthy),
            "ranges": {n: (None if d[n].mm is None else round(d[n].mm, 1)) for n in ORDER},
            "status": {n: d[n].state.value for n in ORDER},
            "ages_ms": ages,
            "init_count": self.init_count,
            "read_errors": self.read_errors,
            "reinit_count": self.reinit_events,
        }


def _publish_state(path: str, obj: dict) -> None:
    """原子写共享状态（本机 web 读）。失败静默——绝不拖累主循环。"""
    import json
    try:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, separators=(",", ":"))
        os.replace(tmp, path)
    except OSError:
        pass


def _main() -> int:
    import argparse
    import json
    import socket
    import sys

    ap = argparse.ArgumentParser(description="5x VL53L0X 驱动自测 / UDP 遥测")
    ap.add_argument("--range-bias-mm", type=float, default=0.0)
    ap.add_argument("--udp", default="",
                    help="host:port，10Hz 发 compact JSON（如 192.168.50.123:8889）")
    ap.add_argument("--udp-hz", type=float, default=10.0)
    ap.add_argument("--state-file", default="",
                    help="把最新 frame 原子写入该 JSON（供本机 web 读共享状态）")
    ap.add_argument("--state-hz", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = 一直跑")
    args = ap.parse_args()

    usock = None
    udp_addr = None
    if args.udp:
        host, _, port = args.udp.rpartition(":")
        udp_addr = (host, int(port))
        usock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP -> {host}:{port} @ {args.udp_hz:g}Hz")

    state_file = args.state_file or None
    if state_file:
        d = os.path.dirname(state_file)
        if d:
            os.makedirs(d, exist_ok=True)
        print(f"state -> {state_file} @ {args.state_hz:g}Hz")

    t = Tof5(range_bias_mm=args.range_bias_mm)
    print(f"启动 XSHUT={XSHUT_BCM} 角度={t.angles_deg} origin={t.sensor_origin_m}")
    t.start()
    print(f"初始化 OK（第 {t.init_count} 次）。{args.udp_hz:g}Hz 读取（Ctrl-C 结束）")

    t0 = time.monotonic()
    next_tx = t0
    next_state = t0
    try:
        while True:
            f = t.frame()
            parts = []
            for n in ORDER:
                mm = f["ranges"][n]
                mm_s = "----" if mm is None else f"{mm:6.1f}"
                parts.append(f"{n}:{f['status'][n][:4]}:{mm_s}")
            print("  ".join(parts) +
                  f"   [init={f['init_count']} rerr={f['read_errors']} "
                  f"reinit={f['reinit_count']} healthy={f['healthy']}]", flush=True)

            now = time.monotonic()
            if usock is not None and udp_addr is not None and now >= next_tx:
                try:
                    usock.sendto(json.dumps(f).encode(), udp_addr)
                except OSError as exc:
                    print(f"[udp] send 失败: {exc}", file=sys.stderr)
                next_tx = now + 1.0 / max(0.1, args.udp_hz)

            if state_file is not None and now >= next_state:
                _publish_state(state_file, dict(f, wall=round(time.time(), 3)))
                next_state = now + 1.0 / max(0.5, args.state_hz)

            if args.seconds > 0 and now - t0 >= args.seconds:
                break
            time.sleep(1.0 / max(1.0, args.udp_hz))
    except KeyboardInterrupt:
        print("\n结束")
    finally:
        t.stop()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
