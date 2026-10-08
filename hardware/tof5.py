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
                 range_bias_mm: float = 0.0, min_mm: int = NEAR_MIN_MM,
                 invalid_mm: int = INVALID_MM, health_every: int = 50,
                 reinit_min_interval_s: float = 2.0):
        self.angles_deg = dict(angles_deg or DEFAULT_ANGLES_DEG)
        self.angles_rad = {k: math.radians(v) for k, v in self.angles_deg.items()}
        self.range_bias_mm = range_bias_mm
        self.min_mm = min_mm
        self.invalid_mm = invalid_mm
        self.health_every = health_every
        self.reinit_min_interval_s = reinit_min_interval_s

        self._i2c = None
        self._pins = {}
        self._sensors = {}
        self._read_n = 0
        self._last_reinit_try = 0.0
        self._got_io_error = False

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
        """返回 {L,FL,F,FR,R} -> RangeReading（含显式 state）。"""
        out = {}
        io_err = False
        for n in ORDER:
            try:
                out[n] = classify_range(self._sensors[n].range,
                                        min_mm=self.min_mm, invalid_mm=self.invalid_mm,
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


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="5x VL53L0X 驱动自测")
    ap.add_argument("--range-bias-mm", type=float, default=0.0)
    args = ap.parse_args()
    t = Tof5(range_bias_mm=args.range_bias_mm)
    print(f"启动 XSHUT={XSHUT_BCM} 角度={t.angles_deg}")
    t.start()
    print(f"初始化 OK（第 {t.init_count} 次）。10Hz 读取（Ctrl-C 结束）")
    try:
        while True:
            d = t.read()
            line = "  ".join(f"{n}:{d[n].state.value[:4]}:"
                             f"{'----' if d[n].mm is None else f'{d[n].mm:6.1f}'}"
                             for n in ORDER)
            print(f"{line}  [init={t.init_count} rerr={t.read_errors} "
                  f"reinit={t.reinit_events} healthy={t.healthy}]", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n结束")
    finally:
        t.stop()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
