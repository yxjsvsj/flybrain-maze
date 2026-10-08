"""P2b bring-up：5 x VL53L0X —— XSHUT 地址分配 + 10Hz 五路读取。

接线（BCM）
-----------
  3V3  -> 全部 VIN
  GND  -> 全部 GND
  GPIO2 (SDA) -> 全部 SDA
  GPIO3 (SCL) -> 全部 SCL
  XSHUT: GPIO5->L  GPIO6->FL  GPIO13->F  GPIO19->FR  GPIO26->R
  GPIO1 = NC

地址（VL53L0X 断电后回 0x29，所以**每次启动都必须重新分配**）：
  L 0x30   FL 0x31   F 0x32   FR 0x33   R 0x34

用法：
  source /home/bjlm/pikachu/venv/bin/activate
  python -u /home/bjlm/pikachu/test_tof5.py
"""
import time

import board
import busio
import digitalio
import adafruit_vl53l0x

ORDER = ["L", "FL", "F", "FR", "R"]
XSHUT = {"L": board.D5, "FL": board.D6, "F": board.D13, "FR": board.D19, "R": board.D26}
ADDR = {"L": 0x30, "FL": 0x31, "F": 0x32, "FR": 0x33, "R": 0x34}


def make_i2c():
    return busio.I2C(board.SCL, board.SDA)


def scan(i2c):
    while not i2c.try_lock():
        pass
    try:
        return [hex(a) for a in i2c.scan()]
    finally:
        i2c.unlock()


def main():
    i2c = make_i2c()
    print("初始化：全部 XSHUT LOW（全部进入 reset 状态）")
    pins = {}
    for n in ORDER:
        p = digitalio.DigitalInOut(XSHUT[n])
        p.direction = digitalio.Direction.OUTPUT
        p.value = False
        pins[n] = p
    time.sleep(0.1)
    print("全部 XSHUT LOW 后扫描：", scan(i2c))
    print()

    sensors = {}
    for n in ORDER:
        pins[n].value = True          # 只放这一个出 reset
        time.sleep(0.05)
        print(f" {n} started -> {hex(ADDR[n])}")
        s = adafruit_vl53l0x.VL53L0X(i2c)
        s.set_address(ADDR[n])
        sensors[n] = s

    print()
    print("最终 I2C 地址：")
    print(scan(i2c))
    print()
    print("开始 10Hz 读取（Ctrl-C 结束）")
    try:
        while True:
            parts = []
            for n in ORDER:
                try:
                    parts.append(f"{n}: {sensors[n].range:4d}mm")
                except Exception as exc:                       # noqa: BLE001
                    parts.append(f"{n}: ERR({type(exc).__name__})")
            print("  ".join(parts), flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n结束")


if __name__ == "__main__":
    main()
