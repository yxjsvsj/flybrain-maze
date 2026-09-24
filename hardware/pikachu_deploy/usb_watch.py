#!/usr/bin/env python3
"""监听 CH340 的真实 udev add/remove 事件（不再只看 /dev/serial/by-id 符号链接，
后者在设备重枚举时会被原子重建、`-e` 恒为真，看不出掉线）。

只读诊断：只打印，不做任何写操作、不碰串口。

用法: usb_watch.py            (Ctrl-C 停止)
      usb_watch.py <秒数>     (到时自动停止)
"""
import sys
import time

import pyudev

VENDOR, MODEL = "1a86", "7523"


def is_ch340(dev) -> bool:
    if dev.get("ID_VENDOR_ID") == VENDOR and dev.get("ID_MODEL_ID") == MODEL:
        return True
    product = dev.get("PRODUCT") or ""
    return product.startswith(f"{VENDOR}/{MODEL}/")


def main() -> int:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    start = time.time()
    state = {}
    mon = pyudev.Monitor.from_netlink(pyudev.Context())
    mon.filter_by(subsystem="usb")
    print(f"watching udev usb events for {VENDOR}:{MODEL}   Ctrl-C to stop", flush=True)
    while True:
        dev = mon.poll(timeout=1.0)
        if dev is None:
            if duration > 0 and time.time() - start >= duration:
                print(f"done after {duration:.0f}s", flush=True)
                return 0
            continue
        if not is_ch340(dev):
            continue
        now = time.time()
        prev = state.get(dev.sys_name)
        gap = f"(prev state lasted {now - prev:.0f}s)" if prev else ""
        state[dev.sys_name] = now
        print(f"{time.strftime('%H:%M:%S')}  {dev.action:<6}  {dev.sys_name}  {gap}", flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
