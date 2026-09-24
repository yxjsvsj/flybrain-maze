"""P2a：编码器 x4 正交解码 + CPR 标定，为差速里程计打基础。

平台：Raspberry Pi 5（RP1 GPIO）。用 libgpiod v2 的**边沿事件**（不合并、带时间戳），
不用轮询、不用 lgpio 回调，避免高速时丢计数。

接线（务必先读）
----------------
- 编码器 VCC 接 **3.3V**。若该模块只能 5V 供电，**必须加电平转换**，否则 5V 信号会
  打坏 Pi 的 3.3V GPIO。
- 编码器 GND 与 Pi GND **共地**。
- E1A/E1B/E2A/E2B -> 4 个空闲 GPIO（**BCM 编号**，不是物理引脚号）。
- MG513 的电机电源（12V / VM）**不要**接到 Pi。
- /dev/gpiochip0 属于 `gpio` 组：用 `sudo` 运行，或把用户加入 `gpio` 组。

用法
----
    # 1) 只看实时计数（手动转轮子，确认哪个通道在动、方向符号对不对）
    sudo python3 hardware/encoder_calib.py --e1 17 27 --e2 22 23

    # 2) 标定：录 N 秒到 CSV；期间把**某个轮子正好转 1 圈**（贴胶带做标记）
    sudo python3 hardware/encoder_calib.py --e1 17 27 --e2 22 23 \
        --seconds 30 --csv enc_calib.csv

CPR 预期（若为 MG513 13 PPR Hall + 30:1 减速）：
    x1 = 13*30 = 390 ; x2 = 780 ; x4 = 1560 counts/输出轴圈
    本工具用 x4。手转 1 圈应得到接近 1560 的计数——**以实测为准**，不要假设。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

CHIP = "/dev/gpiochip0"

# x4 正交状态转移表：键 (prevA, prevB, newA, newB) -> ±1；非法/无变化 -> 0
QTAB = {
    (0, 0, 0, 1): +1, (0, 1, 1, 1): +1, (1, 1, 1, 0): +1, (1, 0, 0, 0): +1,
    (0, 0, 1, 0): -1, (1, 0, 1, 1): -1, (1, 1, 0, 1): -1, (0, 1, 0, 0): -1,
}


def _bit(value) -> int:
    """libgpiod Value(enum) -> 0/1。"""
    v = getattr(value, "value", value)
    return 1 if int(v) == 1 else 0


class Quad:
    """单个电机的 x4 正交解码器（A/B 两相）。"""

    def __init__(self, req, a: int, b: int):
        self.req = req
        self.a = a
        self.b = b
        self.state = (_bit(req.get_value(a)), _bit(req.get_value(b)))
        self.count = 0
        self.glitches = 0

    def on_edge(self) -> None:
        new = (_bit(self.req.get_value(self.a)), _bit(self.req.get_value(self.b)))
        d = QTAB.get((self.state[0], self.state[1], new[0], new[1]), 0)
        if d == 0:
            self.glitches += 1
        self.count += d
        self.state = new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2a 编码器 x4 标定")
    ap.add_argument("--e1", nargs=2, type=int, metavar=("A", "B"), required=True,
                    help="电机1 的 A/B 两相 GPIO（BCM）")
    ap.add_argument("--e2", nargs=2, type=int, metavar=("A", "B"), required=True,
                    help="电机2 的 A/B 两相 GPIO（BCM）")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="录多久（0 = 一直跑，Ctrl-C 结束）")
    ap.add_argument("--csv", default="", help="把 (t, e1, e2) 写到这个 CSV")
    ap.add_argument("--print-hz", type=float, default=2.0, help="实时打印频率")
    ap.add_argument("--invert-e1", action="store_true", help="电机1 计数取反")
    ap.add_argument("--invert-e2", action="store_true", help="电机2 计数取反")
    ap.add_argument("--wait-motion", action="store_true",
                    help="先等到有计数变化再开始计时；静默 --settle 秒后自动收尾（不用掐时间）")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="wait-motion 下，静默多少秒算转完")
    ap.add_argument("--max-wait", type=float, default=120.0,
                    help="wait-motion 等待首次转动的最长秒数")
    args = ap.parse_args(argv)

    try:
        import gpiod
        from gpiod.line import Bias, Direction, Edge
    except Exception as exc:                                   # noqa: BLE001
        print(f"libgpiod 不可用：{exc}", file=sys.stderr)
        return 2

    e1a, e1b = args.e1
    e2a, e2b = args.e2
    offsets = [e1a, e1b, e2a, e2b]
    if len(set(offsets)) != 4:
        print("四个 GPIO 必须互不相同", file=sys.stderr)
        return 2

    cfg = {off: gpiod.LineSettings(direction=Direction.INPUT,
                                   edge_detection=Edge.BOTH,
                                   bias=Bias.PULL_UP) for off in offsets}

    print(f"chip={CHIP}  E1=({e1a},{e1b})  E2=({e2a},{e2b})")
    print("预期（13PPR Hall + 30:1, x4）: 1 圈 ≈ 1560。手转一圈看实测。")
    print("Ctrl-C 结束。\n")

    fh = open(args.csv, "w", newline="") if args.csv else None
    writer = csv.writer(fh) if fh else None
    if writer:
        writer.writerow(["t", "e1", "e2"])

    t0 = time.time()
    abs0 = t0
    next_print = t0
    q1 = q2 = None
    base1 = base2 = 0
    try:
        with gpiod.request_lines(CHIP, consumer="encoder_calib", config=cfg) as req:
            q1 = Quad(req, e1a, e1b)
            q2 = Quad(req, e2a, e2b)
            prev = (q1.count, q2.count)
            last_change = None
            waiting = args.wait_motion
            if waiting:
                print(f"[wait-motion] 等待转动…（最多 {args.max_wait:.0f}s）", flush=True)
            while True:
                if req.wait_edge_events(timeout=0.1):
                    for ev in req.read_edge_events():
                        off = ev.line_offset
                        if off in (e1a, e1b):
                            q1.on_edge()
                        else:
                            q2.on_edge()
                now = time.time()
                cur = (q1.count, q2.count)
                if cur != prev:
                    last_change = now
                    if waiting:
                        waiting = False
                        base1, base2 = prev
                        t0 = now                       # 有效计时从"动起来"开始
                        next_print = now
                        print("[wait-motion] 检测到转动，开始记录…", flush=True)
                    prev = cur
                if writer is not None and fh is not None:
                    writer.writerow([f"{now - abs0:.3f}", q1.count, q2.count])
                    fh.flush()
                if (not waiting) and now >= next_print:
                    print(f"t={now - t0:6.1f}s   E1={q1.count - base1:7d}   "
                          f"E2={q2.count - base2:7d}", flush=True)
                    next_print = now + 1.0 / max(0.1, args.print_hz)
                if waiting:
                    if now - abs0 > args.max_wait:
                        print("[wait-motion] 超时：没有检测到任何计数变化。", flush=True)
                        return 1
                    continue
                if last_change is not None and (now - last_change) >= args.settle:
                    break
                if args.seconds and (now - t0) >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if fh:
            fh.close()

    if q1 is None or q2 is None:
        print("未读到编码器（request_lines 失败？）", file=sys.stderr)
        return 1
    d1 = q1.count - base1
    d2 = q2.count - base2
    if args.invert_e1:
        d1 = -d1
    if args.invert_e2:
        d2 = -d2
    print(f"\nΔE1={d1}   ΔE2={d2}   抖动/非法转移: E1={q1.glitches} E2={q2.glitches}")
    print("若你正好把某轮转了 1 圈，该轮 Δ 就是实测 CPR（对比预期 1560）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
