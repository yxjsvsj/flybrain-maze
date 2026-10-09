"""P2a 差速里程计：读 E1/E2 编码器 -> 位姿 (x, y, theta)。

标定结果（wheels-up 实测）
-------------------------
  CPR (x4)   = 1560   （MG513P30_12V，13 PPR Hall，减速 30:1，x4 解码）
  轮径 D     = 0.070 m（名义/几何值）
  轮距 L     = 0.160 m（左右轮接地点中心距）

  落地（P2c ground commissioning）**有效值**（含打滑，生产用；由
  pikachu_deploy/start_both_inner.sh 以 --wheel-diam/--track 传入）：
    有效 D = 0.067 m     有效 L = 0.194 m

编码器通道 vs 物理左右（重要）
------------------------------
  原始通道：
    E1 = GPIO22/23
    E2 = GPIO17/27

  **实体车实测：E1/E2 与物理 left/right 相反**，所以当前生产启动必须用 --swap-lr。
  不要把 E1/E2 的名字本身等同于 left/right——左右轮定义以 swap 之后的物理轮映射为准。

  （各通道前进方向：E1 前进 = 负计数，E2 前进 = 正计数；符号可用
   --left-sign/--right-sign 覆盖，这里的 left/right 指原始通道。）

坐标约定
--------
  x 前、y 左、theta 逆时针为正（右手系）；初始位姿 (0, 0, 0)。

模型
----
  ds_left  = left_sign  * ΔE1 * (πD/CPR)     # left_sign  = -1
  ds_right = right_sign * ΔE2 * (πD/CPR)     # right_sign = +1
  ds     = (ds_left + ds_right) / 2
  dtheta = (ds_right - ds_left) / L
  x += ds*cos(theta + dtheta/2);  y += ds*sin(theta + dtheta/2);  theta += dtheta

解码
----
  用 libgpiod 边沿事件，每个事件只更新**它自己那一相**的电平（另一相沿用已存状态），
  这样两相边沿挨得很近时也不会误判成非法跳变。

用法
----
    python3 hardware/odometry.py --e1 22 23 --e2 17 27 --swap-lr --wait-motion --settle 4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import sys
import time

CHIP = "/dev/gpiochip0"

CPR = 1560.0
WHEEL_DIAM = 0.070
TRACK = 0.160
LEFT_SIGN = -1.0     # E1 前进 = 负计数（实测：车头方向推车）
RIGHT_SIGN = +1.0    # E2 前进 = 正计数

QTAB = {
    (0, 0, 0, 1): +1, (0, 1, 1, 1): +1, (1, 1, 1, 0): +1, (1, 0, 0, 0): +1,
    (0, 0, 1, 0): -1, (1, 0, 1, 1): -1, (1, 1, 0, 1): -1, (0, 1, 0, 0): -1,
}


def _bit(value) -> int:
    v = getattr(value, "value", value)
    return 1 if int(v) == 1 else 0


def _publish_state(path: str, obj: dict) -> None:
    """原子写共享状态（本机 web 读）。失败静默——绝不拖累主循环。"""
    try:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, separators=(",", ":"))
        os.replace(tmp, path)
    except OSError:
        pass


class Quad:
    """单个电机的 x4 正交解码器。事件驱动：只改事件对应的那一相。"""

    def __init__(self, a: int, b: int, init_state: tuple[int, int]):
        self.a = a
        self.b = b
        self.state = init_state
        self.count = 0
        self.glitches = 0

    def on_event(self, offset: int, level: int) -> None:
        pa, pb = self.state
        if offset == self.a:
            na, nb = level, pb
        else:
            na, nb = pa, level
        d = QTAB.get((pa, pb, na, nb), 0)
        if d == 0:
            self.glitches += 1
        self.count += d
        self.state = (na, nb)


class DifferentialOdometry:
    """差速里程计积分器。E1/E2 为原始通道；物理左右由 swap_lr 决定。"""

    def __init__(self, cpr: float = CPR, wheel_diam: float = WHEEL_DIAM,
                 track: float = TRACK, left_sign: float = LEFT_SIGN,
                 right_sign: float = RIGHT_SIGN, swap_lr: bool = False):
        self.m_per_count = math.pi * wheel_diam / cpr
        self.track = track
        self.left_sign = left_sign
        self.right_sign = right_sign
        self.swap_lr = bool(swap_lr)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._e1 = 0
        self._e2 = 0
        self._init = False

    def update(self, e1: int, e2: int):
        if not self._init:
            self._e1, self._e2, self._init = e1, e2, True
            return 0.0, 0.0
        d1 = e1 - self._e1
        d2 = e2 - self._e2
        self._e1, self._e2 = e1, e2

        ds_left = self.left_sign * d1 * self.m_per_count
        ds_right = self.right_sign * d2 * self.m_per_count
        if self.swap_lr:                       # 实测 E1/E2 与物理左右相反
            ds_left, ds_right = ds_right, ds_left
        ds = 0.5 * (ds_left + ds_right)
        dtheta = (ds_right - ds_left) / self.track

        mid = self.theta + 0.5 * dtheta
        self.x += ds * math.cos(mid)
        self.y += ds * math.sin(mid)
        self.theta += dtheta
        return ds, dtheta

    def pose(self):
        return self.x, self.y, self.theta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2a 差速里程计")
    ap.add_argument("--e1", nargs=2, type=int, metavar=("A", "B"), required=True,
                    help="原始通道 E1 的 A/B GPIO（BCM）—— 非物理左轮")
    ap.add_argument("--e2", nargs=2, type=int, metavar=("A", "B"), required=True,
                    help="原始通道 E2 的 A/B GPIO（BCM）—— 非物理右轮")
    ap.add_argument("--wheel-diam", type=float, default=WHEEL_DIAM, help="轮径 m")
    ap.add_argument("--track", type=float, default=TRACK, help="轮距 m")
    ap.add_argument("--cpr", type=float, default=CPR, help="每输出轴圈计数 x4")
    ap.add_argument("--left-sign", type=float, default=LEFT_SIGN,
                    help="E1 前进方向计数符号（+1/-1）")
    ap.add_argument("--right-sign", type=float, default=RIGHT_SIGN,
                    help="E2 前进方向计数符号（+1/-1）")
    ap.add_argument("--swap-lr", action="store_true",
                    help="交换左右（实测 E1/E2 与物理左右相反时用）")
    ap.add_argument("--seconds", type=float, default=30.0, help="跑多久（0 = 一直跑，Ctrl-C 结束）")
    ap.add_argument("--print-hz", type=float, default=2.0)
    ap.add_argument("--csv", default="")
    ap.add_argument("--udp", default="",
                    help="host:port，按 --udp-hz 发 UDP JSON 遥测（如 <windows-ip>:8888）")
    ap.add_argument("--udp-hz", type=float, default=50.0)
    ap.add_argument("--state-file", default="",
                    help="把最新遥测原子写入该 JSON（供本机 web 读共享状态）")
    ap.add_argument("--state-hz", type=float, default=20.0)
    ap.add_argument("--wait-motion", action="store_true",
                    help="先等到累计变化 >= --motion-counts；静默 --settle 秒后收尾")
    ap.add_argument("--motion-counts", type=int, default=40,
                    help="wait-motion 触发所需的累计计数变化（挡噪声）")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--max-wait", type=float, default=180.0)
    ap.add_argument("--debug-events", type=int, default=0,
                    help="打印前 N 个原始边沿事件（诊断）")
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
    odom = DifferentialOdometry(cpr=args.cpr, wheel_diam=args.wheel_diam,
                                track=args.track, left_sign=args.left_sign,
                                right_sign=args.right_sign, swap_lr=args.swap_lr)

    print(f"chip={CHIP}  raw E1=({e1a},{e1b})  raw E2=({e2a},{e2b})  swap_lr={args.swap_lr}")
    print(f"CPR={args.cpr:g}  D={args.wheel_diam*100:.1f}cm  L={args.track*100:.1f}cm  "
          f"({odom.m_per_count*1e3:.4f} mm/count)")
    print("坐标 x前 y左 theta逆时针。Ctrl-C 结束。\n")

    fh = open(args.csv, "w", newline="") if args.csv else None
    writer = csv.writer(fh) if fh else None
    if writer:
        writer.writerow(["t", "e1", "e2", "x", "y", "theta"])

    usock = None
    udp_addr = None
    if args.udp:
        host, _, port = args.udp.rpartition(":")
        if not host or not port:
            print("--udp 需要 host:port", file=sys.stderr)
            return 2
        udp_addr = (host, int(port))
        usock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP -> {host}:{port} @ {args.udp_hz:g}Hz")

    state_file = args.state_file or None
    if state_file:
        try:
            d = os.path.dirname(state_file)
            if d:
                os.makedirs(d, exist_ok=True)
            print(f"state -> {state_file} @ {args.state_hz:g}Hz")
        except OSError as exc:
            print(f"[state] 无法使用 {state_file}: {exc} —— 关闭 state 输出，继续运行",
                  file=sys.stderr)
            state_file = None

    t0 = time.time()
    next_print = t0
    next_udp = t0
    next_state = t0
    seq = 0
    udp_errs = 0
    sent_ok = 0
    session_id = f"{int(time.time())}-{os.getpid()}"
    poll_to = 0.01 if usock is not None else 0.1
    dbg = args.debug_events
    q1 = q2 = None
    try:
        with gpiod.request_lines(CHIP, consumer="odometry", config=cfg) as req:
            q1 = Quad(e1a, e1b, (_bit(req.get_value(e1a)), _bit(req.get_value(e1b))))
            q2 = Quad(e2a, e2b, (_bit(req.get_value(e2a)), _bit(req.get_value(e2b))))
            odom.update(q1.count, q2.count)
            prev = (0, 0)
            moved = 0
            last_change = None
            waiting = args.wait_motion
            if waiting:
                print(f"[wait-motion] 推车…（需累计 {args.motion_counts} 计数；"
                      f"最多 {args.max_wait:.0f}s）", flush=True)
            while True:
                if req.wait_edge_events(timeout=poll_to):
                    for ev in req.read_edge_events():
                        # 取事件自己那一相的当前电平（比 event_type 比较稳）
                        lvl = _bit(req.get_value(ev.line_offset))
                        if dbg > 0:
                            print(f"  [ev] off={ev.line_offset} type={ev.event_type} lvl={lvl}",
                                  flush=True)
                            dbg -= 1
                        if ev.line_offset in (e1a, e1b):
                            q1.on_event(ev.line_offset, lvl)
                        else:
                            q2.on_event(ev.line_offset, lvl)
                odom.update(q1.count, q2.count)
                now = time.time()
                cur = (q1.count, q2.count)
                if cur != prev:
                    moved += abs(cur[0] - prev[0]) + abs(cur[1] - prev[1])
                    last_change = now
                    prev = cur
                    if waiting and moved >= args.motion_counts:
                        waiting = False
                        print(f"[wait-motion] 检测到移动（累计 {moved} 计数），记录中…",
                              flush=True)
                x, y, th = odom.pose()
                send_udp = usock is not None and udp_addr is not None and now >= next_udp
                send_state = state_file is not None and now >= next_state
                if send_udp or send_state:
                    msg = {
                        "ver": 2, "session_id": session_id, "seq": seq,
                        "t": round(now - t0, 4), "wall": round(now, 3),
                        "left": q1.count, "right": q2.count,
                        "x": round(x, 5), "y": round(y, 5), "theta": round(th, 6),
                    }
                    seq += 1
                    if send_udp and usock is not None and udp_addr is not None:
                        try:
                            usock.sendto(json.dumps(msg).encode(), udp_addr)
                            sent_ok += 1
                        except OSError as exc:
                            udp_errs += 1
                            if udp_errs <= 3 or udp_errs % 200 == 0:
                                print(f"[udp] send 失败 #{udp_errs}: {exc}", file=sys.stderr)
                        next_udp = now + 1.0 / max(1.0, args.udp_hz)
                    if send_state and state_file is not None:
                        _publish_state(state_file, msg)
                        next_state = now + 1.0 / max(0.5, args.state_hz)
                if writer is not None and fh is not None:
                    writer.writerow([f"{now - t0:.3f}", q1.count, q2.count,
                                     f"{x:.4f}", f"{y:.4f}", f"{th:.4f}"])
                    fh.flush()
                if (not waiting) and now >= next_print:
                    print(f"t={now - t0:5.1f}s  x={x:+7.3f}  y={y:+7.3f}  "
                          f"th={math.degrees(th):+7.1f}deg   (E1={q1.count} E2={q2.count})",
                          flush=True)
                    next_print = now + 1.0 / max(0.1, args.print_hz)
                if waiting:
                    if now - t0 > args.max_wait:
                        print("[wait-motion] 超时：没有检测到足够移动。", flush=True)
                        return 1
                    continue
                if args.wait_motion and last_change is not None \
                        and (now - last_change) >= args.settle:
                    break
                if args.seconds > 0 and (now - t0) >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if fh:
            fh.close()

    x, y, th = odom.pose()
    print(f"\n最终位姿: x={x:+.3f} m  y={y:+.3f} m  th={math.degrees(th):+.1f} deg")
    if q1 is not None and q2 is not None:
        print(f"抖动/非法转移: E1={q1.glitches} E2={q2.glitches}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
