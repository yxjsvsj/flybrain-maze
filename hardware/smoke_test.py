"""Pikachu Bot wheels-up 冒烟测试（轮子必须架空！）。

三种用法：

    # 1) 只看计划，不动车（默认）
    python -m hardware.smoke_test --url http://192.168.1.50:8000

    # 2) 四动作方向验收
    python -m hardware.smoke_test --url http://192.168.1.50:8000 --confirm

    # 3) 逐级找"最低可靠启动命令"（只测一个方向，重复 N 次让操作者判断可靠性）
    python -m hardware.smoke_test --url http://192.168.1.50:8000 --confirm \
        --only forward --mag 0.35 --repeat 5

PWM 参考（Pikachu Nano fast 模式固件：drivePwm=70, turnPwm=60）：
    V=0.30 -> drive 约 21/255 ；W=0.30 -> turn 约 18/255
    这两个值可能不足以让真实电机可靠起转，所以要逐级往上试，不要一次跳很多。
    本工具只负责按你给的幅值反复发指令，**可靠性必须靠肉眼观察**（有没有转起来）。
"""
from __future__ import annotations

import argparse
import sys
import time

from hardware.pikachu_bridge import http_json

# 方向 -> (V 方向, W 方向, 期望现象)
ACTION_DEFS = {
    "forward":  (+1.0, 0.0, "两个轮子同时向前"),
    "backward": (-1.0, 0.0, "两个轮子同时向后"),
    "left":     (0.0, +1.0, "原地逆时针（车头向左转）"),
    "right":    (0.0, -1.0, "原地顺时针（车头向右转）"),
}


def build_plan(only: str, mag: float, repeat: int, hold_s: float, stop_s: float):
    """返回 [(label, V, W, duration, expected)]。"""
    seq = [("STOP", 0.0, 0.0, stop_s, "两个轮子停住")]
    names = list(ACTION_DEFS) if only == "all" else [only]
    for name in names:
        vs, ws, expect = ACTION_DEFS[name]
        for k in range(repeat):
            label = name if repeat == 1 else f"{name}#{k + 1}"
            seq.append((label, vs * mag, ws * mag, hold_s, expect))
            seq.append(("STOP", 0.0, 0.0, stop_s, "两个轮子停住"))
    return seq


def pwm_hint(v: float, w: float, drive_pwm: int, turn_pwm: int):
    """给出两个通道的 PWM 估计（不是固件真值，只是给你一个量级参考）。"""
    return abs(v) * drive_pwm, abs(w) * turn_pwm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pikachu wheels-up 冒烟测试")
    ap.add_argument("--url", required=True, help="例如 http://192.168.1.50:8000")
    ap.add_argument("--confirm", action="store_true",
                    help="真正发送指令。**必须先把轮子架空**")
    ap.add_argument("--only", default="all",
                    choices=["all", *ACTION_DEFS], help="只测某个方向（找最低可靠命令用）")
    ap.add_argument("--mag", type=float, default=0.30, help="动作幅值（归一化）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="每个方向重复几次（每次之间插 STOP），用于判断'是否每次都可靠起转'")
    ap.add_argument("--hold", type=float, default=1.5, help="每个动作持续秒数")
    ap.add_argument("--gap", type=float, default=1.0, help="动作之间的 STOP 秒数")
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=0.08)
    ap.add_argument("--drive-pwm", type=int, default=70, help="固件 drivePwm，用于估算")
    ap.add_argument("--turn-pwm", type=int, default=60, help="固件 turnPwm，用于估算")
    args = ap.parse_args(argv)

    base = args.url.rstrip("/")
    period = 1.0 / args.rate_hz
    plan = build_plan(args.only, args.mag, args.repeat, args.hold, args.gap)

    print("=" * 78)
    print("Pikachu wheels-up 冒烟测试")
    print(f"  url={base}  mag={args.mag}  only={args.only}  repeat={args.repeat}  "
          f"rate={args.rate_hz}Hz  timeout={args.timeout}s  confirm={args.confirm}")
    print(f"  PWM 参考: drivePwm={args.drive_pwm} turnPwm={args.turn_pwm}  "
          f"(V={args.mag:.2f} -> drive≈{args.mag * args.drive_pwm:.0f}/255, "
          f"W={args.mag:.2f} -> turn≈{args.mag * args.turn_pwm:.0f}/255)")
    print("=" * 78)
    print("\n计划：")
    for label, v, w, dur, expect in plan:
        d_pwm, t_pwm = pwm_hint(v, w, args.drive_pwm, args.turn_pwm)
        print(f"  {label:<10} V={v:+.2f} W={w:+.2f}  {dur:.1f}s   "
              f"PWM≈(drive {d_pwm:4.1f}, turn {t_pwm:4.1f})/255   {expect}")
    if not args.confirm:
        print("\n[plan only] 未加 --confirm，不发送任何指令。")
        print("确认轮子已架空了再加 --confirm 真跑。")
        return 0

    # ---- 预检 ----
    print("\n[preflight]")
    try:
        _, st = http_json(base + "/api/status", None, args.timeout)
    except Exception as exc:                                       # noqa: BLE001
        print(f"  FAIL GET /api/status -> {type(exc).__name__}: {exc}")
        return 1
    if st.get("guard_mode"):
        print("  FAIL guard_mode=true，先在网页上关掉 guard 模式")
        return 1
    print(f"  /api/status ok  guard_mode=false  serial_open={bool(st.get('open'))}")

    _, rc = http_json(base + "/api/reconnect", {}, args.timeout)
    rst = rc.get("status") or {}
    if not rc.get("ok") or not rst.get("open"):
        print(f"  FAIL /api/reconnect ok={rc.get('ok')} open={rst.get('open')} "
              f"err={rst.get('last_error')}")
        return 1
    print(f"  /api/reconnect ok  port={rst.get('port')} baud={rst.get('baud')}")

    def send(v: float, w: float):
        try:
            status, body = http_json(base + "/api/drive", {"v": v, "w": w},
                                     args.timeout)
        except Exception as exc:                                   # noqa: BLE001
            return False, 0, {}, f"{type(exc).__name__}: {exc}"
        st = body.get("status") or {}
        if status == 409:
            return False, status, body, "409 guard mode"
        if not body.get("ok"):
            return False, status, body, f"json ok=false (http {status})"
        if not st.get("open"):
            return False, status, body, f"serial not open ({st.get('last_error')})"
        return True, status, body, ""

    # ---- 动作 ----
    print("\n[run]  请盯着轮子，记录每个动作是否真的按期望方向转起来了")
    failures = 0
    for label, v, w, dur, expect in plan:
        n_ok = 0
        t_end = time.perf_counter() + dur
        while time.perf_counter() < t_end:
            ok, status, body, note = send(round(v, 2), round(w, 2))
            if ok:
                n_ok += 1
            else:
                failures += 1
                print(f"  {label:<10} FAIL http={status} {note}")
                if failures >= 3:
                    break
            time.sleep(period)
        print(f"  {label:<10} V={v:+.2f} W={w:+.2f}  发送 {n_ok} 次   {expect}")
        if failures >= 3:
            break

    # ---- 无论如何收尾 STOP ----
    print("\n[final stop]")
    for _ in range(3):
        ok, status, body, note = send(0.0, 0.0)
        print(f"  STOP -> {'ok' if ok else f'FAIL {note}'}")
        time.sleep(0.05)

    print("\n" + "=" * 78)
    if failures:
        print(f"结果：FAIL（{failures} 次发送失败）")
        return 1
    print("发送链路 PASS。")
    print("请人工填写：")
    for name, _, expect in [(k, *v[1:]) for k, v in ACTION_DEFS.items()]:
        print(f"  {name:<9} 是否正确? [ ]   现象: ______________   ({expect})")
    print(f"  最低可靠启动命令 minimum_reliable_command = ________  "
          f"(本次测试幅值 {args.mag})")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
