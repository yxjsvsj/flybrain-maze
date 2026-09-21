"""Pikachu Bot wheels-up 冒烟测试（轮子必须架空！）。

    # 只看计划，不动车（默认）
    python -m hardware.smoke_test --url http://192.168.1.50:8000

    # 真的动（轮子架空）
    python -m hardware.smoke_test --url http://192.168.1.50:8000 --confirm

顺序：预检 -> STOP -> 前进 -> STOP -> 后退 -> STOP -> 左转 -> STOP -> 右转 -> STOP
每个动作期间以 rate_hz 重复发送，保证 Nano 的 400ms 看门狗不断粮。
任何一次发送失败 -> 立刻 STOP 并以非 0 退出。
"""
from __future__ import annotations

import argparse
import sys
import time

from hardware.pikachu_bridge import PikachuConfig, http_json

# (名字, V, W, 持续秒数)
ACTIONS = [
    ("STOP",     0.0,  0.0,  1.0),
    ("forward",  +1.0, 0.0,  1.5),
    ("STOP",     0.0,  0.0,  1.0),
    ("backward", -1.0, 0.0,  1.5),
    ("STOP",     0.0,  0.0,  1.0),
    ("left",     0.0,  +1.0, 1.5),
    ("STOP",     0.0,  0.0,  1.0),
    ("right",    0.0,  -1.0, 1.5),
    ("STOP",     0.0,  0.0,  1.0),
]

EXPECTED = {
    "forward": "两个轮子同时向前",
    "backward": "两个轮子同时向后",
    "left": "原地逆时针（车头向左转）",
    "right": "原地顺时针（车头向右转）",
    "STOP": "两个轮子停住",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pikachu wheels-up 冒烟测试")
    ap.add_argument("--url", required=True, help="例如 http://192.168.1.50:8000")
    ap.add_argument("--confirm", action="store_true",
                    help="真正发送指令。**必须先把轮子架空**")
    ap.add_argument("--mag", type=float, default=0.30, help="动作幅值（归一化）")
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=0.08)
    args = ap.parse_args(argv)

    base = args.url.rstrip("/")
    period = 1.0 / args.rate_hz

    print("=" * 74)
    print("Pikachu wheels-up 冒烟测试")
    print(f"  url={base}  mag={args.mag}  rate={args.rate_hz}Hz  "
          f"timeout={args.timeout}s  confirm={args.confirm}")
    print("=" * 74)
    print("\n计划：")
    for name, V, W, dur in ACTIONS:
        v = V * args.mag
        w = W * args.mag
        print(f"  {name:<9} V={v:+.2f} W={w:+.2f}  {dur:.1f}s   期望：{EXPECTED[name]}")
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
    print("\n[run]")
    failures = 0
    for name, Vs, Ws, dur in ACTIONS:
        v, w = Vs * args.mag, Ws * args.mag
        n_ok = 0
        t_end = time.perf_counter() + dur
        while time.perf_counter() < t_end:
            ok, status, body, note = send(round(v, 2), round(w, 2))
            if ok:
                n_ok += 1
            else:
                failures += 1
                print(f"  {name:<9} FAIL http={status} {note}")
                if failures >= 3:
                    break
            time.sleep(period)
        print(f"  {name:<9} V={v:+.2f} W={w:+.2f}  发送 {n_ok} 次 ok"
              f"   {EXPECTED[name]}")
        if failures >= 3:
            break

    # ---- 无论如何收尾 STOP ----
    print("\n[final stop]")
    for _ in range(3):
        ok, status, body, note = send(0.0, 0.0)
        print(f"  STOP -> {'ok' if ok else f'FAIL {note}'}")
        time.sleep(0.05)

    if failures:
        print(f"\n结果：FAIL（{failures} 次发送失败）")
        return 1
    print("\n结果：发送链路 PASS。请人工确认四个动作的方向是否正确。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
