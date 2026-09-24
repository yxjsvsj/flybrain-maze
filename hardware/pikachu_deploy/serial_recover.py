#!/usr/bin/env python3
"""CH340 ADD 后恢复**通信**（由 udev -> pikachu-serial-recover.service 触发）。

流程：等设备节点稳定 -> POST /api/reconnect -> 校验 JSON ok 且 status.open。

边界（安全语义，勿改）：
- 只恢复串口通信，**绝不**解除 bridge 的 FAILSAFE latch。
- 运动恢复必须人工重新启动 shadow_run。
"""
import json
import os
import sys
import time
import urllib.request

API = os.environ.get("PIKACHU_API", "http://127.0.0.1:8000")
DELAY = float(os.environ.get("PIKACHU_RECOVER_DELAY", "1.0"))
RETRIES = int(os.environ.get("PIKACHU_RECOVER_RETRIES", "5"))
LOG = "/home/bjlm/pikachu/serial_events.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def reconnect() -> dict:
    req = urllib.request.Request(
        API + "/api/reconnect", data=b"{}", method="POST",
        headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=3) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


def main() -> int:
    time.sleep(DELAY)                       # 等 tty/ttyUSB 节点与 by-id 符号链接稳定
    for attempt in range(1, RETRIES + 1):
        try:
            body = reconnect()
        except Exception as exc:            # noqa: BLE001
            log(f"add-recover attempt {attempt}: {type(exc).__name__}: {exc}")
            time.sleep(1)
            continue
        st = body.get("status") or {}
        if body.get("ok") and st.get("open"):
            log(f"add-recover OK (attempt {attempt}) port={st.get('port')} "
                f"（通信已恢复；bridge latch 不解除）")
            return 0
        log(f"add-recover attempt {attempt}: ok={body.get('ok')} "
            f"open={st.get('open')} err={st.get('last_error')}")
        time.sleep(1)
    log("add-recover FAILED after retries")
    return 1


if __name__ == "__main__":
    sys.exit(main())
