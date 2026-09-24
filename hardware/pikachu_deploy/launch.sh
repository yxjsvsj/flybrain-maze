#!/bin/bash
# Pikachu Bot web 启动器（带串口守卫）。
# 这块 Pi 同时跑 Klipper，Klipper 的 MCU 也是 USB 串口。
# 一旦 Pikachu 打开错的串口，可能把正在打印的机器搞乱，所以启动前做硬校验。
# 用法: ./launch.sh            # 前台
#       ./launch.sh --detach   # 后台
set -u

BASE=/home/bjlm/pikachu
REPO=$BASE/Pikachu_Bot
VENV=$BASE/venv
PORT_BYID=${PIKACHU_PORT:-/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0}
HTTP_PORT=${PIKACHU_HTTP_PORT:-8000}
EXPECT_VIDPID=${PIKACHU_EXPECT_VIDPID:-1a86:7523}
LOG=$BASE/pikachu_web.log

fail() { echo "REFUSE: $*" >&2; exit 1; }

echo "=== guard ==="

case "$PORT_BYID" in
  /dev/serial/by-id/*) : ;;
  *) fail "PIKACHU_PORT 必须是 /dev/serial/by-id/... 路径，当前: $PORT_BYID" ;;
esac

case "$PORT_BYID" in
  *Klipper_*) fail "这是 Klipper MCU 的串口（$PORT_BYID），拒绝打开" ;;
esac

[ -e "$PORT_BYID" ] || fail "$PORT_BYID 不存在（Nano 没插好？先 ls -l /dev/serial/by-id/）"

REAL=$(readlink -f "$PORT_BYID")
VID=$(udevadm info -q property -n "$REAL" 2>/dev/null | sed -n 's/^ID_VENDOR_ID=//p')
PID=$(udevadm info -q property -n "$REAL" 2>/dev/null | sed -n 's/^ID_MODEL_ID=//p')
GOT="${VID}:${PID}"
if [ "${PIKACHU_SKIP_VIDPID:-0}" != "1" ]; then
  [ "$GOT" = "$EXPECT_VIDPID" ] || fail "VID:PID 不符：期望 $EXPECT_VIDPID 实际 $GOT（换板子设 PIKACHU_EXPECT_VIDPID，或 PIKACHU_SKIP_VIDPID=1）"
fi
echo "port      : $PORT_BYID -> $REAL  (VID:PID $GOT)"

if ss -ltn | grep -q ":$HTTP_PORT "; then
  fail "端口 $HTTP_PORT 已被占用；先停掉旧的 Pikachu 进程或换端口"
fi
echo "http port : $HTTP_PORT (free)"
echo "klipper   : $(systemctl is-active klipper)   moonraker: $(systemctl is-active moonraker)"

echo "=== start ==="
cd "$REPO/software/web"
CMD=("$VENV/bin/python" app.py --port "$PORT_BYID" --baud 115200 --host 0.0.0.0 --port-http "$HTTP_PORT")

if [ "${1:-}" = "--detach" ]; then
  setsid nohup "${CMD[@]}" < /dev/null > "$LOG" 2>&1 &
  sleep 3
  echo "--- procs ---"
  pgrep -af "app.py" | head -3 || true
  echo "log: $LOG"
else
  exec "${CMD[@]}"
fi
