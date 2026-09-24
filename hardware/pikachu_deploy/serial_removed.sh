#!/usr/bin/env bash
# udev REMOVE 回调：仅记录设备消失，不做任何 reconnect / close。
echo "$(date '+%F %T') udev REMOVE ch340 (no action; serial writes will fail -> bridge FAILSAFE; Nano 400ms timeout stops motors)" >> /home/bjlm/pikachu/serial_events.log
