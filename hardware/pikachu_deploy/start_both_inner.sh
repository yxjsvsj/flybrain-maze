#!/usr/bin/env bash
# Pikachu 传感器发送端启动器（P2c ground commissioning 快照）。
#
# 启动两路遥测（setsid 脱离当前会话）：
#   odometry : /usr/bin/python3（Pi 系统 python，有 libgpiod）→ UDP 8888
#   tof5     : ./venv/bin/python（有 blinka）→ UDP 8889
# 两者都原子写共享状态 /run/pikachu/{odom,tof}.json 供本机 Ground Control 网页读。
# （/run/pikachu 由 /etc/tmpfiles.d/pikachu.conf 开机自建，属 bjlm。）
#
# 地面标定的有效里程计参数（P2c）：
#   wheel diameter D = 0.067 m   （名义 0.070；带载打滑 −4.5%）
#   track width    L = 0.194 m   （名义 0.160；原地转打滑 +21%）
#
# Windows 脑主机 IP 用环境变量覆盖：
#   PIKACHU_WIN_IP=192.168.0.2 bash start_both_inner.sh
cd /home/bjlm/pikachu
WIN="${PIKACHU_WIN_IP:-<windows-ip>}"
pkill -f 'odometry.py' 2>/dev/null || true
pkill -f 'tof5.py' 2>/dev/null || true
sleep 1.5
setsid bash -c "/usr/bin/python3 -u odometry.py --e1 22 23 --e2 17 27 --seconds 0 \
  --swap-lr --wheel-diam 0.067 --track 0.194 --udp ${WIN}:8888 --state-file /run/pikachu/odom.json; \
  echo \"[odom EXIT \$?] \$(date +%T)\"" \
  > /home/bjlm/pikachu/odom_run.log 2>&1 < /dev/null &
sleep 1
setsid bash -c "./venv/bin/python -u tof5.py --udp ${WIN}:8889 \
  --state-file /run/pikachu/tof.json; echo \"[tof EXIT \$?] \$(date +%T)\"" \
  > /home/bjlm/pikachu/tof_run.log 2>&1 < /dev/null &
sleep 3
pgrep -af 'odometry.py|tof5.py' | grep -v pgrep || echo NONE
