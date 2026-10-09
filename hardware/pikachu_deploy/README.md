# Pikachu Bot 部署（HW-shadow-v1 + P2c commissioning 快照）

从树莓派实机抓取的部署快照，供 `HW-shadow-v1` tag 复现。目标机：MainsailOS 3.0.0
（Debian 13 trixie, aarch64）。Pikachu 应用装在 `/home/bjlm/pikachu/`，
**Klipper / Moonraker 原样保留，不重装不卸载**。

## 文件

| 文件 | 目标位置 | 作用 |
|---|---|---|
| `launch.sh` | `/home/bjlm/pikachu/launch.sh` | 带串口守卫的**前台**启动器（systemd `ExecStart`） |
| `pikachu-web.service` | `/etc/systemd/system/` | Flask 服务，`Restart=on-failure` |
| `pikachu-serial-recover.service` | `/etc/systemd/system/` | CH340 ADD 后的一次性串口恢复（oneshot） |
| `serial_recover.py` | `/home/bjlm/pikachu/` | 恢复逻辑：延迟 1s + 重试 + 校验 `ok`/`open` |
| `serial_removed.sh` | `/home/bjlm/pikachu/` | udev REMOVE 回调：仅记录，不做任何操作 |
| `usb_watch.py` | `/home/bjlm/pikachu/` | pyudev 只读监视器（诊断 add/remove） |
| `start_both_inner.sh` | `/home/bjlm/pikachu/` | 启动 odom（`--swap-lr --wheel-diam 0.067 --track 0.194`）+ ToF 遥测发送端 |
| `99-pikachu-serial.rules` | `/etc/udev/rules.d/` | REMOVE 记录 / ADD 触发恢复 |

## 依赖

- 应用源码 `Pikachu_Bot`（`software/web`）
- venv `/home/bjlm/pikachu/venv`（flask、pyserial、pyudev）
- 运行用户 `bjlm`（在 `dialout` 组）
- 串口：`/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`（CH340, `1a86:7523`）

## 安装

```bash
sudo cp pikachu-web.service pikachu-serial-recover.service /etc/systemd/system/
sudo cp 99-pikachu-serial.rules /etc/udev/rules.d/
sudo cp launch.sh serial_recover.py serial_removed.sh usb_watch.py start_both_inner.sh /home/bjlm/pikachu/
sudo chmod +x /home/bjlm/pikachu/{launch.sh,serial_removed.sh,serial_recover.py,usb_watch.py,start_both_inner.sh}
# 共享状态目录（odom/tof state.json 供网页读），开机自建属 bjlm：
echo 'd /run/pikachu 0775 bjlm bjlm -' | sudo tee /etc/tmpfiles.d/pikachu.conf
sudo systemd-tmpfiles --create /etc/tmpfiles.d/pikachu.conf
sudo systemctl daemon-reload
sudo systemctl enable --now pikachu-web
sudo udevadm control --reload-rules
```

`start_both_inner.sh` 的里程计用 **P2c 落地标定的有效参数**（`D=0.067 m`、
`L=0.194 m`）；Windows 脑主机 IP 用 `PIKACHU_WIN_IP` 覆盖。in-place 启动（无参数）
之前必须先 `systemd-tmpfiles --create`，否则 sender 写不了 `/run/pikachu`。

`launch.sh` 的串口守卫：只接受 `/dev/serial/by-id/...`；拒绝 `*Klipper_*` 设备；
设备必须存在；VID:PID 必须是 `1a86:7523`（`PIKACHU_SKIP_VIDPID=1` 可跳过）；HTTP
端口必须空闲。

## 安全语义

- **掉线**：udev REMOVE 仅记录 → 串口写自然失败 → bridge 达到 `max_failures` →
  **latched FAILSAFE**（只发 STOP）→ Nano 自身 `COMMAND_TIMEOUT_MS=400` 物理停车。
- **恢复**：udev ADD → oneshot 延迟 1 秒 → `POST /api/reconnect`（校验 `ok` 且
  `status.open`）→ **仅恢复通信**。
- **运动永不自动恢复**：bridge 的 FAILSAFE latch 不解除，必须人工重新启动
  `loop.shadow_run`。

## 验收（实测通过）

| # | 场景 | 结果 |
|---|---|---|
| 1 | SIGKILL `pikachu-web` | systemd 拉起（`NRestarts`+1，active）；bridge 保持 FAILSAFE，不续跑 |
| 2 | 影子运动中断开 CH340 | bridge 转 FAILSAFE；Nano ≤400ms 停车 |
| 3 | 重新插入 CH340 | udev ADD → `add-recover OK`，`open=true`；bridge 仍 latch |
| 4 | 恢复运动 | 只能人工重启 `shadow_run` |
