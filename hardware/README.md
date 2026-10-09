# Hardware — P2 physical platform

> 实体平台是 **two-wheel differential drive with a passive front caster**
> （后双驱动轮 + 前被动导向轮）。**不是四驱车，也不是三轮驱动。**
>
> 软件总览见根目录 [`README.md`](../README.md)；本文件只描述硬件、接线、端口与安全语义。

## 底盘拓扑

```text
          前方
           ○
       被动导向轮（caster，不驱动）
            │

    [ VL53L0X × 5 ]

        车体 / Pi 5

 MG513 L             MG513 R
   O===================O
         后驱动轴
```

## 部件清单

| 部件 | 当前实物 |
|---|---|
| 底盘 | R6 圆形底盘 |
| 驱动结构 | 后双轮差速 + 前被动导向轮（caster）|
| 驱动电机 | 2 × WHEELTEC MG513P30_12V（12V，30:1）|
| 编码器 | AB quadrature，实测 x4 CPR = 1560 |
| 电机驱动 | D153C / TB6612FNG 双路 |
| MCU | Arduino Nano（CH340 USB 串口）|
| SBC | Raspberry Pi 5（MainsailOS / Debian trixie aarch64）|
| 距离传感 | 5 × VL53L0X |
| ToF 方位角 | +70°, +30°, 0°, −30°, −70° |
| Brain host | Windows + RTX 4060 Laptop |
| 里程计 | Pi GPIO + libgpiod |
| 控制链 | HTTP → Pi → Nano → TB6612 |

> 注意：`world/car.py`（P1 仿真）是抽象 differential-drive 模型，**不对应实体轮数**；
> 实体结构以本文件为准。

## 接线

### Arduino Nano → TB6612FNG（`hardware/firmware/nano_car/nano_car.ino`）

| 信号 | Nano 引脚 |
|---|---|
| PWMA | D5 |
| PWMB | D6 |
| AIN1 | D11 |
| AIN2 | D4 |
| BIN1 | D7 |
| BIN2 | D3 |
| STBY | 硬件接 5V（固件 `STBY=-1`）|
| LED1 / LED2 | A2 / A3 |
| RELAY1 / RELAY2 | A4 / A5 |

混控：`motorA = V + W`，`motorB = V − W`；`W > 0` = 左转。
串口 115200，`V<lin> W<ang>\n`（−1.0…1.0）；`COMMAND_TIMEOUT_MS = 400` 看门狗停车。

### 树莓派编码器 GPIO（`hardware/odometry.py`）

| 通道 | GPIO（BCM）|
|---|---|
| 原始 E1 | 22 / 23 |
| 原始 E2 | 17 / 27 |

> **实体车实测 E1/E2 与物理 left/right 相反** → 生产启动必须 `--swap-lr`。
> E1/E2 是原始通道名，**不等于**物理左右轮。

### ToF（5 × VL53L0X，`hardware/tof5.py`）

| 方位 | XSHUT (BCM) | I2C 地址 | 角度 |
|---|---|---|---|
| L | GPIO5 | 0x30 | +70° |
| FL | GPIO6 | 0x31 | +30° |
| F | GPIO13 | 0x32 | 0° |
| FR | GPIO19 | 0x33 | −30° |
| R | GPIO26 | 0x34 | −70° |

I2C：GPIO2 (SDA) / GPIO3 (SCL)。上电后地址回 0x29 → 每次启动用 XSHUT 逐个重分配。
continuous ranging，~10 Hz。

## 通信

### Windows → Pi → Nano（运动指令）

```text
RealFlyBrain (Windows)  --HTTP POST-->  Pi Flask  --USB serial-->  Arduino Nano  --PWM-->  TB6612
```

| 端点 | 语义 |
|---|---|
| `POST /api/drive {"v","w"}` | 归一化 V/W (−1…1)，`W>0`=左转 |
| `POST /api/manual/drive {"v","w"}` | 网页人工驱动（需先接管 MANUAL）|
| `GET /api/manual/state` | 实时状态（odom/ToF/串口/控制权）|
| `POST /api/reconnect` | 重开串口 |
| `GET /api/status` | 串口/守卫状态（字段在**顶层**）|

Flask 端口 **8000**；串口固定用 `/dev/serial/by-id/...`（VID:PID `1a86:7523`），拒绝 Klipper 设备。

### Pi → Windows（遥测）

| 流 | UDP 端口 | 内容 |
|---|---|---|
| odometry | **8888** | x/y/theta + left/right ticks（session_id / seq）|
| ToF | **8889** | 五路 mm + status + ages（session_id / seq）|

`odometry.py` 用系统 `python3`（libgpiod）；`tof5.py` 用 venv（blinka）。
两者也可用 `--state-file` 原子写共享 JSON（`/run/pikachu/{odom,tof}.json`）供本机网页读取。

## 安全语义

### 控制权 MANUAL / AUTO / E-STOP

| 状态 | `/api/drive`（Windows 链路）| `/api/manual/drive`（网页）|
|---|---|---|
| **AUTO**（默认）| **200 允许** | 403 `manual_not_active` |
| **MANUAL**（网页接管）| **403 `manual_mode_active`** | **200 允许** |
| **E-STOP latched** | **403 `estop_latched`** | **403 `estop_latched`** |

- E-STOP 下 `STOP` / `status` / `clear-estop` **仍然可用**。
- 浏览器**关闭/失焦** → 发 STOP 并释放本页 MANUAL ownership（**只释放网页侧**，不触碰 Windows bridge 自身 latch）。
- **stale / unhealthy** 的 odom/ToF → **禁止 MANUAL 运动**。

### 掉线 / 故障

| 层 | 机制 |
|---|---|
| 服务守护 | systemd `Restart=on-failure` |
| 串口守卫 | 固定 by-id + VID:PID 校验 |
| 发送节流 | latest-wins 单槽邮箱（过期指令不排队）|
| 实时节拍 | pacer 落后超限 → 停车 |
| 熔断 | 连续失败 → **latched FAILSAFE**（只发 STOP，需人工重启）|
| 固件看门狗 | Nano `COMMAND_TIMEOUT_MS=400`（独立于上位机）|
| 运行状态目录 | `/run/pikachu`（tmpfiles.d 开机自建）|

**安全语义**：故障时**通信可自动恢复**（串口重开、服务 systemd 拉起），但**运动永不自动恢复**——
bridge 一旦 FAILSAFE 就 latch，必须人工重启。

## 地面调试工具

| 工具 | 用途 |
|---|---|
| `hardware/ground_commission.py` | 短脉冲手动标定（`--mode pulse`）+ 单墙停车（`--mode wall`）|
| Ground Control 网页 `/ground` | 人工方向键 / 速度 / 脉冲 / 五路 ToF / odom / E-STOP / 单墙测试 |
| `hardware/smoke_test.py` | wheels-up / 落地四方向验收 |
| `hardware/encoder_calib.py` | 编码器标定 |
| `hardware/selftest.py` | 纯逻辑自检（不连硬件）|

## 当前标定状态（preliminary，未冻结）

| 项 | 值 |
|---|---|
| 固件 PWM cap | straight 200 / turn 180（fast）；slow 120 |
| 固件 ramp | 5 PWM / 20 ms（~250 PWM/s）|
| wheels-up 起转门槛 | ≈0.60（**旧固件** drivePwm=70 时）|
| 落地 forward reliable | ≈0.30（1.5s 稳态）|
| 落地 turn reliable | ≈0.40–0.45（右转更弱，≥0.60 才稳）|
| `meters_per_cell` | 0.40（provisional）|
| 已知问题 | VM 电池欠压（~10V）→ 慢且门槛高；右转左轮后退偏弱；短命令受 ramp 拖慢 |

> 以上均为 **commissioning 数据，不是最终控制参数**；step 6（真脑地面闭环）前才冻结。
