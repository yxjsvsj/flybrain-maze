"""全局配置。要调参就改这里，不要在 loop/ 里塞魔法数字。

单位约定：长度以"迷宫格"为单位（1 格 = 1.0），速度格/秒，角度弧度。
"""
from dataclasses import dataclass, field


@dataclass
class CarConfig:
    radius: float = 0.22          # 车体半径，必须 < 0.5 才能在 1 格宽走廊里走
    max_speed: float = 0.9        # 最大前进速度 (格/秒)
    max_omega: float = 2.6        # 最大角速度 (rad/s)
    physics_dt: float = 0.001     # 物理子步长；脑每步会拆成多个子步
    escape_back_frac: float = 0.30   # 触发逃逸时后退速度占 max_speed 的比例


@dataclass
class EncoderConfig:
    """传感器 -> 感觉神经元。

    所有 gain 的单位都是"稳态电压增量"（LIF 的 v 会衰减，稳态增量 = 每步注入/(1-decay)），
    不是每步注入电压。这样和 dt 解耦，换步长不用重调。
    """
    n_rays: int = 15
    fov_deg: float = 240.0        # 果蝇视野接近全景；车用 240 度
    max_range: float = 6.0
    wall_size: float = 0.5        # 墙的"视角大小"系数：size = wall_size / 距离
    loom_size_gain: float = 2.0   # size -> LPLC2 稳态电压
    loom_growth_gain: float = 1.0  # size 变化率(size/秒) -> LPLC2
    threat_gain: float = 1.5      # size 变化率 -> LC4（快速逼近）
    threat_thresh: float = 1.0    # size/秒，超过才算"逼近"
    cap: float = 2.5              # 单通道稳态电压增量上限
    # LC10a 是转向通路（LC10a -> DNa02），要拿"静态 proximity"而不是 looming，
    # 所以 size 和 growth 分开给。真脑预设里 loom_size_gain=0 / chase_size_gain=2.0。
    chase_size_gain: float = 0.0    # size -> LC10a
    chase_growth_gain: float = 0.0  # size 变化率 -> LC10a
    # 避障只用前向扇区，避免左后方/右后方的近墙主导转向。
    # 240° 全部射线仍保留给建图用。
    avoidance_fov_deg: float = 120.0  # 参与避障的总张角（±60°）
    front_cone_deg: float = 40.0      # "正前方"锥角（±20°），用于安全限速
    # 常量注入到前进神经元（DNg100）。模型静息时 DNg100 只有 0.25 Hz，连接组里没有
    # "自发前进"这个指令——Eon 的 embodied fly 同样是手工激活 oDN1 让虚拟果蝇走起来的。
    # 这是全项目唯一一处"凭空加的信号"，别把它当成连接组的产物。
    forward_drive_dv: float = 0.0


@dataclass
class DecoderConfig:
    """下行神经元 -> (v, omega)。发放率先除以 ref_hz 归一化，增益无量纲。"""
    window_s: float = 0.25        # 发放率滑窗
    ref_hz: float = 10.0          # 归一化参考发放率
    turn_gain: float = 0.8        # (r_turnL - r_turnR) -> omega / max_omega
    turn_sign: float = 1.0        # 转向相反就改成 -1
    fwd_gain: float = 0.6         # r_fwd -> v / max_speed
    back_gain: float = 0.6        # r_back -> 后退
    deadzone: float = 0.15        # 归一化死区，抑制噪声引起的漂移
    baseline_hz: float = 0.0      # 静息发放率，先减掉再归一化（真脑静息不是 0）
    escape_thresh: float = 1.0    # r_escape 超过它才算触发逃逸（= ref_hz 的 1 倍）
    escape_hold_s: float = 0.6    # 逃逸冷却，防止"倒车-撞墙-再逃逸"的抖动
    escape_omega_frac: float = 0.9  # 逃逸时 omega / max_omega
    stall_after_s: float = 0.30   # 想走但走不动超过这么久 -> 判定卡死
    stall_hold_s: float = 0.50    # 一旦判定卡死，至少持续这么久（否则会"倒退-前进"抖动）
    stall_back_frac: float = 0.35   # 卡死时后退速度 / max_speed
    stall_omega_frac: float = 1.0   # 卡死时 omega / max_omega
    stall_dir: float = -1.0       # 正对墙时左右 loom 对称，需要一个固定偏好方向
    # 卡死时的转向策略：
    #   "spin"     固定偏好方向满舵盲转（原始脱困行为，假脑靠它 4/4，真脑 0/4）
    #   "hold"     保持当前转向（转向听规划器，真脑 2/4，假脑 0/4）
    #   "straight" 直退不转（最中性）
    stall_turn_mode: str = "spin"
    smooth: float = 0.35          # 输出低通系数，0 = 不平滑

    # "dn"      = 读下行神经元（真·连接组在环里）
    # "scripted" = 手写贴墙基线控制器，完全绕过脑，用来做消融实验
    # "readout"  = 用训练好的线性读出头读全部下行神经元（1314 个）
    mode: str = "dn"
    follow_side: str = "R"        # scripted 模式贴哪一侧墙
    follow_distance: float = 0.9  # 想维持的墙距（格）。注意用距离不用 proximity——
                                  # proximity 是 max_range 归一化的，走廊里设定点根本不可达
    follow_gain: float = 0.6      # (目标距离 - 实际距离) -> 归一化转向
    follow_speed_frac: float = 0.75

    readout_path: str = ""        # 训练好的 .npz（loop/train_readout.py 产出）
    readout_tau: float = 0.2      # 读出头特征的脉冲痕迹时间常数
    readout_fwd_frac: float = 0.75  # readout 模式下固定前进速度占比

    # 记忆层（nav/memory.py）给的"目标方位"偏置。两者可分别消融：
    #   brain_gain=1, memory_gain=0  -> 纯反射（现状，1/6）
    #   brain_gain=0, memory_gain=1  -> 纯规划
    #   brain_gain=1, memory_gain=1  -> 脑管避障，记忆管往哪走
    brain_gain: float = 1.0
    memory_gain: float = 0.0
    turn_slowdown: float = 0.5    # 急转时降速：v *= (1 - k*|turn|)，减少撞墙

    # 前方安全限速层：正前方快撞墙时降速/停车。**只限速，不决定方向**——
    # 决定方向是果蝇脑的活，不能用手写规则顶替。
    front_safety: bool = True
    front_slow_dist: float = 1.0   # 正前方墙距小于它开始线性降速
    front_stop_dist: float = 0.25  # 小于它禁止继续前进


@dataclass
class BrainConfig:
    kind: str = "fake"            # "fake" | "real"
    dt: float = 0.01              # 脑步长；0.02 是作者标定值，0.01 控制更细
    device: str = "auto"          # "cpu" | "cuda" | "auto"
    sensory_input: bool = False   # False = 切掉感觉神经元上的突触，避免嗅觉自激（作者建议）
    seed: int = 64


@dataclass
class NavConfig:
    """高层导航（nav/memory.py）。"""
    unknown_cost: float = 4.0     # 未知格通行代价。1=很敢穿未知区，越大越保守
    lookahead: float = 0.9        # 纯追踪前瞻距离（格）
    arrive_dist: float = 0.6      # 判定"到达当前路点"的距离，触发重规划


@dataclass
class NoiseConfig:
    """仿真噪声。默认全 0 —— 先保证无噪声版本稳定，再分阶段加。
    上实物前至少要把 range_noise_std 打开测一遍。"""
    range_noise_std: float = 0.0      # 测距高斯噪声标准差（格）
    range_dropout_prob: float = 0.0   # 每条射线丢失概率（返回 max_range）
    pose_xy_noise_std: float = 0.0    # 位姿 xy 噪声
    pose_theta_noise_std: float = 0.0  # 位姿朝向噪声
    motor_tau: float = 0.0            # 电机一阶延迟时间常数（秒），0 = 无延迟


@dataclass
class Config:
    car: CarConfig = field(default_factory=CarConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    nav: NavConfig = field(default_factory=NavConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)


# 假脑和真脑的发放率量级差很多（真脑静息时下行神经元只有 ~0.25-0.75 Hz），增益不能共用。
#
# 真脑那套的关键点来自实测（见 README 的 "What we found" 和 baseline 测量）：
#   LPLC2 --(loom)--> DNp01  逃逸      0.25 -> 4.6 Hz
#   LC10a --(目标)--> DNa02  转向      0.50 -> 1.5 Hz（同侧）
# 所以静态的"墙在那边"必须走 LC10a，不能走 LPLC2——否则逃逸会一直开着。
# turn_sign 是猜的：LC10a 是追目标通路，同侧激活应该产生"朝目标转"，
# 避障要反号。不确定就自己试 +1 / -1。
PRESETS = {
    "fake": dict(
        enc=dict(chase_size_gain=0.0, chase_growth_gain=0.0, forward_drive_dv=0.0),
        dec=dict(window_s=0.25, ref_hz=10.0, baseline_hz=0.0, turn_gain=0.8,
                 fwd_gain=0.6, back_gain=0.6, escape_thresh=1.0, deadzone=0.15,
                 turn_sign=1.0, stall_turn_mode="spin"),
    ),
    # DNa02 每侧只有 1 个神经元，发放率被量化成 1/window_s 一跳（0.4s -> 2.5 Hz），
    # 所以 deadzone 基本不起作用，差值总是远大于它。turn_gain 实测 2.0 最好
    # （track: 0 碰撞 0 卡死）。静态 proximity 走 chase（LC10a），LPLC2 只留 growth，
    # 否则逃逸一直开着。
    # 已知弱点：2 格宽的 corridor 上无论怎么调都崩（覆盖率 ~5%，40+ 次碰撞/卡死）。
    # 单神经元指令的量化精度不够，要解决得换群体读出头。
    "real": dict(
        enc=dict(chase_size_gain=2.0, chase_growth_gain=0.0, forward_drive_dv=0.5,
                 loom_size_gain=0.0, loom_growth_gain=0.5),
        dec=dict(window_s=0.4, ref_hz=4.0, baseline_hz=0.3, turn_gain=2.0,
                 fwd_gain=3.0, back_gain=1.5, escape_thresh=0.4, deadzone=0.15,
                 turn_sign=-1.0, stall_turn_mode="hold"),
    ),
}


def apply_preset(cfg: "Config", kind: str) -> None:
    p = PRESETS[kind]
    for k, v in p["enc"].items():
        setattr(cfg.encoder, k, v)
    for k, v in p["dec"].items():
        setattr(cfg.decoder, k, v)
