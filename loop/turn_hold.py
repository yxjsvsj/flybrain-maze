"""P2d-14：神经转向信号的短时决策保持（P2 离线，最小参数、预先注册）。

输入**只有**基线校正 + EMA 后的 DNa02 神经转向信号；方向与时机完全由神经活动决定，
不读地图/目标/Dijkstra/ToF 左右距离来指定方向；不默认左/右。
保持机制只"续持"神经**已经**产生的决定：保持**神经触发时的实际方向和幅度**，
计时**有界**（≤ HOLD_S）。持续强神经输入**直接**控制转向，不因保持升级为满舵。
front_safety 保留。

人工参数（预先固定、明确披露，不按 seed 调参、无网格搜索）：
    HOLD_THRESH  = 0.30   神经(校正+EMA)幅度达到它才算"有方向证据"
    HOLD_S       = 0.60   证据消失后最多再保持这么久（秒）
    HOLD_MAG_CAP = 1.00   保持幅度的上限（保持用的是神经实际 |x|，绝不放大到满舵）
若神经无足够证据（|x|<THRESH 且保持已到期），**原样放行 x（通常≈0）**，绝不发明方向。
这不是果蝇天然运动记忆，是**人工**的短时保持机制。
"""
from __future__ import annotations

import math

import numpy as np

from loop.dn_steering import TAU_S, DNReadout
from world.maze import Maze

HOLD_THRESH = 0.30
HOLD_S = 0.60
HOLD_MAG_CAP = 1.00


class HoldReadout:
    """基线校正+EMA 的 DNa02 转向 -> 有界短时方向/幅度保持 -> turn_cmd。"""

    def __init__(self, dt, *, b0=0.0, tau_s=TAU_S, thresh=HOLD_THRESH, hold_s=HOLD_S,
                 mag_cap=HOLD_MAG_CAP):
        self.rd = DNReadout(dt, b0=b0, tau_s=tau_s)
        self.thresh = float(thresh)
        self.mag_cap = float(mag_cap)
        self.hold_steps = max(1, int(round(hold_s / dt)))
        self.hold_left = 0
        self.hold_dir = 0.0
        self.hold_mag = 0.0
        self.activations = 0            # 保持"接管"（证据消失后进入保持）的次数
        self._held = False
        self._was_strong = False

    def reset(self):
        self.rd.reset()
        self.hold_left = 0
        self.hold_dir = 0.0
        self.hold_mag = 0.0
        self.activations = 0
        self._held = False
        self._was_strong = False

    def __call__(self, raw_turn) -> float:
        x = self.rd(raw_turn)                       # 基线校正 + EMA（神经）
        if abs(x) >= self.thresh:
            # 神经有方向证据：由**神经本身**控制；记录其方向+实际幅度并重新武装有界计时
            self.hold_dir = 1.0 if x > 0 else -1.0
            self.hold_mag = min(abs(x), self.mag_cap)       # 实际幅度，不升级为满舵
            self.hold_left = self.hold_steps
            self._was_strong = True
            self._held = False
            return float(x)
        # 证据不足：若保持未到期，续持"最后一次神经决定"的方向与幅度（有界）
        self._was_strong = False
        if self.hold_left > 0:
            self.hold_left -= 1
            if not self._held:
                self.activations += 1
                self._held = True
            return float(np.clip(self.hold_dir * self.hold_mag, -1.0, 1.0))
        self._held = False
        return float(x)                             # 不发明方向


def required_omega_sign(rows) -> float:
    """方向坐标审计：从**起点位姿**到目标方位求所需 omega 符号（不参与控制，仅评估/对账）。

    约定：maze 行号 y 向下增大；world.car 正 omega 使 theta 增大 -> 朝向转向 +y。
    故 must_left(开口在 -y) 需要负 omega；must_right(开口在 +y) 需要正 omega。
    """
    m = Maze(rows)
    sx, sy, sth = m.start
    if m.goal is None:
        return 0.0
    gx, gy = m.goal[0], m.goal[1]
    err = (math.atan2(gy - sy, gx - sx) - sth + math.pi) % (2 * math.pi) - math.pi
    return float(np.sign(err))


def make_source(variant, b0, dt):
    """返回 (turn_source(ctl)->turn, stateful_obj|None)。"""
    if variant == "DN_ema":
        o = DNReadout(dt, b0=b0)
        return (lambda c: o(c.last_brain_turn)), o
    if variant == "DN_hold":
        o = HoldReadout(dt, b0=b0)
        return (lambda c: o(c.last_brain_turn)), o
    if variant == "DN_zero":
        return (lambda c: 0.0), None
    if variant == "DN_swap":
        o = DNReadout(dt, b0=b0)
        return (lambda c: -o(c.last_brain_turn)), o
    raise KeyError(variant)
