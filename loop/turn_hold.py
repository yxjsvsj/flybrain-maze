"""P2d-14：神经转向信号的短时决策保持（P2 离线，最小参数、预先注册）。

输入**只有**基线校正 + EMA 后的 DNa02 神经转向信号；方向与时机完全由神经活动决定，
不读地图/目标/Dijkstra/ToF 左右距离来指定方向；不默认左/右。
允许短时迟滞/积分来"保持"神经**已经**产生的转向决定。front_safety 保留。

人工参数（预先固定、明确披露，不按 seed 调参、无网格搜索）：
    HOLD_THRESH = 0.30   神经(校正+EMA)幅度达到它才算"有方向证据"
    HOLD_S      = 0.60   保持时长（秒）；期间维持已锁定的方向
    HOLD_LEVEL  = 1.00   保持期间施加的转向幅度（±）
若神经无足够证据（|x|<THRESH），**原样放行 x（通常≈0）**，绝不发明方向。
这不是果蝇天然运动记忆，是**人工**的短时保持机制。
"""
from __future__ import annotations

import math

import numpy as np

from loop.dn_steering import TAU_S, DNReadout
from world.maze import Maze

HOLD_THRESH = 0.30
HOLD_S = 0.60
HOLD_LEVEL = 1.00


class HoldReadout:
    """基线校正+EMA 的 DNa02 转向 -> 短时方向保持 -> turn_cmd。"""

    def __init__(self, dt, *, b0=0.0, tau_s=TAU_S, thresh=HOLD_THRESH, hold_s=HOLD_S,
                 level=HOLD_LEVEL):
        self.rd = DNReadout(dt, b0=b0, tau_s=tau_s)
        self.thresh = float(thresh)
        self.level = float(level)
        self.hold_steps = max(1, int(round(hold_s / dt)))
        self.hold_left = 0
        self.hold_dir = 0.0
        self.activations = 0

    def reset(self):
        self.rd.reset()
        self.hold_left = 0
        self.hold_dir = 0.0
        self.activations = 0

    def __call__(self, raw_turn) -> float:
        x = self.rd(raw_turn)                       # 基线校正 + EMA（神经）
        if abs(x) >= self.thresh:                   # 神经给出了方向证据 -> 锁定/续锁
            d = 1.0 if x > 0 else -1.0
            if d != self.hold_dir:
                self.activations += 1
            self.hold_dir = d
            self.hold_left = self.hold_steps
        if self.hold_left > 0:
            self.hold_left -= 1
            return float(np.clip(self.hold_dir * self.level, -1.0, 1.0))
        return float(x)                             # 无强证据：不发明方向


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
