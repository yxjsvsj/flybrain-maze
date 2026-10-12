"""P2d-12 任务二：与冻结 Decoder(readout 模式)**同语义**的 P2 控制融合层。

P2d-11 直接在 Decoder 输出后改 omega，忽略了内部 turn_cmd 裁剪、turn_slowdown、front_safety
与 smooth 平滑状态，导致 F/SHAM 与普通增益**不在同一语义下比较**。本层复刻冻结组合
（`loop/decode.py` readout 分支）的：
    v        = max_speed * readout_fwd_frac
    turn     = clip(bg*brain_turn + mg*pursuit_turn, -1, 1)
    omega    = max_omega * turn ;  v *= (1 - turn_slowdown*|turn|)
    front_safety（按 dmin_F）;  平滑 self.v/self.omega = (1-smooth)*prev + smooth*new
并允许对 brain_turn 做 P2 校正/替换（bias / sham / 条件门控）。

**不修改 loop/decode.py。** 供离线消融使用。
"""
from __future__ import annotations

import numpy as np


class P2Fusion:
    def __init__(self, dec_cfg, car_cfg):
        self.dcfg = dec_cfg
        self.ccfg = car_cfg
        self.reset()

    def reset(self):
        self.v = 0.0
        self.omega = 0.0

    def step(self, *, brain_turn: float, pursuit_turn: float = 0.0, dmin_F: float = float("inf"),
             mode: str = "plain", bg: float | None = None, mg: float | None = None,
             bias: float = 0.0, gate_reduce: float = 0.25, gate_thresh: float = 0.02):
        """返回 (v, omega)。mode: plain | bias | sham | cond。"""
        cfg = self.dcfg
        bg = cfg.brain_gain if bg is None else bg
        mg = cfg.memory_gain if mg is None else mg
        bt = float(np.clip(brain_turn, -1.0, 1.0))   # 与冻结一致：先裁剪 brain_turn
        if mode == "bias":                       # 去常驻偏置，保留时变
            bt = bt - bias
        elif mode == "sham":                      # 去掉时变，只留静息常数
            bt = bias
        if mode == "cond" and (bt * pursuit_turn < 0) and abs(pursuit_turn) > gate_thresh:
            bg = bg * gate_reduce                 # 神经与规划器显著反向 -> 降神经权重

        turn_cmd = float(np.clip(bg * bt + mg * pursuit_turn, -1.0, 1.0))
        v = self.ccfg.max_speed * cfg.readout_fwd_frac
        omega = self.ccfg.max_omega * turn_cmd
        v *= 1.0 - cfg.turn_slowdown * abs(turn_cmd)

        front_blocked = False
        if cfg.front_safety and v > 0:
            if dmin_F <= cfg.front_stop_dist:
                v = 0.0
                front_blocked = True
            elif dmin_F < cfg.front_slow_dist:
                ratio = ((dmin_F - cfg.front_stop_dist)
                         / (cfg.front_slow_dist - cfg.front_stop_dist))
                v *= float(np.clip(ratio, 0.0, 1.0))

        # 与冻结 loop/decode.py 一致：front_block 只硬停 **v**；omega 每步都做低通
        a = cfg.smooth
        if front_blocked:
            self.v = 0.0
        else:
            self.v = (1 - a) * self.v + a * v
        self.omega = (1 - a) * self.omega + a * omega
        return self.v, self.omega
