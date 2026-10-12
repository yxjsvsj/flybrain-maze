"""P2d-13：统一的 P2 离线控制接口（公平对照）。

把"神经转向来源"与"运动执行包络"解耦，使对照公平：
  - 相同的基础前进速度（v = max_speed * fwd_frac）
  - 相同的急转降速、输出低通、前方安全停车（同 cfg.decoder 语义）
  - 唯一允许不同的，是 turn_source 产生的归一化转向 [-1,1]

神经前端固定复用一个**冻结 Decoder(mode='dn')**：它负责滚动发放率窗口、脉冲痕迹，
并把单细胞 DNa02 的 turn_L-turn_R 转向暴露在 dec.last_brain_turn 上（本模块不改冻结实现）。
turn_source 只读前端输出，不得引入目标路线/隐藏地图/Dijkstra/直接 ToF 规则。

注意：dec.update() 的返回值（含逃逸/卡死等 dn 模式反射）**本模块一概不用**，
只取其 (rates, last_brain_turn)。原始"DN 同时控制 v/omega"的完整行为作为**独立实验**单独跑。
"""
from __future__ import annotations

import numpy as np


class P2Control:
    """统一包络控制器。update() 与冻结 Decoder 同签名，可直接喂给 run_dynamic。"""

    def __init__(self, cfg, dec, turn_source, name="CTRL", fwd_frac=None):
        self.cfg = cfg                      # DecoderConfig（run_dynamic 会读 cfg.brain_gain 等）
        self.car_cfg = dec.car_cfg
        self.dt = dec.dt
        self.dec = dec                      # 冻结 Decoder(mode='dn')：rates + 单细胞 DN 转向
        self.turn_source = turn_source
        self.name = name
        self.fwd_frac = cfg.readout_fwd_frac if fwd_frac is None else float(fwd_frac)
        self.v = 0.0
        self.omega = 0.0
        self.last_brain_turn = 0.0          # 单细胞 DNa02 转向（来自冻结前端）
        self.last_turn_cmd = 0.0            # 实际归一化转向
        self.last_pursuit_turn = 0.0
        self.front_blocks = 0
        self.front_blocked = False
        self.steps = 0
        self.sat_steps = 0                  # |turn|>=0.999 的步数（转向饱和）
        self.stop_steps = 0                 # 本步 v==0 的步数（被前方安全停车/降速到 0）
        self.turn_hist = []
        self.fired = None                   # 本步发放的神经元索引（供群体 turn_source 读）

    def update(self, fired, info=None, stalled: bool = False):
        self.dec.update(fired, info, stalled)     # 冻结前端：推进 rates/trace，更新 last_brain_turn
        info = info or {}
        self.fired = fired
        self.last_brain_turn = float(self.dec.last_brain_turn)
        turn = float(np.clip(self.turn_source(self), -1.0, 1.0))
        self.last_turn_cmd = turn
        self.last_pursuit_turn = 0.0

        v = self.car_cfg.max_speed * self.fwd_frac
        v *= 1.0 - self.cfg.turn_slowdown * abs(turn)      # 急转降速（统一）
        omega = self.car_cfg.max_omega * turn

        self.front_blocked = False
        if self.cfg.front_safety and v > 0:
            front = float(info.get("dmin_F", np.inf))
            if front <= self.cfg.front_stop_dist:
                v = 0.0
                self.front_blocked = True
                self.front_blocks += 1
            elif front < self.cfg.front_slow_dist:
                ratio = ((front - self.cfg.front_stop_dist)
                         / (self.cfg.front_slow_dist - self.cfg.front_stop_dist))
                v *= float(np.clip(ratio, 0.0, 1.0))

        a = self.cfg.smooth
        if self.front_blocked:
            self.v = 0.0
        else:
            self.v = (1 - a) * self.v + a * v
        self.omega = (1 - a) * self.omega + a * omega

        if abs(turn) >= 0.999:
            self.sat_steps += 1
        if v == 0.0:
            self.stop_steps += 1
        self.steps += 1
        self.turn_hist.append(turn)
        return self.v, self.omega

    # 诊断
    def saturation_ratio(self):
        return self.sat_steps / self.steps if self.steps else 0.0

    def stop_ratio(self):
        return self.stop_steps / self.steps if self.steps else 0.0


# ------------------------------------------------------------- turn sources ---
def turn_dn_single(ctl):
    """原始单细胞 DNa02 转向（冻结 Decoder 的 dn brain_turn）。"""
    return ctl.last_brain_turn


def turn_zero(ctl):
    """神经转向置零。"""
    return 0.0


def turn_swap(ctl):
    """左右交换消融：DNa02 转向取反。"""
    return -ctl.last_brain_turn


def make_turn_population(dn_idx, signs, ref, gain, tau, dt, window_s, deadzone=0.0):
    """（Task 3）最小群体解码器：对选定下行神经元做带符号、归一化的群体差分。

    turn = clip(gain * Σ_i sign_i * (trace_i - ref_i) / scale)
      trace_i : 该神经元脉冲的指数痕迹（tau）
      ref_i   : 无遮挡基线的稳态痕迹（受控刺激上测得，用于去基线）
      dn_idx / signs / ref : **人工选定的群体与符号**，不冒充果蝇天然运动解码
    返回一个 turn_source(ctl) 闭包，内部维护每个神经元的痕迹。
    """
    n = len(dn_idx)
    state = {"tr": np.zeros(n, np.float32)}

    def src(ctl):
        # 从 ctl 取本步发放的神经元索引（冻结前端已 step 过）
        fired = ctl.fired
        if tau <= 0 or dt <= 0:
            raise ValueError("tau/dt 必须为正")
        decay = float(np.exp(-dt / tau))
        tr = state["tr"] * decay
        if len(fired):
            hit = np.nonzero(np.isin(dn_idx, fired))[0]
            if len(hit):
                tr[hit] += 1.0
        state["tr"] = tr
        dev = (tr - np.asarray(ref, np.float32)) / max(1e-6, float(window_s) / dt)
        raw = float(np.sum(np.asarray(signs, np.float32) * dev))
        t = gain * raw
        if abs(t) < deadzone:
            t = 0.0
        return t

    return src
