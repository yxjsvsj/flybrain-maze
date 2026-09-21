"""下行神经元 -> (v, omega)。

这是全项目最"手工"的一块：连接组只给接线，不告诉你 DN 发放率怎么变成轮速。
Eon Systems 的 embodied fly 也是手工选的映射（他们自己承认"somewhat arbitrarily
chosen by hand"），所以这里的所有增益都是可调的，别当成物理常数。

发放率用滑窗统计，先除以 ref_hz 归一化，增益就变成无量纲的"占满量程的比例"。
"""
from __future__ import annotations

import numpy as np

from config import DecoderConfig


class Decoder:
    def __init__(self, brain, groups: dict, cfg: DecoderConfig, car_cfg, dt: float,
                 dn_idx: np.ndarray | None = None):
        self.cfg = cfg
        self.car_cfg = car_cfg
        self.dt = dt
        self.names = list(groups)
        self.index = {name: i for i, name in enumerate(self.names)}

        self.slot = np.full(brain.n, -1, np.int32)
        for gi, name in enumerate(self.names):
            idx = groups[name]
            if len(idx):
                self.slot[np.asarray(idx)] = gi

        # readout 模式：把全部下行神经元的脉冲痕迹喂给训练好的线性读出头
        self.readout = None
        self.trace = None
        if cfg.mode == "readout":
            from flybrain import Readout, Trace
            if not cfg.readout_path:
                raise SystemExit("--mode readout 需要 --readout <path.npz>"
                                 "（先用 python -m loop.train_readout 训练）")
            if dn_idx is None or len(dn_idx) == 0:
                raise SystemExit("readout 模式需要下行神经元索引（dn_idx）")
            self.readout = Readout.load(cfg.readout_path)
            self.trace = Trace(brain, idx=np.asarray(dn_idx), tau=cfg.readout_tau)

        self.window = max(1, int(round(cfg.window_s / dt)))
        self.hist = np.zeros((self.window, len(self.names)), np.float32)
        self.sizes = np.array([max(1, len(groups[n])) for n in self.names], np.float32)
        self.i = 0
        self.filled = 0
        self.rates = np.zeros(len(self.names), np.float32)
        self.v, self.omega = 0.0, 0.0
        self.escapes = 0
        self.stalls = 0
        self.front_blocked = False    # 本步是否被前方安全层硬停
        self.front_blocks = 0         # 累计触发次数
        # 诊断用：把这一步的各个转向分量暴露出来，便于定位失败原因
        self.last_brain_turn = 0.0
        self.last_pursuit_turn = 0.0
        self.last_turn_cmd = 0.0
        self.steps = 0
        self.escape_hold = max(1, int(round(cfg.escape_hold_s / dt)))
        self.escape_until = -1
        self.stall_after = max(1, int(round(cfg.stall_after_s / dt)))
        self.stall_hold = max(1, int(round(cfg.stall_hold_s / dt)))
        self.stall_steps = 0
        self.stall_until = -1
        self.in_stall = False

    def _rate_hz(self, fired: np.ndarray) -> np.ndarray:
        """每个神经元的人均发放率（Hz）。除以池大小，否则大池天然占优。"""
        counts = np.zeros(len(self.names), np.float32)
        if len(fired):
            slots = self.slot[fired]
            slots = slots[slots >= 0]
            if len(slots):
                counts = np.bincount(slots, minlength=len(self.names)).astype(np.float32)
        self.hist[self.i] = counts
        self.i = (self.i + 1) % self.window
        self.filled = min(self.filled + 1, self.window)
        return self.hist.sum(0) / (self.filled * self.dt * self.sizes)

    def update(self, fired: np.ndarray, info=None, stalled: bool = False):
        """info 是 Encoder.inject() 返回的那个 dict（至少要 bias）；
        为了兼容也接受一个 float 当作 bias。"""
        cfg = self.cfg
        if info is None:
            info = {}
        elif not isinstance(info, dict):
            info = {"bias": float(info)}
        bias = float(info.get("bias", 0.0))

        rates = self._rate_hz(fired)
        self.rates = rates
        # 真脑静息时下行神经元不是 0（DNg100 ~0.25 Hz），先减掉基线再归一化，
        # 否则静止的噪声会被当成指令。
        r = np.maximum(0.0, rates - cfg.baseline_hz) / cfg.ref_hz
        g = self.index

        if cfg.mode == "scripted":
            # 手写基线：把某一侧的墙维持在固定距离上。完全不用脑，也不吃脑触发的逃逸。
            key = "dmin_R" if cfg.follow_side == "R" else "dmin_L"
            dist = float(info.get(key, 99.0))
            brain_turn = cfg.follow_gain * (cfg.follow_distance - dist)
            v = self.car_cfg.max_speed * cfg.follow_speed_frac
        elif cfg.mode == "readout":
            # 1314 个下行神经元的脉冲痕迹 -> 线性读出头 -> 归一化转向
            feats = self.trace.observe(fired)
            brain_turn = float(np.clip(self.readout.predict(feats), -1.0, 1.0))
            v = self.car_cfg.max_speed * cfg.readout_fwd_frac
        else:
            turn = cfg.turn_sign * (r[g["turn_L"]] - r[g["turn_R"]])
            if abs(turn) < cfg.deadzone:
                turn = 0.0
            brain_turn = cfg.turn_gain * turn
            drive = cfg.fwd_gain * r[g["fwd"]] - cfg.back_gain * r[g["back"]]
            v = self.car_cfg.max_speed * float(np.clip(drive, -1, 1))

        # 记忆只提供"往哪走"的高层偏置，低层转向仍由脑给出。两者权重可分别消融。
        self.last_brain_turn = float(brain_turn)
        self.last_pursuit_turn = float(info.get("pursuit_turn", 0.0))
        turn_cmd = (cfg.brain_gain * brain_turn
                    + cfg.memory_gain * self.last_pursuit_turn)
        turn_cmd = float(np.clip(turn_cmd, -1.0, 1.0))
        self.last_turn_cmd = turn_cmd
        omega = self.car_cfg.max_omega * turn_cmd
        # 急转降速：不减速的话会带着满舵冲进墙里
        v *= 1.0 - cfg.turn_slowdown * abs(turn_cmd)

        if (cfg.mode == "dn" and r[g["escape"]] > cfg.escape_thresh
                and self.steps >= self.escape_until):
            self.escapes += 1
            self.escape_until = self.steps + self.escape_hold
            omega = self.car_cfg.max_omega * cfg.escape_omega_frac * float(np.clip(bias, -1, 1))
            v = -self.car_cfg.max_speed * self.car_cfg.escape_back_frac

        # 卡死：想走却走不动。正对墙时左右 loom 对称、转向输出为零，光靠视觉出不来，
        # 果蝇靠本体感觉（腿的接触反馈）发现被卡住再挣脱，这里用 stall 标志代替。
        self.stall_steps = self.stall_steps + 1 if stalled else 0
        if self.stall_steps > self.stall_after and self.steps >= self.stall_until:
            if not self.in_stall:
                self.stalls += 1
            self.in_stall = True
            self.stall_until = self.steps + self.stall_hold
        if self.steps < self.stall_until:
            v = -self.car_cfg.max_speed * cfg.stall_back_frac
            if cfg.stall_turn_mode == "spin":
                d = bias if abs(bias) > 0.2 else cfg.stall_dir
                omega = self.car_cfg.max_omega * cfg.stall_omega_frac * float(np.clip(d, -1, 1))
            elif cfg.stall_turn_mode == "straight":
                omega = 0.0
            # "hold" 什么都不做：转向保持规划器/脑给的值
        else:
            self.in_stall = False

        # 前方安全限速层：正前方快撞墙时降速/停车。
        # **只限速，不决定方向**——方向是果蝇脑/规划器的活，手写规则不能顶替它。
        self.front_blocked = False
        if cfg.front_safety and v > 0:
            front = float(info.get("dmin_F", np.inf))
            if front <= cfg.front_stop_dist:
                v = 0.0
                self.front_blocked = True
                self.front_blocks += 1
            elif front < cfg.front_slow_dist:
                ratio = ((front - cfg.front_stop_dist)
                         / (cfg.front_slow_dist - cfg.front_stop_dist))
                v *= float(np.clip(ratio, 0.0, 1.0))

        a = cfg.smooth
        if self.front_blocked:
            # 硬停车不能被输出低通留住旧的正向速度
            self.v = 0.0
        else:
            self.v = (1 - a) * self.v + a * v
        self.omega = (1 - a) * self.omega + a * omega
        self.steps += 1
        return self.v, self.omega
