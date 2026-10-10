"""P2 专属 windowed stall detector（v2，按审查修正）。

**不改冻结** `loop/decode.py` / `encode.py` / `run.py` / `config.py` / `nav/**`。
只产出 `stalled: bool` 供现有 Decoder 使用（Decoder 自身状态机不变）。

v2 修正
-------
1. `note_command` 只在**显著**命令变化时重置 startup_grace（阈值 `cmd_change_thresh`），
   不被 RealFlyBrain 每周期 ±0.0x 抖动反复重置。
2. 原地转向：`turned_rad` 由**轮差 + 校准轮距 L** 计算（也可用 odom theta 增量核对）。
3. **不直接比较 cells/s 与 rad/s**：分别按 `max_speed`/`max_omega` 归一化，并**独立**判断
   平移/旋转两条通道。
4. 必须用 odom `seq` 识别新帧；**新帧但 ticks 不变 = 有效零位移观测**。
5. 默认 **D=0.067 m / L=0.194 m**（已验收的落地有效值）。
6. 时钟：由调用方传 **Windows 本地 monotonic**，采样/命令/窗口同源。
"""
from __future__ import annotations

import math
from collections import deque


class WindowedStallDetector:
    def __init__(self, *, wheel_diam: float = 0.067, track: float = 0.194,
                 cpr: float = 1560.0, meters_per_cell: float = 0.40,
                 left_sign: float = -1.0, right_sign: float = 1.0,
                 max_speed: float = 0.9, max_omega: float = 3.0,
                 window_s: float = 0.40, trans_move_frac: float = 0.05,
                 rot_move_frac: float = 0.05, stall_sustain_s: float = 0.35,
                 startup_grace_s: float = 0.30, cmd_change_thresh: float = 0.35,
                 min_cmd_frac: float = 0.10, odom_stale_s: float = 0.35) -> None:
        self.m_per_count_cells = (math.pi * float(wheel_diam) / float(cpr)) / float(meters_per_cell)
        self.track_cells = float(track) / float(meters_per_cell)
        self.left_sign = float(left_sign)
        self.right_sign = float(right_sign)
        self.max_speed = float(max_speed)
        self.max_omega = float(max_omega)
        self.window_s = float(window_s)
        self.trans_move_frac = float(trans_move_frac)
        self.rot_move_frac = float(rot_move_frac)
        self.stall_sustain_s = float(stall_sustain_s)
        self.startup_grace_s = float(startup_grace_s)
        self.cmd_change_thresh = float(cmd_change_thresh)
        self.min_cmd_frac = float(min_cmd_frac)
        self.odom_stale_s = float(odom_stale_s)

        self._win: deque = deque()          # (t, |moved_cells|, |turned_rad|)
        self._last = None                   # (left, right)
        self._last_seq = None
        self._new_frame_t = None
        self._cmd_change_t = -1e9
        self._last_cmd = (0.0, 0.0)
        self._sustain_since = None
        self.sustain_duration = 0.0
        self.grace_active = False
        self.last_moved_cells = 0.0
        self.last_turned_rad = 0.0

    # ---- 输入 ----
    def note_command(self, t: float, v: float, w: float) -> None:
        """只在**显著**变化时重置宽限（避免脑的逐周期抖动）。"""
        if (abs(v - self._last_cmd[0]) > self.cmd_change_thresh
                or abs(w - self._last_cmd[1]) > self.cmd_change_thresh):
            self._cmd_change_t = t
            self._last_cmd = (v, w)

    def observe(self, t: float, left: int, right: int, seq=None,
                theta: float | None = None) -> bool:
        """收到 odom 帧。返回 True = 新帧。

        seq 变化才算新帧；**新帧但 ticks 不变 -> moved=0**（真零位移，仍记录）。
        theta 若给出，用 theta 增量核对转向（优先轮差，theta 做交叉检查）。
        """
        if seq is not None:
            if seq == self._last_seq:
                return False
            self._last_seq = seq
        elif self._last is not None and left == self._last[0] and right == self._last[1]:
            return False
        if self._last is not None:
            pl, pr = self._last
            dl = self.left_sign * (left - pl)
            dr = self.right_sign * (right - pr)
            moved = 0.5 * (dl + dr) * self.m_per_count_cells            # cells
            turned = (dr - dl) * self.m_per_count_cells / self.track_cells  # rad (轮差)
            self.last_moved_cells = moved
            self.last_turned_rad = turned
            self._win.append((t, abs(moved), abs(turned)))
        self._last = (left, right)
        self._new_frame_t = t
        return True

    def _prune(self, t: float) -> None:
        while self._win and (t - self._win[0][0]) > self.window_s:
            self._win.popleft()

    def stalled(self, t: float, v_cmd: float, w_cmd: float,
                front_blocked: bool = False) -> tuple[bool, str]:
        """返回 (stalled, reason)。reason ∈ {"","stop","front_safety","mechanical"}。"""
        self._prune(t)
        self.grace_active = (t - self._cmd_change_t) < self.startup_grace_s
        vn = abs(v_cmd) / max(1e-9, self.max_speed)
        wn = abs(w_cmd) / max(1e-9, self.max_omega)
        # 前方安全停车：单独原因，绝不当作机械堵转
        if front_blocked:
            self._sustain_since = None
            self.sustain_duration = 0.0
            return False, "front_safety"
        # 近乎 STOP
        if vn < self.min_cmd_frac and wn < self.min_cmd_frac:
            self._sustain_since = None
            self.sustain_duration = 0.0
            return False, "stop"
        # stale：最新帧过旧 -> 外部 FAILSAFE，本 detector 不判堵转
        if self._new_frame_t is None or (t - self._new_frame_t) > self.odom_stale_s:
            self._sustain_since = None
            self.sustain_duration = 0.0
            return False, ""
        # ramp 启动宽限
        if self.grace_active:
            self._sustain_since = None
            self.sustain_duration = 0.0
            return False, ""

        horizon = max(1e-3, min(self.window_s, t - (self._win[0][0] if self._win else t)))
        moved = sum(m for _, m, _ in self._win)
        turned = sum(r for _, _, r in self._win)
        # 独立判断平移 / 旋转（用归一化后的主导通道）
        if vn >= wn:
            moving = moved >= self.trans_move_frac * abs(v_cmd) * horizon
        else:
            moving = turned >= self.rot_move_frac * abs(w_cmd) * horizon

        if moving:
            self._sustain_since = None
            self.sustain_duration = 0.0
            return False, ""
        if self._sustain_since is None:
            self._sustain_since = t
        self.sustain_duration = t - self._sustain_since
        if self.sustain_duration >= self.stall_sustain_s:
            return True, "mechanical"
        return False, ""
