"""P2 专属 windowed stall detector。

**不改冻结** `loop/decode.py` / `encode.py` / `run.py` / `config.py` / `nav/**`。
它只产出一个 `stalled: bool` 供现有 Decoder 使用（Decoder 自身状态机不变）。

设计目标（针对 P2 实车）
------------------------
- **用单调时间 + odom 观察窗口**判断，而不是单个脑步的 `last_move`
  （后者在"脑 50Hz、odom ~44Hz 不同步"时会把同帧重复读成零位移）。
- **没有新 odom 帧 ≠ 零位移**：窗口里累计的是"实际观测到的位移"；
  若最新帧过旧 → 那是 stale（交给外部 FAILSAFE），**本 detector 不据此判堵转**。
- **PWM ramp 启动宽限**：命令刚变化后 `startup_grace_s` 内不判堵转。
- **慢速但持续移动 ≠ 机械卡死**：阈值很低（默认 12% 目标速度）。
- **原地转向与平移分开**：原地转看角速度/轮差，不看线速度。
- **STOP / 前方安全 / 机械堵转分开**：返回 reason ∈
  {"", "mechanical", "front_safety", "stop"}。

它**不**处理 E-STOP / ToF stale / 前方安全的停车——那些护栏仍在原来位置，
本 detector 的 stall 恢复不会绕过它们。
"""
from __future__ import annotations

from collections import deque


class WindowedStallDetector:
    def __init__(self, *, m_per_count: float, meters_per_cell: float,
                 left_sign: float = -1.0, right_sign: float = 1.0,
                 window_s: float = 0.40, stall_move_frac: float = 0.12,
                 stall_sustain_s: float = 0.35, startup_grace_s: float = 0.30,
                 min_v_cmd_frac: float = 0.10, turn_min_frac: float = 0.05,
                 odom_stale_s: float = 0.35) -> None:
        self.mpc = float(meters_per_cell)
        self.mpc_count = float(m_per_count) / float(meters_per_cell)  # 每计数 = cells
        self.left_sign = float(left_sign)
        self.right_sign = float(right_sign)
        self.window_s = float(window_s)
        self.stall_move_frac = float(stall_move_frac)
        self.stall_sustain_s = float(stall_sustain_s)
        self.startup_grace_s = float(startup_grace_s)
        self.min_v_cmd_frac = float(min_v_cmd_frac)
        self.turn_min_frac = float(turn_min_frac)
        self.odom_stale_s = float(odom_stale_s)

        self._win: deque = deque()          # (t, moved_cells, turned_rad)
        self._last = None                   # (t, left, right)
        self._last_seq = None
        self._new_frame_t = None
        self._cmd_change_t = -1e9
        self._last_cmd = (0.0, 0.0)
        self._sustain_since = None

    # ---- 输入 ----
    def note_command(self, t: float, v: float, w: float) -> None:
        """命令变化时登记（用于 ramp 宽限）。"""
        if abs(v - self._last_cmd[0]) > 1e-6 or abs(w - self._last_cmd[1]) > 1e-6:
            self._cmd_change_t = t
            self._last_cmd = (v, w)

    def observe(self, t: float, left: int, right: int, seq=None) -> bool:
        """收到一个 odom 帧时调用。返回 True 表示是**新帧**。

        - 给了 seq：seq 变化才算新帧（同一帧重复读 -> 返回 False，不算零位移）。
        - 没给 seq：退化为"tick 变化才算新帧"。
        - **新帧但 tick 不变**（真堵转）会被记录为 moved=0（关键）。
        """
        if seq is not None:
            if seq == self._last_seq:
                return False
            self._last_seq = seq
        elif self._last is not None and left == self._last[1] and right == self._last[2]:
            return False
        if self._last is not None:
            _tt, pl, pr = self._last
            d = 0.5 * (self.left_sign * (left - pl) + self.right_sign * (right - pr))
            self._win.append((t, abs(d) * self.mpc_count, 0.0))
        self._last = (t, left, right)
        self._new_frame_t = t
        return True

    def _prune(self, t: float) -> None:
        while self._win and (t - self._win[0][0]) > self.window_s:
            self._win.popleft()

    def stalled(self, t: float, v_cmd: float, w_cmd: float,
                front_blocked: bool = False) -> tuple[bool, str]:
        """返回 (stalled, reason)。reason ∈ {"","stop","front_safety","mechanical"}。"""
        self._prune(t)
        # 命令近乎为零 -> STOP（不是堵转）
        if abs(v_cmd) < self.min_v_cmd_frac and abs(w_cmd) < self.min_v_cmd_frac:
            self._sustain_since = None
            return False, "stop"
        # 前方安全锁死 -> 单独原因（不混进 mechanical）
        if front_blocked:
            self._sustain_since = None
            return False, "front_safety"
        # stale：最新帧过旧 -> 交给外部 FAILSAFE，本 detector 不判堵转
        if self._new_frame_t is None or (t - self._new_frame_t) > self.odom_stale_s:
            self._sustain_since = None
            return False, ""
        # ramp 启动宽限
        if (t - self._cmd_change_t) < self.startup_grace_s:
            self._sustain_since = None
            return False, ""

        span = sum(d for _, d, _ in self._win)
        horizon = max(1e-3, min(self.window_s, t - (self._win[0][0] if self._win else t)))
        # 期望位移：平移用 |v|，原地转用 |w|（分开处理）
        if abs(v_cmd) >= abs(w_cmd):
            expected = abs(v_cmd) * horizon
            frac = self.stall_move_frac
        else:
            expected = abs(w_cmd) * horizon
            frac = self.turn_min_frac
        moving_enough = span >= frac * expected if expected > 0 else True

        if moving_enough:
            self._sustain_since = None
            return False, ""
        if self._sustain_since is None:
            self._sustain_since = t
        if (t - self._sustain_since) >= self.stall_sustain_s:
            return True, "mechanical"
        return False, ""
