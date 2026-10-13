"""P2d-15c：固定 MaleCNS L50/R50 群体的在线转向（P2 模块）。

候选群体由 P2d-15b 在 screen seeds 64/101/202/303 上、按**同一条预声明规则**选出，随后
**固化**在此模块中（`L_IDX`/`R_IDX` + 校验和）；后续测试**不再重新筛选、不改 K**。

在线量：D(t) = mean(trace[L_IDX]) − mean(trace[R_IDX])，Trace τ=0.2s，**只用当前与过去**的发放。
固定人工电机映射（在 screen seeds 上一次性确定后冻结）：
    FL(左前)障碍 -> D 上升 -> 避障角速度**负**方向；FR(右前) -> D 下降 -> **正**方向
    即 turn_cmd = sign * gain * (D(t) - b0)   (sign = -1)
  b0 = 每次运行前在**无遮挡 A** 状态下估计的基线，估计后固定。

⚠ 披露：这些神经元是**实验筛选出的候选群体**；电机映射是**人工设计**，不冒充天然果蝇运动连接。
P2d-15b 的时间匹配 AAA 差分**仅供离线评价**，绝不是闭环控制器的输入。
"""
from __future__ import annotations

import hashlib

import numpy as np

from loop.dn_dynamics_probe import PLANS, per_dn_metrics, run_plan

# 由 screen_indices() 生成后固化（P2d-15c 第一步）。见 __main__ 复核。
L_IDX: tuple[int, ...] = (
    215, 932, 177, 788, 1262, 1013, 1163, 971, 412, 1243, 1029, 1050, 620, 171, 1175, 520, 963,
    31, 1150, 460, 625, 476, 859, 1057, 55, 1210, 1107, 1302, 521, 438, 134, 72, 146, 304, 484,
    1226, 266, 1140, 511, 428, 925, 410, 1239, 3, 271, 1256, 1030, 1162, 1039, 523)
R_IDX: tuple[int, ...] = (
    1227, 287, 931, 513, 94, 1079, 983, 904, 968, 267, 262, 143, 797, 311, 294, 526, 1055, 1001,
    416, 299, 434, 1126, 649, 404, 850, 1008, 958, 245, 907, 1000, 409, 234, 886, 11, 1275, 902,
    681, 948, 1003, 418, 1015, 860, 1016, 1090, 770, 1308, 692, 315, 275, 1264)
INDEX_CHECKSUM = "9285ac175082"
GAIN = 8.0        # 幅度比例：一次性由 screen |D_stim| 尺度(~0.12) 定为 ~1/0.12（披露，非搜索）
SIGN = -1.0       # D>0(左障碍) -> 负 omega（避障向开口）
TAU = 0.2
K_GROUP = 50


def checksum(L, R) -> str:
    arr = np.array(list(L) + list(R), np.int64)
    return hashlib.sha1(np.sort(arr).tobytes()).hexdigest()[:12]


def screen_indices(cfg, brain, groups, dn_idx, screen_seeds, *, k=K_GROUP):
    """按预声明规则在 screen seeds 上选 L/R 群体（仅用于**生成**固化索引）。"""
    from flybrain import Trace
    dt = brain.dt
    ml = {sd: per_dn_metrics(run_plan(cfg, brain, groups, dn_idx, Trace, sd, PLANS["left"]),
                             PLANS["left"], dt) for sd in screen_seeds}
    mr = {sd: per_dn_metrics(run_plan(cfg, brain, groups, dn_idx, Trace, sd, PLANS["right"]),
                             PLANS["right"], dt) for sd in screen_seeds}
    idx = np.mean([ml[sd]["dresp"] - mr[sd]["dresp"] for sd in screen_seeds], 0)
    order = np.argsort(-np.abs(idx))
    L = [int(i) for i in order if idx[i] > 0][:k]
    R = [int(i) for i in order if idx[i] < 0][:k]
    return L, R


class PopulationTurn:
    """在线群体转向：D(t)=mean(L)-mean(R)，turn=sign*gain*(D-b0)。

    L_IDX/R_IDX 是 `brain.cells(['descending_neuron'])` 的**位置**；构造时用 dn 解析成脑内真实索引。
    """

    def __init__(self, brain, dt, *, b0=0.0, gain=GAIN, sign=SIGN, tau=TAU, dn=None, idx=None):
        from flybrain import Trace
        pos = np.array(list(L_IDX) + list(R_IDX), int)
        if idx is None:
            idx = np.asarray(dn)[pos] if dn is not None else pos
        self.idx = np.asarray(idx, int)
        self.nL = len(L_IDX)
        self.trace = Trace(brain, idx=self.idx, tau=tau)
        self.dt = dt
        self.b0 = float(b0)
        self.gain = float(gain)
        self.sign = float(sign)
        self.last_D = 0.0

    def D(self, fired) -> float:
        a = np.asarray(self.trace.observe(fired), np.float32)
        d = float(a[:self.nL].mean() - a[self.nL:].mean())
        self.last_D = d
        return d

    def __call__(self, fired) -> float:
        d = self.D(fired)
        return float(np.clip(self.sign * self.gain * (d - self.b0), -1.0, 1.0))


def make_pop_source(brain, dt, b0, *, mode="POP", gain=GAIN, sign=SIGN, tau=TAU, dn=None):
    """返回 (turn_source(ctl)->turn, obj)。mode: POP / POP_zero / POP_swap。"""
    obj = PopulationTurn(brain, dt, b0=b0, gain=gain, sign=sign, tau=tau, dn=dn)
    if mode == "POP":
        return (lambda c: obj(c.fired)), obj
    if mode == "POP_zero":
        return (lambda c: 0.0), None
    if mode == "POP_swap":
        return (lambda c: -obj(c.fired)), obj
    raise KeyError(mode)


def estimate_b0(cfg, brain, groups, dn_idx, seed, *, seconds=1.2, tof_hz=10.0, mpc=0.40, tau=TAU):
    """运行前在无遮挡 A 状态估计 D 基线 b0（之后固定）。"""
    from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
    from loop.encode_tof import PhysicalToFEncoder
    from loop.dn_dynamics_probe import make_frame, STIMS
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    pt = PopulationTurn(brain, brain.dt, tau=tau, dn=dn_idx)
    brain.reset(seed)
    dt = brain.dt
    n = int(round(seconds / dt))
    period = 1.0 / tof_hz
    frame, seq, last_t = None, 0, -1e9
    vals = []
    for k in range(n):
        t = k * dt
        if frame is None or (t - last_t) >= period - 1e-9:
            seq += 1
            frame = make_frame(STIMS["A"], seq, t)
            last_t = t
        d = enc.sense(frame)
        enc.update(frame, d)
        inj, _info = enc.inject()
        fired = brain.step(inject=inj)
        vals.append(pt.D(fired))
    return float(np.mean(vals[int(len(vals) / 2):]))


def verify_indices(brain) -> bool:
    dn = np.asarray(brain.cells(["descending_neuron"]))
    ok = (len(L_IDX) == K_GROUP and len(R_IDX) == K_GROUP
          and len(set(L_IDX) & set(R_IDX)) == 0
          and max(list(L_IDX) + list(R_IDX)) < len(dn)
          and checksum(L_IDX, R_IDX) == INDEX_CHECKSUM)
    return bool(ok)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from loop.run import make_setup
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device="cuda", seed=64,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn = np.asarray(brain.cells(["descending_neuron"]))
    L, R = screen_indices(cfg, brain, groups, dn, [64, 101, 202, 303])
    print("L_IDX =", tuple(L))
    print("R_IDX =", tuple(R))
    print("CHECKSUM =", checksum(L, R))
