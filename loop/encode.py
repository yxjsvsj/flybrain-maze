"""传感器 -> 感觉神经元注入。

为什么不用 flybrain.eyes.FeatureDetectors：它是给"屏幕上的一个对手"设计的
（opp=(dx, size)），多路测距塞不进去。但注入的通道和细胞类型跟它保持一致
（LPLC2 = looming，LC4 = 快速 looming / 逃逸），方便对照和替换。

强度单位是"稳态电压增量"（config.EncoderConfig 里的 gain 都是这个单位）：
LIF 的 v 会衰减，每步注入 amount 的稳态贡献是 amount/(1-decay)，所以
    amount = dv * (1 - decay),  decay = exp(-dt/tau)
这样 gain 就和 dt 无关，换步长不用重调。

如果真脑不转向：README 里 LC10a -> DNa02 是已知的转向通路（LPLC2/LC4 主要通
DNp01 逃逸），届时把 LC10a 也加成一个注入通道即可。
"""
from __future__ import annotations

import numpy as np

from config import EncoderConfig


class Encoder:
    def __init__(self, groups: dict, cfg: EncoderConfig, brain_dt: float,
                 car_max_speed: float = 0.9, car_radius: float = 0.22,
                 tau: float = 0.100):
        self.cfg = cfg
        self.groups = groups
        self.dt = brain_dt
        self.max_speed = car_max_speed
        self.car_radius = car_radius
        self.angles = np.deg2rad(np.linspace(-cfg.fov_deg / 2, cfg.fov_deg / 2, cfg.n_rays))
        # 相对角逆时针为正。theta=0 时 forward=+x，左侧 = +y = 正角度。
        self.left = self.angles > 0
        self.right = self.angles < 0
        self.per_step = 1.0 - np.exp(-brain_dt / tau)
        self.prev_size: dict[str, float] = {"L": 0.0, "R": 0.0}
        self.seen: dict[str, bool] = {"L": False, "R": False}

    def sense(self, maze, car) -> np.ndarray:
        """返回每条射线的**原始**距离（未扣车体半径）。

        建图要用原始距离：扣了半径后，短的背向射线"命中点"会落在车自己的格子里，
        把自身标成墙，整张图就废了。
        """
        x, y, theta = car.pose
        d = maze.raycast_many(x, y, theta + self.angles, self.cfg.max_range)
        return np.clip(d, 0.05, self.cfg.max_range)

    def _side(self, dists: np.ndarray, mask: np.ndarray, side: str):
        # 注意：不要用 soft-min。它给出的下界可以变成负数，贴墙时 size 变负、
        # loom 被裁到 0——最危险的时候反而没有信号。
        d_min = max(float(dists[mask].min()) - self.car_radius, 0.05)
        size = min(self.cfg.wall_size / d_min, 1.5)

        if not self.seen[side]:
            self.seen[side] = True
            self.prev_size[side] = size          # 首帧不产生 growth，避免启动瞬态
            dsize = 0.0
        else:
            # 射线切换会让"最近距离"跳到另一条射线上，产生物理上不可能的变化。
            # 用最大接近速度给变化量设上限，超出的部分直接砍掉。
            dsize_max = self.cfg.wall_size * self.max_speed / d_min ** 2 * self.dt
            dsize = float(np.clip(size - self.prev_size[side], 0.0, dsize_max))
            self.prev_size[side] = size
        growth = dsize / self.dt                    # 每秒

        loom = self.cfg.loom_size_gain * size + self.cfg.loom_growth_gain * growth
        loom = float(np.clip(loom, 0.0, self.cfg.cap))

        chase = self.cfg.chase_size_gain * size + self.cfg.chase_growth_gain * growth
        chase = float(np.clip(chase, 0.0, self.cfg.cap))

        threat = 0.0
        if growth > self.cfg.threat_thresh:
            threat = float(np.clip(self.cfg.threat_gain * growth, 0.0, self.cfg.cap))
        return loom, chase, threat, size, d_min

    def inject(self, dists: np.ndarray):
        """返回 (inject_list, info)。info 里有 bias（用于逃逸方向）和调试量。"""
        loom_l, chase_l, thr_l, _, dmin_l = self._side(dists, self.left, "L")
        loom_r, chase_r, thr_r, _, dmin_r = self._side(dists, self.right, "R")

        inject = []
        for name, dv in (("loom_L", loom_l), ("loom_R", loom_r),
                         ("chase_L", chase_l), ("chase_R", chase_r),
                         ("threat_L", thr_l), ("threat_R", thr_r)):
            if dv > 0:
                inject.append((self.groups[name], float(dv * self.per_step)))

        # 常量前进驱动（见 EncoderConfig.forward_drive_dv 的说明）
        if self.cfg.forward_drive_dv > 0:
            inject.append((self.groups["fwd"], float(self.cfg.forward_drive_dv * self.per_step)))

        prox_l = 1.0 - dmin_l / self.cfg.max_range
        prox_r = 1.0 - dmin_r / self.cfg.max_range
        info = {
            "bias": float(prox_r - prox_l),      # >0 表示右侧更危险 -> 往左躲
            "prox_l": float(prox_l), "prox_r": float(prox_r),
            "loom_L": loom_l, "loom_R": loom_r,
            "threat_L": thr_l, "threat_R": thr_r,
            "dmin_L": dmin_l, "dmin_R": dmin_r,
        }
        return inject, info
