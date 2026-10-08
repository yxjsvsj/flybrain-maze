"""P2b：真实 ToF 版 Encoder（架构 A —— 不改冻结 `loop/encode.py`）。

复用冻结 `Encoder` 的**神经元组**与**注入公式**（`_side`/`inject` 的 loom/chase/threat
公式、`per_step = 1-exp(-dt/tau)`），只把"输入"从虚拟射线换成 5 路真实 ToF：

- `angles`  由传入的传感器角度决定（v1.3: L+70/FL+30/F0/FR-30/R-70）
- `left/right/front` 掩码**由 angles + cfg.avoidance_fov_deg 计算**，不手写集合。
  baseline ±60° 下即 left={FL,F}, right={FR,F}, front={F}；L/R 参与建图、不参与神经避障。
- `sense(frame)`  -> 每条传感器射线的距离(米)：VALID=mm/1000；NO_TARGET=max_range；
                     TOO_NEAR=min_distance(触发 front safety)
- `update(frame, dists) -> bool`  新 tof_seq 时，**按真实 ToF 帧间隔**算 side 缓存
                     （size/growth/loom/chase/threat）；返回是否消费了新帧。
- `inject()`  -> (inject_list, info)，用缓存（帧间**零阶保持**），按 per_step(brain_dt) 缩放。

Memory 门控不在这里——由 `physical_run.py` 用 `update()` 的返回值决定是否 update 一次。
"""
from __future__ import annotations

import numpy as np

from config import EncoderConfig
from loop.encode import Encoder

MIN_DIST_M = 0.05


class PhysicalToFEncoder(Encoder):
    def __init__(self, groups: dict, cfg: EncoderConfig, brain_dt: float,
                 angles_deg: dict, sensor_order=("L", "FL", "F", "FR", "R"),
                 car_max_speed: float = 0.9, car_radius: float = 0.22,
                 tau: float = 0.100):
        super().__init__(groups, cfg, brain_dt, car_max_speed, car_radius, tau)
        self.names = list(sensor_order)
        self.angles = np.deg2rad(np.array([angles_deg[n] for n in self.names], dtype=float))
        # 掩码由实际 angles 和可配置 avoidance_fov 计算
        avoid_half = np.deg2rad(cfg.avoidance_fov_deg / 2)
        center = np.isclose(self.angles, 0.0, atol=1e-9)
        fwd = np.abs(self.angles) <= avoid_half
        self.left = fwd & ((self.angles > 0) | center)
        self.right = fwd & ((self.angles < 0) | center)
        front_half = np.deg2rad(cfg.front_cone_deg / 2)
        self.front = np.abs(self.angles) <= front_half

        self._cache: dict = {}
        self._info: dict = {}
        self.last_seq = None
        self._last_frame_t = None
        self.frames_consumed = 0

    # ---- 输入 ----
    def sense(self, frame) -> np.ndarray:  # type: ignore[override]
        """ToF 帧 -> 每条射线的距离(米)。"""
        d = np.full(len(self.names), self.cfg.max_range, dtype=np.float32)
        for i, n in enumerate(self.names):
            st = frame.status.get(n)
            mm = frame.ranges.get(n)
            if st == "TOO_NEAR":
                d[i] = MIN_DIST_M                       # 太近 -> min distance
            elif st == "VALID" and mm is not None:
                d[i] = float(mm) / 1000.0
            else:
                d[i] = self.cfg.max_range               # NO_TARGET / IO_ERROR
        return np.clip(d, MIN_DIST_M, self.cfg.max_range)

    def update(self, frame, dists: np.ndarray) -> bool:
        """新 tof_seq -> 用真实帧间隔刷新 side 缓存。返回是否消费了新帧。"""
        if frame.seq == self.last_seq:
            return False
        if self._last_frame_t is None:
            dt_tof = self.dt
        else:
            dt_tof = max(1e-3, float(frame.t) - self._last_frame_t)
        self._last_frame_t = float(frame.t)
        self.last_seq = frame.seq
        self.frames_consumed += 1

        old_dt = self.dt
        self.dt = dt_tof                                # growth 用真实 ToF 帧间隔
        try:
            self._cache["L"] = self._side(dists, self.left, "L")
            self._cache["R"] = self._side(dists, self.right, "R")
        finally:
            self.dt = old_dt

        _, _, _, _, dmin_l = self._cache["L"]
        _, _, _, _, dmin_r = self._cache["R"]
        dmin_f = max(float(dists[self.front].min()) - self.car_radius, MIN_DIST_M)
        self._info = {
            "bias": float((1.0 - dmin_r / self.cfg.max_range) - (1.0 - dmin_l / self.cfg.max_range)),
            "prox_l": float(1.0 - dmin_l / self.cfg.max_range),
            "prox_r": float(1.0 - dmin_r / self.cfg.max_range),
            "dmin_L": dmin_l, "dmin_R": dmin_r, "dmin_F": float(dmin_f),
        }
        return True

    def inject(self):  # type: ignore[override]
        """用缓存产出 (inject_list, info)（帧间零阶保持，按 per_step 缩放）。"""
        cl = self._cache.get("L", (0.0, 0.0, 0.0, 0.0, self.cfg.max_range))
        cr = self._cache.get("R", (0.0, 0.0, 0.0, 0.0, self.cfg.max_range))
        loom_l, chase_l, thr_l = cl[0], cl[1], cl[2]
        loom_r, chase_r, thr_r = cr[0], cr[1], cr[2]

        inject = []
        for name, dv in (("loom_L", loom_l), ("loom_R", loom_r),
                         ("chase_L", chase_l), ("chase_R", chase_r),
                         ("threat_L", thr_l), ("threat_R", thr_r)):
            if dv > 0:
                inject.append((self.groups[name], float(dv * self.per_step)))
        if self.cfg.forward_drive_dv > 0:
            inject.append((self.groups["fwd"],
                           float(self.cfg.forward_drive_dv * self.per_step)))

        info = dict(self._info)
        info.update({"loom_L": loom_l, "loom_R": loom_r,
                     "threat_L": thr_l, "threat_R": thr_r})
        return inject, info
