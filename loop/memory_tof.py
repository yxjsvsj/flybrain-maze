"""P2 建图适配层：支持 5 条**不同原点**的射线，保持冻结 `OccupancyMemory` 的
冲突保护 / 目标格保护 / map_revision / invariant 语义。**不改 `nav/memory.py`。**

冻结的 `OccupancyMemory.update(x, y, angles, dists)` 假定所有射线从同一点 (x, y) 发出。
真实 ToF 五路各有各的光学窗口，所以这里复用它的 DDA 原语
`_traverse(ox, oy, angle, dist, is_hit)`，再**逐条照抄** update() 里的打标逻辑：

  * UNKNOWN -> FREE；
  * 精确命中（kind="hit"）-> OCCUPIED；FREE 命中需累积 `free_conflict_threshold` 才翻转；
  * 终点格永不写墙（`self.goal` 保护）；
  * **车格**（访问位置，不是传感器光学窗口）标 FREE + visited；
  * `map_revision`、`goal_reached_map`、`check_invariants()` 语义不变。

TOO_NEAR 只贡献"自由空间到 min 距离"，**不产生精确墙面命中点**。
"""
from __future__ import annotations

import math

from nav.memory import FREE, OCCUPIED, UNKNOWN, OccupancyMemory


class ToFMemory(OccupancyMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_tof_seq = None

    def update_rays(self, car_x: float, car_y: float, rays, tof_seq=None) -> bool:
        """rays: 可迭代 (ox, oy, world_angle, dist_cells, kind)，kind ∈ {"hit","near"}。
        全部量在 **cells**。车访问位置用 `(car_x, car_y)`，**不是**传感器光学窗口。

        `tof_seq` 给出且与上次相同 -> 直接跳过（同一帧绝不重复消费）。返回是否更新。
        """
        if tof_seq is not None:
            if tof_seq == self.last_tof_seq:
                return False
            self.last_tof_seq = tof_seq

        car_cx, car_cy = int(car_x), int(car_y)
        changed = False
        for (ox, oy, ang, d, kind) in rays:
            d = float(d)
            ang = float(ang)
            is_hit = (kind == "hit" and d < self.max_range - 1e-6)
            free, hit = self._traverse(float(ox), float(oy), ang, d, is_hit)
            if is_hit and hit is None:
                # 冻结 _traverse 只在**端点落在格边界**时才判命中（P1 迷宫墙正好在边界）。
                # 真实 ToF 的墙在任意距离 -> 端点在格内部 -> 它返回 None。
                # 这里补：命中格 = "端点微步进越过表面"后的那一格。
                eps = 1e-3 * max(1.0, d)
                ex = float(ox) + (d + eps) * math.cos(ang)
                ey = float(oy) + (d + eps) * math.sin(ang)
                hit = (int(math.floor(ex)), int(math.floor(ey)))
                if free and free[-1] == hit:
                    free = free[:-1]          # 别把墙格当自由
            for gx, gy in free:
                if self.known[gy, gx] == UNKNOWN:
                    self.known[gy, gx] = FREE
                    changed = True
            # 只有精确命中才写墙；near/NO_TARGET 不写。
            if not is_hit or hit is None or hit == (int(ox), int(oy)):
                continue
            hx, hy = hit
            if not (0 <= hx < self.w and 0 <= hy < self.h):
                continue
            if self.goal is not None and (hx, hy) == self.goal:
                continue
            cur = self.known[hy, hx]
            if cur == UNKNOWN:
                self.known[hy, hx] = OCCUPIED
                changed = True
            elif cur == FREE:
                if self.conflicts[hy, hx] < 255:
                    self.conflicts[hy, hx] += 1
                if self.conflicts[hy, hx] >= self.free_conflict_threshold:
                    self.known[hy, hx] = OCCUPIED
                    self.conflicts[hy, hx] = 0
                    changed = True

        # 车自身格子（访问位置）—— NOT the sensor optical windows.
        if 0 <= car_cx < self.w and 0 <= car_cy < self.h:
            if self.known[car_cy, car_cx] != FREE:
                self.known[car_cy, car_cx] = FREE
                self.conflicts[car_cy, car_cx] = 0
                changed = True
            self.visited[car_cy, car_cx] = True
        if self.goal is not None and self.known[self.goal[1], self.goal[0]] == FREE:
            self.goal_reached_map = True
        if changed:
            self.map_revision += 1
        return True
