"""四轮差速车（非完整约束）：命令是 (v, omega)，不是左右轮速。

碰撞处理：每个物理子步先试"平移+旋转"，不行就只试旋转（贴墙时还能蹭着转出来），
都不行就停下并记一次碰撞。
"""
from __future__ import annotations

import math

import numpy as np

from config import CarConfig


class DiffDriveCar:
    def __init__(self, x: float, y: float, theta: float, cfg: CarConfig):
        self.cfg = cfg
        self.x, self.y, self.theta = float(x), float(y), float(theta)
        self.v, self.omega = 0.0, 0.0
        self.distance = 0.0
        self.collisions = 0          # 有接触的脑步数
        self.collision_events = 0    # 从"无接触"变成"有接触"的次数（真正的撞墙）
        self.last_move = 0.0         # 上一步实际平移距离（用于卡死检测）
        self._was_hit = False
        self.visited = set()
        self._mark_visited()
        self._probe = np.linspace(0, 2 * math.pi, 12, endpoint=False)

    def _mark_visited(self):
        self.visited.add((int(math.floor(self.x)), int(math.floor(self.y))))

    def _free(self, maze, x: float, y: float) -> bool:
        r = self.cfg.radius
        if maze.is_wall(x, y):
            return False
        px = x + r * np.cos(self._probe)
        py = y + r * np.sin(self._probe)
        for a, b in zip(px, py):
            if maze.is_wall(float(a), float(b)):
                return False
        return True

    def set_command(self, v: float, omega: float) -> None:
        self.v = float(np.clip(v, -self.cfg.max_speed, self.cfg.max_speed))
        self.omega = float(np.clip(omega, -self.cfg.max_omega, self.cfg.max_omega))

    def step(self, maze, dt: float) -> bool:
        """推进 dt 秒，返回这一步是否发生碰撞。

        圆盘车的碰撞与朝向无关，所以"旋转"永远合法——平动被挡住时只保留旋转，
        车就会贴着墙转出来，不会卡死。
        """
        n = max(1, int(round(dt / self.cfg.physics_dt)))
        h = dt / n
        hit = False
        moved = 0.0
        for _ in range(n):
            nt = self.theta + self.omega * h
            nx = self.x + self.v * math.cos(self.theta) * h
            ny = self.y + self.v * math.sin(self.theta) * h
            if self._free(maze, nx, ny):
                moved += math.hypot(nx - self.x, ny - self.y)
                self.x, self.y, self.theta = nx, ny, nt
                self.distance += abs(self.v) * h
            else:
                self.theta = nt
                self.v = 0.0
                hit = True
        self.last_move = moved
        self._mark_visited()
        if hit:
            self.collisions += 1
            if not self._was_hit:
                self.collision_events += 1
        self._was_hit = hit
        return hit

    def coverage(self, maze) -> float:
        total = len(maze.free_cells())
        return len(self.visited) / total if total else 0.0

    @property
    def pose(self):
        return self.x, self.y, self.theta
