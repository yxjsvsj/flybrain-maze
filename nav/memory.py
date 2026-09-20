"""记忆模块：从射线建占用栅格，追踪已探索区域，给出"下一步往哪走"的瞄准点。

设计约束：**只能用传感器数据 + 位姿**，不偷看真值迷宫。这样实物上换成 SLAM
（位姿有漂移）时，这一层不用重写。终点坐标是已知输入（不需要靠探索发现它）。

规划策略：**直接朝已知坐标的终点规划**（Dijkstra，未知区域可通行但代价更高）。
撞到墙以后地图更新、自动重规划。这条比纯 frontier 探索高效得多——实测真脑
从 2/4 提到 4/4，而且覆盖率从 ~80% 降到 34-64%（不用逛完整张图）。
终点被彻底围死时才退回 frontier 探索兜底。

职责划分：
    这一层（高层）  -> 输出一个瞄准点（往哪走）
    果蝇脑（低层）  -> 输出转向（怎么躲开墙走过去）
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Callable

import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class OccupancyMemory:
    """占用栅格 + 已访问图 + frontier 探索。

    known[y, x]: 0 未知 / 1 自由 / 2 占据
    visited[y, x]: 车真正到过的格子
    """

    def __init__(self, width: int, height: int, max_range: float,
                 waypoint_gap: int = 3, ray_step: float = 0.4):
        self.w, self.h = int(width), int(height)
        self.max_range = float(max_range)
        self.lookahead = 0.9
        self.known = np.zeros((self.h, self.w), np.uint8)
        self.visited = np.zeros((self.h, self.w), bool)
        self.goal: tuple[int, int] | None = None
        self.last_target: tuple[float, float] | None = None
        self.path: list[tuple[int, int]] | None = None
        self.wp_idx = 0
        self.path_len = 0
        self.exhausted = False
        self.goal_planned = False        # 规划器当前能解出到终点的路径
        self.goal_reached_map = False    # 终点格子已经在地图上被确认为自由

    # ---- 建图 ----------------------------------------------------------------
    def update(self, maze, x: float, y: float, angles: np.ndarray) -> None:
        """angles 是**世界坐标**下的射线角。用 Maze 的 DDA 精确建图。"""
        ox, oy = int(x), int(y)
        for ang in angles:
            free, hit = maze.raycast_cells(x, y, float(ang), self.max_range)
            for gx, gy in free:
                if self.known[gy, gx] == UNKNOWN:
                    self.known[gy, gx] = FREE
            if hit is not None and hit != (ox, oy):
                hx, hy = hit
                if 0 <= hx < self.w and 0 <= hy < self.h:
                    self.known[hy, hx] = OCCUPIED
        if 0 <= ox < self.w and 0 <= oy < self.h:
            self.known[oy, ox] = FREE
            self.visited[oy, ox] = True
        if self.goal is not None and self.known[self.goal[1], self.goal[0]] == FREE:
            self.goal_reached_map = True

    def set_goal(self, x: float, y: float) -> None:
        self.goal = (int(x), int(y))

    # ---- 探索策略 -------------------------------------------------------------
    def _is_frontier(self, gx: int, gy: int) -> bool:
        """已知自由、没去过、且旁边还有未知区域 -> 值得去。"""
        if self.known[gy, gx] != FREE or self.visited[gy, gx]:
            return False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = gx + dx, gy + dy
                if not (0 <= nx < self.w and 0 <= ny < self.h):
                    return True          # 地图边界外也算未知
                if self.known[ny, nx] == UNKNOWN:
                    return True
        return False

    def _bfs(self, sx: int, sy: int, is_target: Callable[[int, int], bool]):
        """BFS 找最近的满足 is_target 的格子，返回从起点到它的路径（含起点）。"""
        if not (0 <= sx < self.w and 0 <= sy < self.h):
            return None
        prev: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
        q = deque([(sx, sy)])
        while q:
            gx, gy = q.popleft()
            if is_target(gx, gy):
                path = []
                cur: tuple[int, int] | None = (gx, gy)
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = gx + dx, gy + dy
                if (0 <= nx < self.w and 0 <= ny < self.h
                        and (nx, ny) not in prev
                        and self.known[ny, nx] == FREE):
                    prev[(nx, ny)] = (gx, gy)
                    q.append((nx, ny))
        return None

    def _plan_to_goal(self, sx: int, sy: int, unknown_cost: float = 4.0):
        """Dijkstra：已知自由代价 1，未知代价 unknown_cost，占据不可通行。

        这就是"知道终点坐标、直接朝它找过去"的做法：未知区域**允许**通行（代价更高），
        所以车会朝着终点方向扎进没探过的地方；撞到墙以后地图更新、自动重规划。
        比纯 frontier 探索高效得多——不用把整张图逛完才看见终点。
        """
        if self.goal is None:
            return None
        gx0, gy0 = self.goal
        if not (0 <= gx0 < self.w and 0 <= gy0 < self.h):
            return None
        if not (0 <= sx < self.w and 0 <= sy < self.h):
            return None

        dist = np.full((self.h, self.w), np.inf)
        prev: dict[tuple[int, int], tuple[int, int]] = {}
        dist[sy, sx] = 0.0
        pq = [(0.0, sx, sy)]
        while pq:
            d, x, y = heapq.heappop(pq)
            if d > dist[y, x]:
                continue
            if (x, y) == (gx0, gy0):
                path = []
                cur: tuple[int, int] | None = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = prev.get(cur)
                return path[::-1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.w and 0 <= ny < self.h):
                    continue
                k = self.known[ny, nx]
                if k == OCCUPIED:
                    continue
                nd = d + (1.0 if k == FREE else unknown_cost)
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (nd, nx, ny))
        return None

    def next_target(self, x: float, y: float, lookahead: float = 0.9,
                    arrive_dist: float = 0.6):
        """纯追踪的瞄准点，返回 (tx, ty, dist)；没有目标返回 None。

        三个坑：
        * 不能用"到第 N 个路点的方位"——路径拐弯时那个方向会指穿墙。
        * 路径不能每步重规划——相邻步的 frontier 会来回跳，车跟着抖。
        * **走到路径尽头必须重规划**——否则车会绕着最后一个路点画圈，永远"到不了"。
        """
        gx, gy = int(x), int(y)

        if self.path is not None:
            stale = any(self.known[b, a] != FREE for a, b in self.path[self.wp_idx:])
            lx, ly = self.path[-1]
            arrived = math.hypot(lx + 0.5 - x, ly + 0.5 - y) < arrive_dist
            if stale or arrived:
                self.path = None

        if self.path is None:
            # 优先：直接朝已知坐标的终点规划（未知区域可通行、代价更高）
            self.path = self._plan_to_goal(gx, gy)
            self.goal_planned = self.path is not None
            if self.path is None:
                # 兜底：终点被彻底围死时退回 frontier 探索
                self.path = self._bfs(gx, gy, self._is_frontier)
            self.wp_idx = 0
            if self.path is None:
                self.exhausted = True
                self.last_target = None
                return None

        self.exhausted = False
        # 跳过已经走过的路点
        while self.wp_idx < len(self.path) - 1:
            px, py = self.path[self.wp_idx]
            if math.hypot(px + 0.5 - x, py + 0.5 - y) < 0.5:
                self.wp_idx += 1
            else:
                break

        for i in range(self.wp_idx, len(self.path)):
            px, py = self.path[i]
            cx, cy = px + 0.5, py + 0.5
            if math.hypot(cx - x, cy - y) >= lookahead:
                self.last_target = (cx, cy)
                self.path_len = len(self.path) - self.wp_idx
                return cx, cy, math.hypot(cx - x, cy - y)

        px, py = self.path[-1]
        cx, cy = px + 0.5, py + 0.5
        self.last_target = (cx, cy)
        self.path_len = len(self.path) - self.wp_idx
        return cx, cy, math.hypot(cx - x, cy - y)

    # ---- 诊断 ----------------------------------------------------------------
    def explored_fraction(self, free_total: int) -> float:
        if free_total <= 0:
            return 0.0
        return float((self.known == FREE).sum()) / free_total
