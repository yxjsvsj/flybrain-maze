"""记忆模块：从射线建占用栅格，追踪已探索区域，给出"下一步往哪走"的瞄准点。

**只用传感器数据 + 位姿**，不接触 Maze 真值。接口是
    update(x, y, world_ray_angles, dists)
不接收 Maze 实例——否则实物上换 SLAM 时这一层要重写，而且评估会失真。
终点坐标是已知输入（不需要靠探索发现它）。

规划策略：直接朝已知坐标的终点做 Dijkstra。未知区域**允许通行但代价更高**，
所以车会朝终点方向扎进没探过的地方；撞到墙后地图更新、自动重规划。
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
    """占用栅格 + 已访问图 + 目标导向规划。

    known[y, x]: 0 未知 / 1 自由 / 2 占据
    visited[y, x]: 车真正到过的格子
    """

    def __init__(self, width: int, height: int, max_range: float,
                 unknown_cost: float = 4.0, lookahead: float = 0.9,
                 arrive_dist: float = 0.6, off_path_tol: float = 1.5):
        self.w, self.h = int(width), int(height)
        self.max_range = float(max_range)
        # 未知格的通行代价。1 = 完全敢穿未知区，越大越保守（只在必要时才走未知区）
        self.unknown_cost = float(unknown_cost)
        self.lookahead = float(lookahead)
        self.arrive_dist = float(arrive_dist)
        self.off_path_tol = float(off_path_tol)
        self.known = np.zeros((self.h, self.w), np.uint8)
        self.visited = np.zeros((self.h, self.w), bool)
        self.goal: tuple[int, int] | None = None
        self.last_target: tuple[float, float] | None = None
        self.path: list[tuple[int, int]] | None = None
        self.wp_idx = 0
        self.path_len = 0
        self.exhausted = False
        self.goal_planned = False        # 规划器当前能解出到终点的路径
        self.goal_reached_map = False    # 终点格子已在地图上被确认为自由
        self.replans = 0                 # 真正执行规划算法的次数（诊断用）
        self.off_path_events = 0         # 因明显偏离路径而失效重规划的次数
        self.last_offpath_dist = 0.0     # 最近一次判定的"到剩余路径距离"
        self.max_offpath_dist = 0.0      # 全程最大偏离距离（诊断用）
        self.max_target_dist = 0.0       # 全程最大瞄准点距离（诊断用）

    # ---- 建图（只用 angles + dists） -----------------------------------------
    def _traverse(self, ox: float, oy: float, angle: float, dist: float,
                  is_hit: bool):
        """基于网格边界的 DDA。返回 (free_cells, hit_cell)。

        free_cells: 射线在命中前穿过的格子
        hit_cell:   命中距离处的格子；没命中（超量程/出界且非墙）为 None

        不能用"每 0.4 格采样"——采样会漏格子，而且命中点落在格子边界上时
        int() 会算成相邻的自由格，把空地标成墙，整张图就废了。
        """
        dx, dy = math.cos(angle), math.sin(angle)
        gx, gy = int(math.floor(ox)), int(math.floor(oy))

        if dx > 0:
            step_x, t_max_x = 1, ((gx + 1) - ox) / dx
        elif dx < 0:
            step_x, t_max_x = -1, (gx - ox) / dx
        else:
            step_x, t_max_x = 0, math.inf
        t_delta_x = abs(1.0 / dx) if dx else math.inf

        if dy > 0:
            step_y, t_max_y = 1, ((gy + 1) - oy) / dy
        elif dy < 0:
            step_y, t_max_y = -1, (gy - oy) / dy
        else:
            step_y, t_max_y = 0, math.inf
        t_delta_y = abs(1.0 / dy) if dy else math.inf

        free: list[tuple[int, int]] = []
        # 容差不能取 1e-9：dists 是 float32，DDA 的 t 是 float64，float32 舍入会让
        # t 比 dist 大 ~3e-8，于是几乎所有射线都被误判成"没命中"，整张图只剩极少数墙。
        tol = 1e-4 * max(1.0, dist)
        t = 0.0
        while True:
            if t_max_x < t_max_y:
                t, gx = t_max_x, gx + step_x
                t_max_x += t_delta_x
            else:
                t, gy = t_max_y, gy + step_y
                t_max_y += t_delta_y
            if t > dist + tol:
                return free, None            # 射线在进入这个格子前就到头了
            if not (0 <= gx < self.w and 0 <= gy < self.h):
                return free, ((gx, gy) if is_hit else None)   # 出界按墙处理
            if is_hit and t >= dist - tol:
                return free, (gx, gy)        # 命中距离处的格子就是墙
            free.append((gx, gy))

    def update(self, x: float, y: float, angles: np.ndarray, dists: np.ndarray) -> None:
        """angles 是**世界坐标**下的射线角，dists 是 Encoder.sense() 的原始测距。"""
        ox, oy = int(x), int(y)
        for ang, d in zip(angles, dists):
            d = float(d)
            is_hit = d < self.max_range - 1e-6
            free, hit = self._traverse(x, y, float(ang), d, is_hit)
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

    # ---- 探索策略（兜底用） ---------------------------------------------------
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

    def _plan_to_goal(self, sx: int, sy: int):
        """Dijkstra：已知自由代价 1，未知代价 self.unknown_cost，占据不可通行。

        "知道终点坐标、直接朝它找过去"：未知区域允许通行（代价更高），所以车会朝
        终点方向扎进没探过的地方；撞到墙以后地图更新、自动重规划。
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
                nd = d + (1.0 if k == FREE else self.unknown_cost)
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (nd, nx, ny))
        return None

    def _sync_and_check_offpath(self, x: float, y: float) -> bool:
        """判断车是否**明显离开了剩余路径**，同时把 wp_idx 单调向前同步。

        为什么需要它：车一旦跑离路径（打滑、逃逸、急转），旧路径的
        `stale`（格子被证实是墙）和 `arrived`（到达路径末端）都不会触发，
        路径永远不失效。于是前瞻点越来越远，`target_dist` 涨到 5~9 格，
        而纯追踪增益是 `2*v*sin(err)/L`，L 一大增益就塌，车再也转不回来。

        判据用**栅格拓扑 + 连续距离容差**，不用 target_dist 阈值：
          * 车格与剩余路径任一格 8 邻接（含自身）-> 仍算在路径附近（允许切弯）
          * 否则看车到最近剩余路径点的连续距离，超过 off_path_tol 才算明显偏离

        wp_idx **只允许前进**，不做最近点回溯。
        """
        if self.path is None:
            return False
        rem = self.path[self.wp_idx:]
        if not rem:
            return False

        # 最近点只在 [wp_idx, end) 里找，不回头
        best_i, best_d = self.wp_idx, float("inf")
        for i in range(self.wp_idx, len(self.path)):
            px, py = self.path[i]
            d = math.hypot(px + 0.5 - x, py + 0.5 - y)
            if d < best_d:
                best_d, best_i = d, i
        self.last_offpath_dist = best_d
        self.max_offpath_dist = max(self.max_offpath_dist, best_d)
        if best_i > self.wp_idx:                 # 单调向前同步
            self.wp_idx = best_i

        cx, cy = int(x), int(y)
        for px, py in self.path[self.wp_idx:]:
            if abs(cx - px) <= 1 and abs(cy - py) <= 1:
                return False                     # 拓扑上仍贴着路径
        return best_d > self.off_path_tol        # 明显离开

    def next_target(self, x: float, y: float, lookahead: float | None = None):
        """纯追踪的瞄准点，返回 (tx, ty, dist)；没有目标返回 None。

        三个坑：
        * 不能用"到第 N 个路点的方位"——路径拐弯时那个方向会指穿墙。
        * 路径不能每步重规划——相邻步的目标会来回跳，车跟着抖。
        * **走到路径尽头、或明显偏离路径，都必须重规划**——否则车会绕着
          最后一个路点画圈，或者追着一个越来越远的陈旧路点漂走。
        """
        la = self.lookahead if lookahead is None else float(lookahead)
        gx, gy = int(x), int(y)

        if self.path is not None:
            # 只有"计划中的格子被传感器证实是墙"才因 stale 失效。
            # UNKNOWN 不能算 stale——否则 Dijkstra 规划出来的穿未知区路径会在
            # 下一帧立刻被判过期，变成每个控制周期都重新规划。
            stale = any(self.known[b, a] == OCCUPIED for a, b in self.path[self.wp_idx:])
            lx, ly = self.path[-1]
            arrived = math.hypot(lx + 0.5 - x, ly + 0.5 - y) < self.arrive_dist
            off_path = self._sync_and_check_offpath(x, y)
            if off_path:
                self.off_path_events += 1
            if stale or arrived or off_path:
                self.path = None

        if self.path is None:
            # 优先：直接朝已知坐标的终点规划（未知区域可通行、代价更高）
            self.path = self._plan_to_goal(gx, gy)
            self.goal_planned = self.path is not None
            if self.path is None:
                # 兜底：终点被彻底围死时退回 frontier 探索
                self.path = self._bfs(gx, gy, self._is_frontier)
            self.wp_idx = 0
            self.replans += 1        # 只在真正执行规划时 +1，invalidation 不重复计数
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
            d = math.hypot(cx - x, cy - y)
            if d >= la:
                self.last_target = (cx, cy)
                self.path_len = len(self.path) - self.wp_idx
                self.max_target_dist = max(self.max_target_dist, d)
                return cx, cy, d

        px, py = self.path[-1]
        cx, cy = px + 0.5, py + 0.5
        d = math.hypot(cx - x, cy - y)
        self.last_target = (cx, cy)
        self.path_len = len(self.path) - self.wp_idx
        self.max_target_dist = max(self.max_target_dist, d)
        return cx, cy, d

    # ---- 诊断 ----------------------------------------------------------------
    def explored_fraction(self, free_total: int) -> float:
        if free_total <= 0:
            return 0.0
        return float((self.known == FREE).sum()) / free_total
