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

# 路径失效原因（诊断用）。replans 是"真正执行了规划算法"的次数，
# invalidations 是"路径被判失效"的次数——两者分开统计，避免混淆。
INV_OCCUPIED = "OCCUPIED"                  # 计划中的格子被证实是墙
INV_ARRIVED = "ARRIVED"                    # 走到路径末端
INV_OFF_TOPOLOGY = "OFF_PATH_TOPOLOGY"     # 拓扑上不再贴着路径（距离仍在容差内）
INV_OFF_DISTANCE = "OFF_PATH_DISTANCE"     # 到剩余路径距离超过 off_path_tol
INV_NO_PATH = "NO_PATH"                    # 规划算法没能给出路径
INV_REASONS = (INV_OCCUPIED, INV_ARRIVED, INV_OFF_TOPOLOGY, INV_OFF_DISTANCE, INV_NO_PATH)


class OccupancyMemory:
    """占用栅格 + 已访问图 + 目标导向规划。

    known[y, x]: 0 未知 / 1 自由 / 2 占据
    visited[y, x]: 车真正到过的格子
    """

    def __init__(self, width: int, height: int, max_range: float,
                 unknown_cost: float = 4.0, lookahead: float = 0.9,
                 arrive_dist: float = 0.6, off_path_tol: float = 1.5,
                 free_conflict_threshold: int = 3):
        self.w, self.h = int(width), int(height)
        self.max_range = float(max_range)
        # 未知格的通行代价。1 = 完全敢穿未知区，越大越保守（只在必要时才走未知区）
        self.unknown_cost = float(unknown_cost)
        self.lookahead = float(lookahead)
        self.arrive_dist = float(arrive_dist)
        self.off_path_tol = float(off_path_tol)
        # 一个已被观测为 FREE 的格子，要被多少次"冲突的占据观测"才允许翻成 OCCUPIED。
        # 射线擦拐角时端点会被误归到自由格，单次冲突不足以推翻已有的自由证据；
        # 但也不能规定 FREE 永不翻转，否则早期的错误 FREE 永远纠正不回来。
        self.free_conflict_threshold = int(free_conflict_threshold)
        self.known = np.zeros((self.h, self.w), np.uint8)
        # FREE 格子上的冲突占据观测计数（连续累积，用于阈值判定）
        self.conflicts = np.zeros((self.h, self.w), np.uint8)
        self.visited = np.zeros((self.h, self.w), bool)
        self.map_revision = 0            # known 真正发生变化时才 +1
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
        self.last_offpath_dist = 0.0     # 最近一次判定的"到剩余路径距离"（到最近格中心）
        self.max_offpath_dist = 0.0      # 全程最大偏离距离（诊断用）
        self.last_offpath_polyline = 0.0  # 到剩余路径折线的最短距离（诊断对照）
        self.max_offpath_polyline = 0.0
        self.max_target_dist = 0.0       # 全程最大瞄准点距离（诊断用）
        # 失效原因分类计数
        self.invalidations = {r: 0 for r in INV_REASONS}
        # 规划失败的状态指纹 (map_revision, car_cell, goal)。三样都没变时，
        # 不必再跑一遍完全相同的 Dijkstra——NO_PATH 是事件而不是每个控制 tick。
        self.last_failed_plan_key = None
        # invariant 违规次数（goal_reached_map=True 却 known[goal]=OCCUPIED 之类）
        self.invariant_failures = 0
        self.last_invariant_msg = ""
        # 失效事件明细（调用方用 drain_events() 取走）
        self.event_capacity = 4096
        self._pending_events: list[dict] = []

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
        changed = False
        for ang, d in zip(angles, dists):
            d = float(d)
            is_hit = d < self.max_range - 1e-6
            free, hit = self._traverse(x, y, float(ang), d, is_hit)
            for gx, gy in free:
                if self.known[gy, gx] == UNKNOWN:
                    self.known[gy, gx] = FREE
                    changed = True
            if hit is None or hit == (ox, oy):
                continue
            hx, hy = hit
            if not (0 <= hx < self.w and 0 <= hy < self.h):
                continue
            # 终点格保护：任务给定一个可达的终点坐标，任何射线都不许把它写成墙。
            # 擦拐角时端点会被误归到终点格，一旦写成就再也规划不进去。
            # 这不算真值泄漏——终点坐标本来就是实验的输入。
            if self.goal is not None and (hx, hy) == self.goal:
                continue
            cur = self.known[hy, hx]
            if cur == UNKNOWN:
                self.known[hy, hx] = OCCUPIED
                changed = True
            elif cur == FREE:
                # 已有自由证据：单次冲突观测不足以推翻它（射线擦拐角会误判）。
                # 但也不能永不翻转，否则早期错误的 FREE 永远纠正不回来。
                # 累积到阈值才允许翻成 OCCUPIED。
                if self.conflicts[hy, hx] < 255:
                    self.conflicts[hy, hx] += 1
                if self.conflicts[hy, hx] >= self.free_conflict_threshold:
                    self.known[hy, hx] = OCCUPIED
                    self.conflicts[hy, hx] = 0
                    changed = True
        if 0 <= ox < self.w and 0 <= oy < self.h:
            if self.known[oy, ox] != FREE:
                self.known[oy, ox] = FREE
                self.conflicts[oy, ox] = 0
                changed = True
            self.visited[oy, ox] = True
        if self.goal is not None and self.known[self.goal[1], self.goal[0]] == FREE:
            self.goal_reached_map = True
        if changed:
            self.map_revision += 1

    def check_invariants(self) -> bool:
        """不变量检查。返回 True 表示有违规（调用方应报警）。"""
        if self.goal is None:
            return False
        gx, gy = self.goal
        if self.goal_reached_map and self.known[gy, gx] == OCCUPIED:
            self.invariant_failures += 1
            self.last_invariant_msg = ("goal_reached_map=True 但 known[goal]=OCCUPIED "
                                       f"(goal={self.goal})")
            return True
        return False

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
                # 防御性保护：终点格始终允许进入，即使 known 意外为 OCCUPIED
                # （update() 里的终点保护是第一道，这里是第二道）
                if k == OCCUPIED and (nx, ny) != (gx0, gy0):
                    continue
                nd = d + (1.0 if k == FREE else self.unknown_cost)
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (nd, nx, ny))
        return None

    def _polyline_dist(self, x: float, y: float, start_idx: int) -> float:
        """车到"剩余路径折线"（相邻格中心连成的线段）的最短距离。

        诊断对照用：`nearest_dist` 只是到最近格中心的点距离，在转角附近会偏大
        （车切在拐角内侧时，到两个格中心都远，但到折线其实很近）。
        如果 8800 次失效主要来自这个几何量在转角附近的离散误判，
        polyline_dist 会明显小于 nearest_dist。
        """
        if self.path is None:
            return float("inf")
        pts = [(px + 0.5, py + 0.5) for px, py in self.path[start_idx:]]
        if not pts:
            return float("inf")
        if len(pts) == 1:
            return math.hypot(pts[0][0] - x, pts[0][1] - y)
        best = float("inf")
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
            best = min(best, math.hypot(x1 + t * dx - x, y1 + t * dy - y))
        return best

    def _check_offpath(self, x: float, y: float):
        """返回 (reason, info)。reason == "" 表示仍在剩余路径附近。

        判据用**栅格拓扑 + 连续距离容差**，不用 target_dist 阈值：
          * 车格与剩余路径任一格 8 邻接（含自身）-> 仍算在路径附近（允许切弯）
          * 否则看车到最近剩余路径点的连续距离，超过 off_path_tol 才算明显偏离
        wp_idx **只允许前进**，不做最近点回溯。
        """
        if self.path is None:
            return "", {}
        path = self.path
        if not path[self.wp_idx:]:
            return "", {}

        # 最近点只在 [wp_idx, end) 里找，不回头
        best_i, best_d = self.wp_idx, float("inf")
        for i in range(self.wp_idx, len(path)):
            px, py = path[i]
            d = math.hypot(px + 0.5 - x, py + 0.5 - y)
            if d < best_d:
                best_d, best_i = d, i
        self.last_offpath_dist = best_d
        self.max_offpath_dist = max(self.max_offpath_dist, best_d)
        if best_i > self.wp_idx:                 # 单调向前同步
            self.wp_idx = best_i

        cx, cy = int(x), int(y)
        topology_near = any(abs(cx - px) <= 1 and abs(cy - py) <= 1
                            for px, py in path[self.wp_idx:])

        poly_d = self._polyline_dist(x, y, self.wp_idx)
        self.last_offpath_polyline = poly_d
        self.max_offpath_polyline = max(self.max_offpath_polyline, poly_d)

        reason = ""
        if not topology_near:
            reason = INV_OFF_DISTANCE if best_d > self.off_path_tol else INV_OFF_TOPOLOGY

        info = {
            "nearest_path_idx": best_i,
            "nearest_path_cell": path[best_i],
            "nearest_dist": best_d,
            "polyline_dist": poly_d,
            "topology_near": topology_near,
        }
        return reason, info

    def _record_event(self, x: float, y: float, reason: str, off_info: dict,
                      occupied_cell=None) -> None:
        if len(self._pending_events) >= self.event_capacity:
            return
        self._pending_events.append({
            "reason": reason,
            "car_x": float(x), "car_y": float(y), "car_cell": (int(x), int(y)),
            "wp_idx": int(self.wp_idx),
            "path_len": len(self.path) if self.path else 0,
            "old_path_head": self.path[0] if self.path else None,
            "old_path_tail": self.path[-1] if self.path else None,
            "nearest_path_idx": off_info.get("nearest_path_idx"),
            "nearest_path_cell": off_info.get("nearest_path_cell"),
            "nearest_dist": off_info.get("nearest_dist"),
            "polyline_dist": off_info.get("polyline_dist"),
            "topology_near": off_info.get("topology_near"),
            "occupied_cell": occupied_cell,
            "off_path_tol": float(self.off_path_tol),
            "replans": int(self.replans),
            # 重规划之后回填（用于判断"新路径是否几乎一样"）
            "new_path_len": None,
            "new_path_head": None,
            "new_path_tail": None,
        })

    def _fill_after_replan(self) -> None:
        if not self._pending_events:
            return
        ev = self._pending_events[-1]
        if ev.get("new_path_len") is not None:
            return
        ev["new_path_len"] = len(self.path) if self.path else 0
        ev["new_path_head"] = self.path[0] if self.path else None
        ev["new_path_tail"] = self.path[-1] if self.path else None

    def drain_events(self) -> list[dict]:
        """取走并清空待处理事件（调用方补充控制量后自行记录）。"""
        ev, self._pending_events = self._pending_events, []
        return ev

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
            occupied_cell = next(((a, b) for a, b in self.path[self.wp_idx:]
                                  if self.known[b, a] == OCCUPIED), None)
            lx, ly = self.path[-1]
            arrived = math.hypot(lx + 0.5 - x, ly + 0.5 - y) < self.arrive_dist
            off_reason, off_info = self._check_offpath(x, y)

            reason = ""
            if occupied_cell is not None:
                reason = INV_OCCUPIED
            elif arrived:
                reason = INV_ARRIVED
            elif off_reason:
                reason = off_reason

            if reason:
                self.invalidations[reason] += 1
                if reason in (INV_OFF_TOPOLOGY, INV_OFF_DISTANCE):
                    self.off_path_events += 1
                self._record_event(x, y, reason, off_info, occupied_cell)
                self.path = None

        if self.path is None:
            # NO_PATH 是**事件**，不是每个控制 tick 都要报一次的状态。
            # 如果地图（map_revision）、车所在格、终点三样都没变，那么再跑一遍
            # 完全相同的 Dijkstra 只会得到完全相同的失败——直接跳过。
            plan_key = (self.map_revision, (gx, gy), self.goal)
            if self.last_failed_plan_key == plan_key:
                self.exhausted = True
                self.last_target = None
                return None

            self.replans += 1        # 真正执行了规划算法
            # 优先：直接朝已知坐标的终点规划（未知区域可通行、代价更高）
            self.path = self._plan_to_goal(gx, gy)
            self.goal_planned = self.path is not None
            if self.path is None:
                # 兜底：终点被彻底围死时退回 frontier 探索
                self.path = self._bfs(gx, gy, self._is_frontier)
            if self.path is None:
                # 规划失败：记下状态指纹，同状态下不再重复尝试
                self.last_failed_plan_key = plan_key
                self.invalidations[INV_NO_PATH] += 1
                self._record_event(x, y, INV_NO_PATH, {})
                self.exhausted = True
                self.last_target = None
                return None
            self.last_failed_plan_key = None
            self.wp_idx = 0
            self._fill_after_replan()

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
