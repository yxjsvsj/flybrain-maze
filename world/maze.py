"""网格迷宫 + DDA 射线投射 + 递归回溯迷宫生成。

地图是字符串列表：'#' 墙，'.' 空地，'S' 起点，'G' 终点（可选）。
坐标：格 (gx, gy) 覆盖 [gx, gx+1) x [gy, gy+1)，世界坐标用格子数表示。
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


def _carve(cells_x: int, cells_y: int, rng, loop_chance: float) -> list[list[int]]:
    """递归回溯（迭代版）挖迷宫。返回 (2*cy+1) x (2*cx+1) 的 0/1 网格。"""
    h, w = 2 * cells_y + 1, 2 * cells_x + 1
    g = [[1] * w for _ in range(h)]
    g[1][1] = 0
    stack = [(1, 1)]
    while stack:
        x, y = stack[-1]
        nbrs = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = x + dx, y + dy
            if 1 <= nx < w - 1 and 1 <= ny < h - 1 and g[ny][nx] == 1:
                nbrs.append((nx, ny, x + dx // 2, y + dy // 2))
        if not nbrs:
            stack.pop()
            continue
        nx, ny, wx, wy = nbrs[int(rng.integers(len(nbrs)))]
        g[wy][wx] = 0
        g[ny][nx] = 0
        stack.append((nx, ny))

    # 纯树状迷宫到处是死胡同，贴墙走的车进去就出不来。随机打通一些墙制造环路。
    if loop_chance > 0:
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if g[y][x] == 1 and rng.random() < loop_chance:
                    vertical = g[y - 1][x] == 0 and g[y + 1][x] == 0
                    horizontal = g[y][x - 1] == 0 and g[y][x + 1] == 0
                    if vertical or horizontal:      # 只打通直线墙，不碰转角
                        g[y][x] = 0
    return g


def _scale(g: list[list[int]], k: int) -> list[list[int]]:
    """整格放大 k 倍，把 1 格宽走廊变成 k 格宽（车才开得进去）。"""
    if k <= 1:
        return g
    out = []
    for row in g:
        big = [v for v in row for _ in range(k)]
        for _ in range(k):
            out.append(list(big))
    return out


def generate(cells_x: int = 9, cells_y: int = 6, seed: int = 0, scale: int = 2,
             loop_chance: float = 0.08) -> list[str]:
    """生成迷宫字符串。scale=2 得到 2 格宽走廊。起点 S 在左上，终点 G 在右下。"""
    rng = np.random.default_rng(seed)
    g = _scale(_carve(cells_x, cells_y, rng, loop_chance), scale)

    free = [(x, y) for y, row in enumerate(g) for x, v in enumerate(row) if v == 0]
    if not free:
        raise ValueError("生成的迷宫没有空地")
    sx, sy = min(free, key=lambda p: (p[1], p[0]))          # 左上
    gx, gy = max(free, key=lambda p: (p[1], p[0]))          # 右下

    rows = []
    for y, row in enumerate(g):
        line = "".join("#" if v else "." for v in row)
        rows.append(line)
    rows[sy] = rows[sy][:sx] + "S" + rows[sy][sx + 1:]
    rows[gy] = rows[gy][:gx] + "G" + rows[gy][gx + 1:]
    return rows


MAPS = {
    # 绕圈跑道：两间房上下相连，贴墙走会一直绕圈。默认地图。
    "track": [
        "########################",
        "#......................#",
        "#.........#............#",
        "#.........#............#",
        "#.........#............#",
        "#.........#............#",
        "#.........#............#",
        "#.........#............#",
        "#.........#............#",
        "#.........#............#",
        "#.........#............#",
        "#.........#............#",
        "#.........#............#",
        "#......................#",
        "########################",
    ],
    # 开阔房间 + 柱子：测试直行和绕障
    "open": [
        "######################",
        "#....................#",
        "#....................#",
        "#....##....##........#",
        "#....##....##........#",
        "#....................#",
        "#..........##........#",
        "#..........##........#",
        "#....................#",
        "#.......##...........#",
        "#.......##...........#",
        "#....................#",
        "#....................#",
        "######################",
    ],
    # 两格宽的之字形走廊：测试贴墙走
    "corridor": [
        "########################",
        "#......................#",
        "#.####################.#",
        "#.#..................#.#",
        "#.#.################.#.#",
        "#.#.#..............#.#.#",
        "#.#.#.############.#.#.#",
        "#.#.#..............#.#.#",
        "#.#.################.#.#",
        "#.#..................#.#",
        "#.####################.#",
        "#......................#",
        "########################",
    ],
}

# 固定种子的递归回溯迷宫（2 格宽走廊，含死胡同和少量环路）。
# 换一张就改种子：--map gen --maze-seed 3
MAPS["maze"] = generate(seed=7)
MAPS["maze_hard"] = generate(cells_x=13, cells_y=9, seed=11, loop_chance=0.04)


class Maze:
    def __init__(self, rows: list[str] | str):
        if isinstance(rows, str):
            rows = [r for r in rows.strip("\n").splitlines() if r.strip()]
        width = {len(r) for r in rows}
        if len(width) != 1:
            raise ValueError(f"地图每行长度必须一致，现在是 {sorted(width)}")
        self.rows = rows
        self.h, self.w = len(rows), len(rows[0])
        self.grid = np.array([[0 if c in ".SG" else 1 for c in r] for r in rows], np.uint8)
        self.start = self._find("S", default=(1.5, 1.5, 0.0))
        self.goal = self._find("G", default=None)
        self._goal_field: np.ndarray | None = None

    @classmethod
    def named(cls, name: str) -> "Maze":
        if name not in MAPS:
            raise KeyError(f"没有地图 {name!r}，可选：{sorted(MAPS)}")
        return cls(MAPS[name])

    @classmethod
    def load(cls, path: str) -> "Maze":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(fh.read())

    def _find(self, ch: str, default):
        for gy, row in enumerate(self.rows):
            gx = row.find(ch)
            if gx >= 0:
                return (gx + 0.5, gy + 0.5, 0.0)
        return default

    # ---- 查询 ----------------------------------------------------------------
    def is_wall(self, x: float, y: float) -> bool:
        gx, gy = int(math.floor(x)), int(math.floor(y))
        if not (0 <= gx < self.w and 0 <= gy < self.h):
            return True
        return bool(self.grid[gy, gx])

    def raycast(self, ox: float, oy: float, angle: float, max_range: float) -> float:
        """从 (ox, oy) 沿 angle 投射，返回撞墙距离（超出地图按墙处理）。"""
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

        t = 0.0
        while t <= max_range:
            if t_max_x < t_max_y:
                t, gx = t_max_x, gx + step_x
                t_max_x += t_delta_x
            else:
                t, gy = t_max_y, gy + step_y
                t_max_y += t_delta_y
            if not (0 <= gx < self.w and 0 <= gy < self.h):
                return t
            if self.grid[gy, gx]:
                return t
        return max_range

    def raycast_many(self, x: float, y: float, angles: np.ndarray, max_range: float) -> np.ndarray:
        return np.array([self.raycast(x, y, float(a), max_range) for a in angles], np.float32)

    def raycast_cells(self, ox: float, oy: float, angle: float, max_range: float):
        """建图用：返回 (free_cells, hit_cell)。

        free_cells 是射线穿过且未命中的格子；hit_cell 是真正挡住它的墙格（没命中为 None）。
        不要用"沿射线采样"来推命中格——命中点正好落在格子边界上，int() 会算成相邻
        的自由格，把空地标成墙，整张图就废了。
        """
        dx, dy = math.cos(angle), math.sin(angle)
        gx, gy = int(math.floor(ox)), int(math.floor(oy))
        free: list[tuple[int, int]] = []

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

        t = 0.0
        while t <= max_range:
            if t_max_x < t_max_y:
                t, gx = t_max_x, gx + step_x
                t_max_x += t_delta_x
            else:
                t, gy = t_max_y, gy + step_y
                t_max_y += t_delta_y
            if not (0 <= gx < self.w and 0 <= gy < self.h):
                return free, (gx, gy)          # 出界按墙处理
            if self.grid[gy, gx]:
                return free, (gx, gy)
            free.append((gx, gy))
        return free, None

    def free_cells(self):
        ys, xs = np.nonzero(self.grid == 0)
        return list(zip(xs.tolist(), ys.tolist()))

    def goal_field(self) -> np.ndarray | None:
        """从终点 BFS 出每个格子的最短路径长度（-1 = 不可达）。只在第一次调用时算。"""
        if self.goal is None:
            return None
        if self._goal_field is None:
            gx, gy = int(self.goal[0]), int(self.goal[1])
            d = np.full((self.h, self.w), -1, np.int32)
            d[gy, gx] = 0
            q = deque([(gx, gy)])
            while q:
                x, y = q.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.w and 0 <= ny < self.h
                            and self.grid[ny, nx] == 0 and d[ny, nx] < 0):
                        d[ny, nx] = d[y, x] + 1
                        q.append((nx, ny))
            self._goal_field = d
        return self._goal_field

    def dist_to_goal(self, x: float, y: float) -> int:
        """车所在格子到终点的最短路径长度（格数）；-1 = 没有终点或不可达。"""
        f = self.goal_field()
        if f is None:
            return -1
        gx, gy = int(math.floor(x)), int(math.floor(y))
        if not (0 <= gx < self.w and 0 <= gy < self.h):
            return -1
        return int(f[gy, gx])

    def optimal_path(self) -> int:
        """起点到终点的最优路径长度，用作难度参考。"""
        return self.dist_to_goal(*self.start[:2])
