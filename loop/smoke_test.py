"""回归 smoke test：保证后续改动不会悄悄搞坏核心功能。

    python -m loop.smoke_test

每项独立跑，最后打印 PASS/FAIL 汇总，退出码非 0 表示有失败。
"""
from __future__ import annotations

import inspect
import math
import os
import sys
import traceback

import numpy as np

from brain.neurons import resolve
from brain.wrap import make_brain
from config import Config, apply_preset
from loop.decode import Decoder
from loop.encode import Encoder
from loop.run import DEFAULT_DT, make_maze, run
from nav.memory import FREE, OCCUPIED, UNKNOWN, OccupancyMemory
from world.car import DiffDriveCar

RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
        except Exception as exc:                       # noqa: BLE001
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn
    return deco


def _setup(kind="fake", map_name="track", memory_gain=0.0, mode="dn", readout=""):
    cfg = Config()
    cfg.brain.kind = kind
    cfg.brain.device = "cpu"
    cfg.brain.dt = DEFAULT_DT[kind]
    cfg.brain.seed = 64
    apply_preset(cfg, kind)
    cfg.decoder.mode = mode
    cfg.decoder.memory_gain = memory_gain
    if readout:
        cfg.decoder.readout_path = readout
    maze = make_maze(map_name)
    sx, sy, sth = maze.start
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    brain = make_brain(cfg.brain, verbose=False)
    groups = resolve(brain, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    enc = Encoder(groups, cfg.encoder, brain.dt, car_max_speed=cfg.car.max_speed,
                  car_radius=cfg.car.radius)
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    brain.reset(64)
    return cfg, maze, car, brain, groups, enc, dec


# ---- Test 1 -----------------------------------------------------------------
@check("T1 0° 射线同属 left/right，且正对墙时 dmin_F 正确")
def t1():
    cfg, maze, car, brain, groups, enc, dec = _setup()
    zero = np.flatnonzero(np.isclose(enc.angles, 0.0, atol=1e-9))
    assert len(zero) == 1, f"应恰有一条 0° 射线，实际 {len(zero)}"
    i = int(zero[0])
    assert enc.left[i] and enc.right[i], "0° 射线必须同时属于 left 和 right"
    assert enc.front[i], "0° 射线必须属于正前方锥"

    # 把车放到 track 里贴着一面已知墙，朝它，看 dmin_F
    car.x, car.y, car.theta = 1.5, 1.5, 0.0     # 朝 +x，前方是空地
    d = enc.sense(maze, car)
    _, info = enc.inject(d)
    assert info["dmin_F"] > 0.5, f"前方开阔时 dmin_F 不该很小，得到 {info['dmin_F']:.3f}"

    # dmin_F 必须等于"正前方锥内最近距离 - 车半径"，直接对着定义验
    maze2 = make_maze("track")
    car2 = DiffDriveCar(2.5, 2.5, 0.0, cfg.car)
    d2 = enc.sense(maze2, car2)
    _, info2 = enc.inject(d2)
    expect = max(float(d2[enc.front].min()) - cfg.car.radius, 0.05)
    assert abs(info2["dmin_F"] - expect) < 1e-6, \
        f"dmin_F 应为 {expect:.3f}，得到 {info2['dmin_F']:.3f}"
    assert info2["dmin_F"] < 6.0, "正前方锥应该看到侧墙，dmin_F 不该是量程上限"


# ---- Test 2 -----------------------------------------------------------------
@check("T2 OccupancyMemory 不吃 Maze，只靠 angles+dists 建出墙")
def t2():
    sig = inspect.signature(OccupancyMemory.update)
    assert list(sig.parameters) == ["self", "x", "y", "angles", "dists"], \
        f"update 签名必须是 (x, y, angles, dists)，实际 {list(sig.parameters)}"

    cfg, maze, car, brain, groups, enc, dec = _setup()
    mem = OccupancyMemory(maze.w, maze.h, cfg.encoder.max_range)
    # 车放在 track 左上角，朝 +x
    car.x, car.y, car.theta = 2.5, 2.5, 0.0
    dists = enc.sense(maze, car)
    mem.update(car.x, car.y, car.theta + enc.angles, dists)

    # 建出来的图必须和真值一致：标成 OCCUPIED 的格子真值必须是墙，
    # 而且真值是墙的格子不能被标成 FREE。扫全图，不设范围。
    occ = np.argwhere(mem.known == OCCUPIED)
    fre = np.argwhere(mem.known == FREE)
    assert len(occ) >= 5, f"只标出 {len(occ)} 个墙格，建图基本没工作"
    wrong_occ = [tuple(p[::-1]) for p in occ if not maze.grid[p[0], p[1]]]
    wrong_free = [tuple(p[::-1]) for p in fre if maze.grid[p[0], p[1]]]
    assert not wrong_occ, f"把空地标成了墙: {wrong_occ[:5]}"
    assert not wrong_free, f"把墙标成了空地: {wrong_free[:5]}"


# ---- Test 3 -----------------------------------------------------------------
@check("T3 Dijkstra 能规划穿过 UNKNOWN")
def t3():
    cfg, maze, car, brain, groups, enc, dec = _setup()
    mem = OccupancyMemory(maze.w, maze.h, cfg.encoder.max_range)
    mem.set_goal(20.5, 12.5)
    # 地图全未知，只把车所在格设为 FREE
    mem.known[:] = UNKNOWN
    mem.known[2, 2] = FREE
    path = mem._plan_to_goal(2, 2)
    assert path is not None, "全未知地图上应该能规划出一条穿未知区的路径"
    assert path[0] == (2, 2) and path[-1] == (20, 12), "路径端点不对"
    assert len(path) > 10, "路径太短，不像真的规划到了终点"


# ---- Test 4 -----------------------------------------------------------------
@check("T4 只有 OCCUPIED 触发重规划，UNKNOWN 不触发")
def t4():
    cfg, maze, car, brain, groups, enc, dec = _setup()
    mem = OccupancyMemory(maze.w, maze.h, cfg.encoder.max_range)
    mem.set_goal(20.5, 12.5)
    mem.known[:] = UNKNOWN
    mem.known[2, 2] = FREE
    mem.next_target(2.5, 2.5)
    r0 = mem.replans
    assert r0 >= 1, "第一次应该规划"
    # 路径里塞一个 UNKNOWN（本来就是），再调一次不该重规划
    for _ in range(5):
        mem.next_target(2.5, 2.5)
    assert mem.replans == r0, \
        f"UNKNOWN 不该触发重规划，但 replans 从 {r0} 涨到 {mem.replans}"
    # 把路径中某个格子标成 OCCUPIED -> 必须重规划
    assert mem.path is not None, "上一句之后应该有路径"
    mid = mem.path[min(3, len(mem.path) - 1)]
    mem.known[mid[1], mid[0]] = OCCUPIED
    mem.next_target(2.5, 2.5)
    assert mem.replans > r0, "路径被证实是墙时必须重规划"


# ---- Test 5 -----------------------------------------------------------------
@check("T5 --mode readout 能正确初始化 readout / trace")
def t5():
    path = "readout_fake.npz"
    if not os.path.exists(path):
        raise RuntimeError(f"缺少 {path}，先跑 python -m loop.train_readout --brain fake")
    cfg, maze, car, brain, groups, enc, dec = _setup(mode="readout", readout=path)
    assert dec.readout is not None, "readout 没被加载"
    assert dec.trace is not None, "trace 没被建立"
    feats = dec.trace.observe(brain.step(inject=[]))
    assert feats.shape == (len(dec.trace.idx),), f"trace 形状不对: {feats.shape}"


# ---- Test 6 -----------------------------------------------------------------
@check("T6 FakeBrain track 稳定跑，无回归")
def t6():
    cfg, maze, car, brain, groups, enc, dec = _setup()
    steps = int(round(120.0 / cfg.brain.dt))       # 统一 120 物理秒
    st = run(cfg, maze, car, brain, enc, dec, steps, verbose=False)
    assert st["distance"] > 20.0, f"120 秒只走了 {st['distance']:.1f} 格，明显异常"
    assert st["collision_events"] < 0.25 * st["steps_done"], \
        f"碰撞率过高: {st['collision_events']}/{st['steps_done']}"
    assert st["coverage"] > 0.10, f"覆盖率过低: {st['coverage'] * 100:.1f}%"
    assert st["contact_ratio"] < 0.5, f"顶墙时间占比过高: {st['contact_ratio']:.3f}"


# ---- Path 生命周期（a-e） ------------------------------------------------------
def _mem_with_path():
    """造一条水平直路径 y=5, x=5..15，goal=(15,5)，已知格标 FREE。"""
    mem = OccupancyMemory(30, 30, 6.0)
    mem.known[:] = UNKNOWN
    for x in range(5, 16):
        mem.known[5, x] = FREE
    mem.goal = (15, 5)
    mem.path = [(x, 5) for x in range(5, 16)]
    mem.wp_idx = 0
    return mem


@check("T7a 车在路径上 -> 不 replan")
def t7a():
    mem = _mem_with_path()
    mem.next_target(6.5, 5.5)
    assert mem.path is not None, "在路径上不该失效"
    assert mem.replans == 0, f"不该重规划，replans={mem.replans}"
    assert mem.off_path_events == 0, "不该判为偏离"


@check("T7b 车轻微偏离/切弯 -> 不 replan")
def t7b():
    mem = _mem_with_path()
    mem.next_target(6.5, 6.3)          # 偏离 0.8 格，车格 (6,6) 与路径格 8 邻接
    assert mem.path is not None, "轻微偏离不该失效（会误杀正常切弯）"
    assert mem.replans == 0, f"不该重规划，replans={mem.replans}"
    assert mem.off_path_events == 0, "不该判为偏离"


@check("T7c 车明显离开剩余路径 -> 立即失效并重规划")
def t7c():
    mem = _mem_with_path()
    mem.next_target(6.5, 9.5)          # 偏离 4 格，拓扑上也不邻接
    assert mem.off_path_events == 1, f"应判为偏离，得到 {mem.off_path_events}"
    assert mem.replans == 1, f"应重规划一次，得到 {mem.replans}"
    assert mem.path is not None, "重规划后应有新路径"
    assert mem.path[0] == (6, 9), f"新路径应从车所在格出发，得到 {mem.path[0]}"


@check("T7d 车已前进到路径后段 -> wp_idx 向前同步，不追旧路点")
def t7d():
    mem = _mem_with_path()
    tgt = mem.next_target(12.5, 5.5)   # 车格 (12,5) 在路径索引 7
    assert mem.wp_idx >= 7, f"wp_idx 应单调同步到 7，得到 {mem.wp_idx}"
    assert mem.path is not None, "在路径上不该失效"
    assert mem.replans == 0, f"不该重规划，replans={mem.replans}"
    assert tgt is not None, "在路径上应给出瞄准点"
    tx, ty, d = tgt
    assert d < 1.6, f"瞄准点应在前方近处，得到 dist={d:.2f}"
    assert tx > 12.0, f"不该回头追旧路点，得到 target=({tx:.1f},{ty:.1f})"


@check("T7e 剩余路径新出现 OCCUPIED -> 立即重规划")
def t7e():
    mem = _mem_with_path()
    mem.known[5, 8] = OCCUPIED          # 路径中段被证实是墙
    mem.next_target(6.5, 5.5)
    assert mem.replans == 1, f"应重规划，得到 {mem.replans}"
    assert mem.path is not None, "重规划后应有新路径"
    assert (8, 5) not in mem.path, "新路径不该再穿过已证实是墙的格子"


def main():
    print("\n" + "=" * 66)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, msg in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"         {msg}")
    print("-" * 66)
    print(f"  {n_pass}/{len(RESULTS)} passed")
    print("=" * 66)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
