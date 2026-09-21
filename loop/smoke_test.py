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
