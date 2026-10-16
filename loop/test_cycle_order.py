"""P2d-26a 回归：循环序流程在**合成案例**上的行为（不能预给正确环序）。无需脑。

覆盖：标准环、打乱编号的环、线、两簇、均匀全连接、全零、孤立节点。
"""
from __future__ import annotations

import sys

import numpy as np

from loop.cycle_order import order_stability, ring_order

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def cyclic_feature(n, wrap=True):
    """每行 = 环上/线上的局部窗口（i-1,i,i+1）的 one-hot；wrap=环，否则线（端不接）。"""
    M = np.zeros((n, n))
    for i in range(n):
        js = [i - 1, i, i + 1]
        for j in js:
            if wrap:
                M[i, j % n] = 1.0
            elif 0 <= j < n:
                M[i, j] = 1.0
    return M


def is_ring_seq(order, n) -> bool:
    """order 是否为环序（相邻差 mod n 全相等，方向 ±1）。"""
    o = list(order)
    d = np.diff(o + [o[0]]) % n
    return bool(np.all(d == d[0]) and int(d[0]) in (1, n - 1))


def main() -> int:
    print("=========== cycle_order pipeline ===========")
    n = 24
    # 1) 标准环
    r = ring_order(cyclic_feature(n, wrap=True))
    check("standard ring -> cyclic=True", r["cyclic"])
    check("standard ring -> recovered order is a cycle", is_ring_seq(r["order"], n))
    # 2) 打乱编号的环（输入 ID 重排）——仍应判 cyclic，且序对应重排
    perm = np.random.default_rng(1).permutation(n)
    Msh = cyclic_feature(n, wrap=True)[perm][:, perm]
    r_sh = ring_order(Msh)
    check("shuffled ring -> cyclic=True", r_sh["cyclic"])
    check("shuffled ring -> order maps back to the ring (inverse perm)",
          is_ring_seq(perm[r_sh["order"]], n))
    # 3) 线
    check("line -> cyclic=False", not ring_order(cyclic_feature(n, wrap=False))["cyclic"])
    # 4) 两簇
    B = np.zeros((n, n))
    B[:n // 2, :n // 2] = 1.0
    B[n // 2:, n // 2:] = 1.0
    np.fill_diagonal(B, 0.0)
    check("two-cluster -> cyclic=False", not ring_order(B)["cyclic"])
    # 5) 均匀全连接（退化，不该判环）
    check("fully-connected uniform -> cyclic=False", not ring_order(np.ones((n, n)))["cyclic"])
    # 6) 全零
    check("all-zero -> cyclic=False", not ring_order(np.zeros((n, n)))["cyclic"])
    # 7) 孤立节点（一个零行）
    iso = cyclic_feature(n, wrap=True)
    iso[5] = 0.0
    check("with an isolated node -> still runs (bool)", isinstance(ring_order(iso)["cyclic"], bool))
    # 8) 非方阵（特征数 ≠ 源数）：补零列，环仍应可识别
    Mns = np.hstack([cyclic_feature(n, wrap=True), np.zeros((n, 6))])
    check("non-square (n=24, feat=30) runs and stays cyclic", Mns.shape[1] != Mns.shape[0]
          and ring_order(Mns)["cyclic"])
    check("non-square recovered order length == n", len(ring_order(Mns)["order"]) == n)
    # 9) 重排输入 ID 后身份一致：行 id 随行一起重排，逆置换可还原环序
    rowid = np.arange(n) + 10000
    perm2 = np.random.default_rng(7).permutation(n)
    Mrow = cyclic_feature(n, wrap=True)[perm2]
    rid2 = rowid[perm2]
    ord2 = ring_order(Mrow)["order"]
    check("row-ID reorder keeps identity (inverse perm restores ring)",
          is_ring_seq(perm2[ord2], n) and len(rid2) == n)
    # 稳定性：环高、线低
    st_ring = order_stability(cyclic_feature(n, wrap=True), reps=12)
    st_line = order_stability(cyclic_feature(n, wrap=False), reps=12)
    check("stability: ring > line", st_ring > st_line)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
