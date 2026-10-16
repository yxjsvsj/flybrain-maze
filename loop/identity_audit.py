"""P2d-24b：具体感觉神经元**身份对接**审计（只读；不做新的注入实验）。

核对：本机连接组版本/数据来源/索引映射；brain.npz 实际保存了哪些注释字段（是否有原始 bodyId/
class/subclass/type/侧别/神经束）；区分 **数组位置 / 模拟器全脑索引 / 原始 neuron ID**。
逐一核对外部线索 SNta29(触觉机械感)、SNpp59(本体感觉/弦音器)、LgLG3(腿部味觉) 在本机是否可精确命中。
不按名字/位置跨版本强行匹配；不注入、不调 K、不接 P2Control。
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    import flybrain
    from flybrain.data import ensure_data
    import os
    print(f"[P2d-24b] flybrain {getattr(flybrain, '__version__', '?')} @ {os.path.dirname(flybrain.__file__)}")
    p = ensure_data()
    print(f"  数据目录 = {p}")
    meta = np.load(os.path.join(p, "brain.npz"))
    print("\n== brain.npz 实际字段 ==")
    for k in sorted(meta.files):
        a = meta[k]
        sample = a[:3].tolist() if getattr(a, "ndim", 0) >= 1 else a.tolist()
        print(f"  {k:16s} shape={str(a.shape):14s} dtype={a.dtype}  例={sample}")
    # 是否有原始 ID / 分类层级字段
    raw_fields = [k for k in meta.files if any(s in k.lower() for s in
                  ("id", "bodyid", "body", "class", "subclass", "tract", "nerve", "type", "side",
                   "hemisphere", "region"))]
    print(f"\n  可能含原始 ID/分类层级的字段: {raw_fields}")
    print("  说明：**数组位置 = 模拟器全脑索引**（weights.npz 的 W 行列顺序）；"
          "本文件未见独立'原始 neuron ID'字段时，无法跨版本按 ID 对接。")

    W = np.load(os.path.join(p, "weights.npz")) if os.path.exists(os.path.join(p, "weights.npz")) \
        else None
    if W is not None:
        print(f"  weights.npz keys={sorted(W.files)}  n(行列)={W[W.files[0]].shape}")

    ct = np.array([str(x) for x in meta["cell_type"]]) if "cell_type" in meta.files else None
    side = np.array([str(x) for x in meta["side"]]) if "side" in meta.files else None
    print("\n== 外部线索亚型在本机的精确命中 ==")
    if ct is None:
        print("  无 cell_type 字段 -> 无法核对")
        return 0
    for name in ("SNta29", "SNpp59", "LgLG3", "SNpp", "SNta", "LgLG"):
        idx = np.flatnonzero(ct == name) if name in set(ct.tolist()) else \
            np.flatnonzero(np.char.find(ct.astype(str), name) >= 0)
        exact = name in set(ct.tolist())
        sides = sorted(set(side[idx].tolist())) if side is not None and len(idx) else []
        print(f"  {name:8s} 精确类型={exact}  命中 n={len(idx)}  侧别={sides}  例索引={idx[:5].tolist()}")

    # 具体亚型的接口证据：原始 id / 本机索引 / 侧别 / 下游
    from scipy import sparse
    Wm = sparse.load_npz(os.path.join(p, "weights.npz")).tocsc()
    indptr, indices = Wm.indptr, Wm.indices
    sc = np.array([str(x) for x in meta["superclass"]]) if "superclass" in meta.files else None
    dn = set(np.flatnonzero(sc == "descending_neuron").tolist()) if sc is not None else set()
    motor = set(np.flatnonzero(np.char.find(
        np.char.lower(sc.astype(str)), "motor") >= 0).tolist()) if sc is not None else set()
    ids = np.array(meta["ids"]) if "ids" in meta.files else None

    def one_hop(src):
        t = set()
        for j in src:
            for e in range(indptr[j], indptr[j + 1]):
                t.add(int(indices[e]))
        return t

    print("\n== 接口证据表（只读；下游=1跳/2跳；DN=descending_neuron, motor=vnc/cb_motor）==")
    for name in ("SNta29", "SNpp59", "LgLG3"):
        idx = np.flatnonzero(ct == name)
        if len(idx) == 0:
            print(f"  {name}: 本机无此类型")
            continue
        t1 = one_hop(idx.tolist())
        t2 = t1 | set().union(*[one_hop([x]) for x in list(t1)[:2000]]) if t1 else set()
        ids_s = ids[idx].tolist()[:5] if ids is not None else "?"
        sides = sorted(set(side[idx].tolist())) if side is not None else []
        print(f"  {name:8s} n={len(idx):4d} 本机索引例={idx[:3].tolist()} 原始ids例={ids_s} "
              f"侧别={sides} 1跳下游={len(t1)} 2跳到DN={len(t2 & dn)} 到motor={len(t2 & motor)}")
    print("  不确定性：本机 `ids` 为 10001.. 顺序号（非 FlyWire bodyId），跨版本只能靠"
          "**cell_type 名称**对应；未见独立原始 bodyId/神经束字段。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
