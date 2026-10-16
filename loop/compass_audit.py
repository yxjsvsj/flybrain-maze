"""P2d-25：中央复合体"朝向/罗盘"回路（E-PG / P-EN 及相关 ring 神经元）本机核验（只读）。

用**现有同源数据**（brain.npz + 官方 v1.0 注释表 + weights.npz）核对：
- 本机是否存在 EPG(P-EN 上下游)、PEN(P-EN)、ER(ring) 等对应细胞；
- 准确名称 / bodyId / 侧别 / superclass / 官方 class·subclass·supertype；
- **投射/连接**：1–2 跳下游是否到 DN/motor、EPG↔PEN 是否有连接（不凭名字/胞体位置断定"恢复了文献回路"）。
不注入、不控车、不改权重。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

PATTERNS = ("EPG", "PEN", "P-EN", "PEG", "ExR", "ER", "d7", "Δ7", "ring", "PB", "EB", "RB",
            "HeN", "IbS", "vbo", "FC2", "cre", "LAL")


def one_hop(csc, src) -> set:
    """src 的 1 跳下游（CSC 列 = 出边）。"""
    indptr, indices = csc
    return set(int(indices[e]) for j in src for e in range(indptr[j], indptr[j + 1]))


def two_hop(csc, src) -> set:
    t1 = one_hop(csc, src)
    return t1 | (set().union(*[one_hop(csc, [x]) for x in t1]) if t1 else set())


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=r"C:\Users\yang\AppData\Local\Temp\opencode\flyraw"
                                     r"\body-annotations-male-cns-v1.0-minconf-0.5.feather")
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)
    from flybrain.data import ensure_data
    data = ensure_data(args.data)
    meta = np.load(str(data / "brain.npz"))
    ct = np.array([str(x) for x in meta["cell_type"]])
    sc = np.array([str(x) for x in meta["superclass"]])
    side = np.array([str(x) for x in meta["side"]])
    ids = np.asarray(meta["ids"])
    uniq = sorted(set(ct.tolist()))
    print(f"[P2d-25] cell_type 种类={len(uniq)}  总 n={len(ct)}")

    print("\n== cell_type 命中（按外部名称模式；仅列出**精确类型名**）==")
    hits = {}
    for pat in PATTERNS:
        ex = [t for t in uniq if t == pat]
        pre = [t for t in uniq if t.startswith(pat) and t != pat]
        if ex or pre:
            hits[pat] = (ex, pre)
            print(f"  {pat:6s} 精确={ex}  前缀例={pre[:8]}{' …(+%d)' % (len(pre) - 8) if len(pre) > 8 else ''}")

    # 用官方注释核对这些类型的 class/subclass/supertype（避免只看名字）
    try:
        import pyarrow.feather as f
        tab = f.read_table(args.ann)
        cols = tab.column_names
        bid = np.asarray(tab.column("bodyId").to_numpy(), np.int64)
        pos = {int(b): i for i, b in enumerate(bid)}
        def gv(i, k):
            if k not in cols or i is None:
                return "未提供"
            v = tab.column(k)[i].as_py()
            return "未提供" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        print("\n== 官方注释字段（对本机命中的类型各取 1 例）==")
        for pat in ("EPG", "PEN", "ER", "ExR"):
            for t in (hits.get(pat, ([], []))[0] + hits.get(pat, ([], []))[1][:1])[:3]:
                i = int(np.flatnonzero(ct == t)[0])
                j = pos.get(int(ids[i]))
                print(f"  {t:14s} bodyId={int(ids[i])} side={side[i]} superclass={sc[i]} | "
                      f"class={gv(j,'class')} subclass={gv(j,'subclass')} supertype={gv(j,'supertype')}")
    except Exception as e:                  # noqa: BLE001
        print(f"  (官方表核验跳过: {e})")

    # 连接：EPG/PEN 是否彼此相连、是否可达 DN/motor
    from scipy import sparse
    W = sparse.load_npz(str(data / "weights.npz")).tocsc()
    csc = (W.indptr, W.indices)
    dn = set(np.flatnonzero(sc == "descending_neuron").tolist())
    motor = set(np.flatnonzero(np.char.find(np.char.lower(sc.astype(str)), "motor") >= 0).tolist())

    def one(src):
        return one_hop(csc, src)

    def two(src):
        return two_hop(csc, src)

    def grp(names):
        if not names:
            return []
        return np.flatnonzero(np.isin(ct, names)).tolist()

    epg = grp([t for t in uniq if t.startswith("EPG")])
    pen = grp([t for t in uniq if t.startswith("PEN") or t.startswith("P-EN")])
    er = grp([t for t in uniq if t.startswith("ER")])
    for label, src in (("EPG*", epg), ("PEN*/P-EN*", pen), ("ER*", er)):
        if not src:
            print(f"\n[{label}] 本机无此类型")
            continue
        t1, t2 = one(src), two(src)
        print(f"\n[{label}] n={len(src)}  1跳下游={len(t1)} 到DN={len(set(t1) & dn)} "
              f"到motor={len(set(t1) & motor)} | 2跳 到DN={len(t2 & dn)} 到motor={len(t2 & motor)}")
    if epg and pen:
        epg_to_pen = len(one(epg) & set(pen))
        pen_to_epg = len(one(pen) & set(epg))
        print(f"\n[EPG<->PEN] EPG->PEN 直连={epg_to_pen}  PEN->EPG 直连={pen_to_epg}")
    print("\n[边界] 只读核验；不注入/不控车/不改权重。名称命中/胞体位置**不**等同于恢复文献回路。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
