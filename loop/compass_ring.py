"""P2d-26 第一步：固定**可解释的环坐标**（只读核验 + 事先声明的重建法，标为推断）。

目标细胞的**具体 ID** 与**连接矩阵**（EPG→PEN / PEN→EPG，保留方向、权重、符号）。
不把"目标细胞数量"当连接强度。环序**不**用后续活动/角速度/期望朝向优化，**不**按 bodyId、
数组顺序、胞体左右，**不**默认 50=50 等角分区。

重建法（**事先声明、标为推断**）：以 **EPG→PEN 权重矩阵**为特征，做 EPG 间的行相似度，
取相似图的 **Fiedler 向量（谱序列化）** 作为一维环序；再用"相邻对相似度 vs 置换基线 + 首尾闭环"
判断该序**是否呈环**。若不足 → 停在"空间坐标尚不可识别"，不往下算相位。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def ring_metrics(S, order, seed=0):
    """环性指标：相邻对相似均值、首尾闭环、随机置换基线；附左右簇转换数。
    返回 dict；`is_ring` = 首尾闭环 >= 相邻均值*0.8（否则更像线序/两簇）。"""
    order = np.asarray(order)
    adj = np.array([S[order[i], order[(i + 1) % len(order)]] for i in range(len(order))])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(order))
    base = np.array([S[perm[i], perm[(i + 1) % len(perm)]] for i in range(len(perm))]).mean()
    wrap = float(S[order[-1], order[0]])
    return {"adj_mean": float(adj.mean()), "shuffle": float(base),
            "lift": float(adj.mean() / (base + 1e-9)), "wrap": wrap,
            "is_ring": bool(wrap >= adj.mean() * 0.8)}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    from flybrain.data import ensure_data
    from scipy import sparse
    data = ensure_data(args.data)
    meta = np.load(str(data / "brain.npz"))
    ct = np.array([str(x) for x in meta["cell_type"]])
    side = np.array([str(x) for x in meta["side"]])
    ids = np.asarray(meta["ids"])
    W = sparse.load_npz(str(data / "weights.npz")).tocsr()     # rows = postsynaptic
    epg = np.flatnonzero(ct == "EPG")
    epgt = np.flatnonzero(ct == "EPGt")
    pen_a = np.flatnonzero(ct == "PEN_a(PEN1)")
    pen_b = np.flatnonzero(ct == "PEN_b(PEN2)")
    pen = np.concatenate([pen_a, pen_b])
    print(f"[P2d-26 step1] EPG={len(epg)} EPGt={len(epgt)} PEN_a={len(pen_a)} PEN_b={len(pen_b)}")
    for nm, idx in (("EPG", epg), ("EPGt", epgt), ("PEN_a", pen_a), ("PEN_b", pen_b)):
        s = sorted(set(side[idx].tolist()))
        print(f"  {nm:6s} 侧别={s}  bodyId 例={ids[idx][:6].tolist()} … 全部 {len(idx)} 个")

    # 连接矩阵（方向：pre->post = W[post,pre]；保留权重与符号）
    Wc = W.tocsc()
    def sub(pre, post):
        return np.asarray(W[post][:, pre].todense(), np.float64)   # 形状 (post, pre)
    epg2pen = sub(epg, pen)      # EPG(pre) -> PEN(post)
    pen2epg = sub(pen, epg)      # PEN -> EPG
    def brief(m, name):
        nz = np.abs(m) > 0
        print(f"  {name:16s} 非零={nz.sum():4d}/{m.size}  ({nz.mean()*100:4.1f}%)  "
              f"w∈[{m[m!=0].min() if nz.any() else 0:.3f},{m[m!=0].max() if nz.any() else 0:.3f}]  "
              f"符号集={sorted(set(np.sign(m[nz]).astype(int).tolist())) if nz.any() else []}")
    print("\n== 连接矩阵（保留方向/权重/符号）==")
    brief(epg2pen, "EPG->PEN")
    brief(pen2epg, "PEN->EPG")

    # 事先声明的谱序列化（推断）：EPG 行相似度 -> Fiedler 向量
    r = epg2pen
    rn = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-9)
    S = rn @ rn.T                                     # EPG-EPG 相似度
    np.fill_diagonal(S, 0.0)
    Ssym = 0.5 * (S + S.T)
    d = Ssym.sum(1)
    L = np.diag(d) - Ssym
    wv, vv = np.linalg.eigh(L)
    order = np.argsort(vv[:, 1])                      # Fiedler 序
    # 环性判据：相邻对相似度 vs 置换基线；以及首尾闭环
    adj = np.array([Ssym[order[i], order[(i + 1) % len(order)]] for i in range(len(order))])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(order))
    adj_p = np.array([Ssym[perm[i], perm[(i + 1) % len(perm)]] for i in range(len(perm))])
    print("\n== 环序（谱序列化，推断）==")
    print(f"  Fiedler 相邻对 相似度 均值={adj.mean():.3f}  随机置换基线={adj_p.mean():.3f}  "
          f"提升={(adj.mean() / (adj_p.mean() + 1e-9)):.2f}x")
    print(f"  前 10 的 EPG bodyId 序={ids[epg][order][:10].tolist()} …")
    # 不需要闭环也可为"线序"；报告是否更像环：首尾对 vs 相邻对
    wrap = Ssym[order[-1], order[0]]
    print(f"  首尾闭环对 相似度={wrap:.3f}  vs 相邻均值={adj.mean():.3f}  "
          f"→ {'像环' if wrap > adj.mean() * 0.8 else '更像线序/无闭环'}")
    # 该序是否只是把左右两半球分开（假环）？
    s_ord = side[epg][order]
    trans = int(np.sum(s_ord[1:] != s_ord[:-1]))
    print(f"  序内左右侧转换次数={trans}（=1 表示只是 L…R 两簇，非环）  侧别序列前20={s_ord[:20].tolist()}")
    # 半球内部再各自做一次序列化，看是否出现子环
    for h in ("L", "R"):
        sub_i = np.flatnonzero(s_ord == h)
        if len(sub_i) < 4:
            continue
        o2 = order[sub_i]
        s2 = Ssym[np.ix_(o2, o2)]
        a2 = np.array([s2[i, (i + 1) % len(o2)] for i in range(len(o2))])
        print(f"  [{h}] n={len(o2)} 相邻相似均值={a2.mean():.3f} 首尾={s2[-1,0]:.3f} "
              f"→ {'像环' if s2[-1, 0] > a2.mean() * 0.8 else '非环'}")
    print("\n[边界] 环序为**连接图重建(推断)**，无官方 PB/EB 分区标签可验证；"
          "不据此算精确朝向相位；无足够依据则停在此。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
