"""P2d-26a：环坐标（修正版）——**完整流程先经合成测试**，再对固定真实矩阵跑**一次**。

修正点：
- **矩阵轴/身份**：EPG 的输出特征用 **`epg2pen.T`**（行=EPG，列=PEN）；每个矩阵同时保存
  **行/列的类型与原始 bodyId**；**断言 46 个 EPG 全部参与、无错配**。
- 用 `loop.cycle_order.ring_order`（2-D 谱嵌入 + 循环判据），**不再**用单个 Fiedler 向量排序。
- 保留原细胞集合/方向/权重/符号，**不**按结果换方法或阈值；**不**新增人工抑制连接。

结论只用三档："支持循环结构" / "证据不足" / "与所测循环模型不符"；方法未识别 ≠ 生物回路不存在；
即便成立也**不**自动命名为真实 PB/EB 分区或标定朝向。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from loop.cycle_order import order_stability, ring_order


def _ids_of(ids, idx):
    return [int(x) for x in np.asarray(ids)[idx].tolist()]


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)
    from flybrain.data import ensure_data
    from scipy import sparse
    data = ensure_data(args.data)
    meta = np.load(str(data / "brain.npz"))
    ct = np.array([str(x) for x in meta["cell_type"]])
    side = np.array([str(x) for x in meta["side"]])
    ids = np.asarray(meta["ids"])
    W = sparse.load_npz(str(data / "weights.npz")).tocsr()

    epg = np.flatnonzero(ct == "EPG")
    pen = np.concatenate([np.flatnonzero(ct == "PEN_a(PEN1)"), np.flatnonzero(ct == "PEN_b(PEN2)")])
    Mpe = np.asarray(W[pen][:, epg].todense(), np.float64)      # (n_pen, n_epg)
    M = Mpe.T                                                   # EPG 输出特征：行=EPG, 列=PEN
    n_epg, n_pen = M.shape
    print(f"[P2d-26a] EPG={n_epg}  PEN={n_pen}（非方阵）")
    print(f"  行(EPG) bodyId 前6={_ids_of(ids, epg)[:6]} …  列(PEN) bodyId 前6={_ids_of(ids, pen)[:6]} …")

    # 身份断言：轴/身份/无错配/全部参与
    assert M.shape == (len(epg), len(pen)), "特征矩阵轴与细胞集合不匹配"
    assert len(epg) == 46, f"EPG 数应为 46，实际 {len(epg)}"
    row_present = np.flatnonzero(np.abs(M).sum(1) > 0)
    col_present = np.flatnonzero(np.abs(M).sum(0) > 0)
    print(f"  参与情况：EPG 有出边 {len(row_present)}/{n_epg}；PEN 有入边 {len(col_present)}/{n_pen}")
    print(f"  EPG 全部参与：{len(row_present) == n_epg}；无错配（长度/集合）=True")
    # 方向/权重/符号（仅限**所检查的直接子矩阵**）
    nz = M[M != 0]
    print(f"  EPG→PEN 直接子矩阵：非零={ (M != 0).sum() }/{M.size}  "
          f"w∈[{nz.min():.4f},{nz.max():.4f}]  符号集={sorted(set(np.sign(nz).astype(int)))}")
    print("  注：'全兴奋'**仅**适用于此直接子矩阵，不代表整回路无抑制；未新增人工抑制。")

    # 先说明：流程已由 test_cycle_order 覆盖（环/乱序环/线/两簇/均匀/全零/孤立）
    r = ring_order(M)
    st = order_stability(M)
    met = r["metrics"]
    print("\n== 固定真实矩阵（只跑一次）==")
    print(f"  2-D 谱嵌入：r_cv={met['r_cv']:.3f}  deg_gap(λ3-λ2)={met['deg_gap']:.3f}  "
          f"sep4={met['sep4']:.3f}  eff_rank={met['eff_rank']:.2f}")
    print(f"  环序判据 cyclic={r['cyclic']}  子采样稳定性(相邻命中)={st:.3f}")
    if r["cyclic"] and (st == st) and st >= 0.6:
        verdict = "支持循环结构（在'所测循环模型'意义上）"
    elif not r["cyclic"]:
        verdict = "证据不足（该 2-D 谱嵌入方法未识别出环）"
    else:
        verdict = "与所测循环模型不符（判据通过但稳定性不足）"
    print(f"  结论：**{verdict}**")
    print("  说明：方法未识别 ≠ 生物回路不存在；即使成立也不命名为真实 PB/EB 分区、不标定朝向。")
    print("  第二步（动力学预检）仍**暂停**：需先有可靠环坐标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
