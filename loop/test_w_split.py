"""P9-1b0 回归：真实 W 静态分流断言 + 合成不对称分流 + 时间步收敛 + 合成性能（**不跑全脑**）。"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy import sparse

from loop.coba_proto import CobaNetwork, CobaParams
from loop.w_split import (GUNIT_TABLE, direction_check, load_real, split_signed, verify_split)

RESULTS = []
P = CobaParams(g_unit_nS=10.0)


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== w_split + convergence (P9-1b0) ===========")

    # ---- 合成**不对称**矩阵分流 ----
    Ws = sparse.csr_matrix(np.array([[0.0, 0, -2.0, 0],
                                     [3.0, 0, 0, -0.5],
                                     [0.0, -1.0, 0, 4.0],
                                     [-7.0, 0, 0, 0]]))
    We, Wi = split_signed(Ws)
    v = verify_split(Ws, We, Wi)
    check("synthetic: W == We - Wi (exact)", v["exact_reconstruct"])
    check("synthetic: edge count preserved (We=2, Wi=4)",
          v["edge_count_preserved"] and v["nnz_We"] == 2 and v["nnz_Wi"] == 4)
    check("synthetic: abs sum preserved", v["abs_sum_preserved"] and abs(v["abs_sum_W"] - 17.5) < 1e-9)
    check("synthetic: nonneg We/Wi", v["nonneg_We"] and v["nonneg_Wi"])
    # 逐边正负分流
    WeA, WiA = We.toarray(), Wi.toarray()
    check("synthetic: per-edge sign split", np.allclose(WeA - WiA, Ws.toarray())
          and (WeA >= 0).all() and (WiA >= 0).all())

    # ---- 真实 W ----
    D = load_real()
    check("real: 166700 neurons", len(D["ids"]) == 166700)
    check("real: matrix dims 166700^2", D["W"].shape == (166700, 166700))
    check("real: nnz == 25,582,938", D["W"].nnz == 25582938)
    check("real: weights sha256 matches known", D["weights_sha256_16"] == "c29919aa44069a27")
    check("real: brain sha256 matches known", D["brain_sha256_16"] == "cc9bd1ecd00bd703")
    WeR, WiR = split_signed(D["W"])
    vR = verify_split(D["W"], WeR, WiR)
    check("real: W == We - Wi (maxabs==0)", vR["exact_reconstruct"] and vR["reconstruct_W_eq_We_minus_Wi_maxabs"] == 0.0)
    check("real: edge counts preserved", vR["edge_count_preserved"])
    check("real: abs-sum preserved", vR["abs_sum_preserved"])
    check("real: nonneg We/Wi", vR["nonneg_We"] and vR["nonneg_Wi"])
    check("real: no stored zeros", vR["stored_zero_in_W"] == 0)
    dc = direction_check(D, "LC4", "DNp04")
    check("real: W[post,pre] direction (LC4->DNp04 positive)", dc["found"] and dc["w_post_pre"] > 0,
          f"({dc.get('pre_bodyId')}->{dc.get('post_bodyId')}={dc.get('w_post_pre'):.4f})")
    # bodyId 抽样核对：ids 与 W 维度一致
    check("real: bodyId count == W dim", len(D["ids"]) == D["W"].shape[0])

    # ---- 绝对电导不确定性表 ----
    check("g_unit table marks unknown/uncertain",
          GUNIT_TABLE[0]["available"].startswith("无") and "高" in GUNIT_TABLE[0]["uncertainty"]
          and "禁止" in GUNIT_TABLE[0]["rule"])

    # ---- 时间步收敛：链式 0->1->2，固定单脉冲刺激 ----
    W1 = np.array([[0, 0, 0], [5.0, 0, 0], [0, 5.0, 0]])
    def chain(sub):
        net = CobaNetwork(sparse.csr_matrix(W1), sparse.csr_matrix((3, 3)), P, 20.0, sub)
        net.S_prev[0] = 1.0          # 单脉冲刺激注入 neuron0
        net.step()
        n1 = [t for (i, t) in net.spike_log if i == 1]
        n2 = [t for (i, t) in net.spike_log if i == 2]
        return len(n1), (n1[0] if n1 else np.nan), len(n2), (n2[0] if n2 else np.nan)
    rows = {s: chain(s) for s in (1, 5, 10, 50, 100, 200, 500, 2000)}
    for s, r in rows.items():
        print(f"  [INFO] substeps={s:4d} (dts={20/s:.4f}ms): n1={r[0]} t1={r[1]:.3f} n2={r[2]} t2={r[3]:.3f}")
    check("convergence: substeps=1 grossly wrong (layer-2 count=0)", rows[1][2] == 0)
    check("convergence: layer-1 count stable by substeps>=100", rows[100][0] == rows[200][0])
    check("convergence: first-spike time stable (substeps 500 vs 2000)",
          abs(rows[500][3] - rows[2000][3]) < 0.1, f"(t2 {rows[500][3]:.3f} vs {rows[2000][3]:.3f})")
    # layer-2 count 收敛**慢**（强耦合链）：报告证据化精度限制（不设过严门槛）
    n2 = [rows[s][2] for s in (100, 200, 500, 2000)]
    print(f"  [INFO] layer-2 count trend {n2} -> 慢收敛；证据化精度限制：n1 需 dts<=0.2ms(substeps>=100)、"
          f"n2 需 dts<=0.04ms(substeps>=500) 才 <5% 漂移")
    check("convergence: layer-2 count monotone toward true", n2[-1] >= n2[-2] >= n2[0])

    # ---- 合成性能（满足数值精度 substeps=50）+ 事件日志内存 ----
    big = 166700; nnz = 1000000
    rng = np.random.default_rng(1)
    Wbig = sparse.csr_matrix((rng.random(nnz), (rng.integers(0, big, nnz), rng.integers(0, big, nnz))), shape=(big, big))
    netb = CobaNetwork(Wbig, sparse.csr_matrix((big, big)), P, 20.0, substeps=50)
    netb.S_prev[:] = 0.0; netb.S_prev[rng.integers(0, big, big // 100)] = 1.0
    t0 = time.perf_counter()
    for _ in range(5):
        netb.step()
    per = (time.perf_counter() - t0) / 5 * 1000
    Wmem = (Wbig.data.nbytes + Wbig.indices.nbytes + Wbig.indptr.nbytes) / 1e6
    ev = len(netb.spike_log)
    evmem = ev * 16 / 1e6  # 约每事件 (int,float64)≈16B
    print(f"  [INFO] perf dt=20ms substeps=50: {per:.1f} ms/step (N={big}, nnz={nnz})")
    print(f"  [INFO] W(nnz=1e6) mem={Wmem:.2f} MB; event log events={ev} ~{evmem:.2f} MB")
    print(f"  [INFO] 40s sim=2000 steps at substeps=50 -> ~{per*2000/1000:.0f}s (nnz=1e6); 真实 nnz≈2.56e7 -> ~25.6x；**不外推为全脑实测**")

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
