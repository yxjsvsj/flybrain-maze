"""P9-1a-b 回归：COBA 原型的**脉冲时间精度**与数值/性能基准（合成小网络；不跑真实全脑、不接车）。"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy import sparse

from loop.coba_proto import CobaNetwork, CobaParams, mirror_network

RESULTS = []
P = CobaParams(g_unit_nS=10.0)


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def run(params, We, Wi=None, spikes=None, T=60.0, dt=1.0, substeps=1, V0=None):
    n = We.shape[0]
    Wi = sparse.csr_matrix((n, n)) if Wi is None else sparse.csr_matrix(Wi)
    net = CobaNetwork(sparse.csr_matrix(We), Wi, params, dt, substeps, V0)
    if spikes:
        for i in spikes:
            net.S_prev[i] = 1.0
    for _ in range(int(round(T / dt))):
        net.step()
    return net, net.V_peak


def main() -> int:
    print("=========== coba_proto (P9-1a-b) ===========")

    # 1) 静息回归（V0<E_l，不触发发放）
    net = CobaNetwork(sparse.csr_matrix((3, 3)), sparse.csr_matrix((3, 3)), P, 0.1)
    net.reset(V0_mV=-70.0)
    for _ in range(2000):
        net.step()
    check("resting regression V->E_l", np.allclose(net.V, P.E_l_mV, atol=0.2), f"(V={net.V[0]:.2f})")
    check("no input -> no spikes", net.spikes.sum() == 0 and len(net.spike_log) == 0)

    # 2) 单个兴奋/抑制突触
    We = np.zeros((2, 2)); We[1, 0] = 1.0
    net_e, pk_e = run(P, We, spikes=[0], T=60, dt=0.5)
    check("single excitatory synapse depolarizes", pk_e[1] > P.E_l_mV + 1.0, f"(peak={pk_e[1]:.2f})")
    check("excitatory g decays to ~0", abs(net_e.g_exc[1]) < 1e-9)
    Wi = np.zeros((2, 2)); Wi[1, 0] = 1.0
    net_i, _ = run(P, np.zeros((2, 2)), Wi=Wi, spikes=[0], T=60, dt=0.5)
    check("single inhibitory synapse hyperpolarizes", net_i.V_min[1] < P.E_l_mV - 1.0, f"(Vmin={net_i.V_min[1]:.2f})")

    # 3) 分流抑制
    ps = CobaParams(E_inh_mV=P.E_l_mV, g_unit_nS=10.0)
    We3 = np.zeros((3, 3)); We3[2, 0] = 1.0
    Wi3 = np.zeros((3, 3)); Wi3[2, 1] = 1.0
    pe = run(ps, We3, spikes=[0], T=20, dt=0.5, V0=-60.0)[1][2]
    pn = run(ps, We3, Wi=Wi3, spikes=[0, 1], T=20, dt=0.5, V0=-60.0)[1][2]
    check("shunting inhibition reduces depolarization", pn < pe - 1.0, f"(exc={pe:.2f}, shunted={pn:.2f})")

    # 4) 时间叠加
    net_t = CobaNetwork(sparse.csr_matrix(We), sparse.csr_matrix((2, 2)), P, 0.5)
    gmax = 0.0
    for k in range(40):
        if k % 2 == 0:
            net_t.S_prev[0] = 1.0
        net_t.step(); gmax = max(gmax, net_t.g_exc[1])
    check("temporal summation accumulates g", gmax > 1.5 * P.g_unit_nS, f"(gmax={gmax:.2f})")

    # 5) 撤除后指数衰减（仅事件传递）
    T, dt = 1.5, 0.5
    g = run(P, We, spikes=[0], T=T, dt=dt)[0].g_exc[1]
    check("conductance decays exponentially after removal", abs(g - P.g_unit_nS * np.exp(-(T - dt) / P.tau_syn_e_ms)) < 1e-9, f"(g={g:.4f})")

    # 6) 左右镜像
    m = 4
    Wm = sparse.random(m, m, density=0.4, random_state=0, format="csr")
    Wmir, _ = mirror_network(Wm, np.array(["L", "R", "L", "R"]))
    na = CobaNetwork(Wm, sparse.csr_matrix((m, m)), P, 0.5); na.S_prev[0] = 1.0
    nb = CobaNetwork(Wmir, sparse.csr_matrix((m, m)), P, 0.5); nb.S_prev[1] = 1.0
    va, vb = [], []
    for _ in range(20):
        na.step(); nb.step(); va.append(na.V.copy()); vb.append(nb.V.copy())
    check("left/right mirror symmetry", np.allclose(np.array(va)[:, 1], np.array(vb)[:, 0], atol=1e-9))

    # 7) 亚阈 dt 收敛（固定时刻 V）
    Psub = CobaParams(g_unit_nS=10.0, V_th_mV=-30.0)
    T2 = 4.0
    vals = [run(Psub, We, spikes=[0], T=T2, dt=d)[0].V[1] for d in (2.0, 1.0, 0.5, 0.25, 0.1)]
    diffs = [abs(vals[i] - vals[i + 1]) for i in range(len(vals) - 1)]
    check("subthreshold dt convergence (monotone, ratio>5x)",
          all(diffs[i + 1] < diffs[i] for i in range(len(diffs) - 1)) and diffs[-1] < diffs[0] * 0.2,
          f"(diffs={[round(d,4) for d in diffs]})")
    v_sub = run(Psub, We, spikes=[0], T=T2, dt=4.0, substeps=40)[0].V[1]
    check("substeps match fine dt", abs(v_sub - vals[-1]) < 0.01, f"(sub={v_sub:.3f}, fine={vals[-1]:.3f})")

    # 8) **确定性反例**：20ms 内多次发放，substeps=1 会**漏记**（只看末值/末布尔）
    #    （经**脉冲事件**注入：We[1,0]=5 -> g_unit*5=50 nS；S 在子步 0 施加）
    Wc = np.zeros((2, 2)); Wc[1, 0] = 5.0
    def counter(steps):
        n1 = CobaNetwork(sparse.csr_matrix(Wc), sparse.csr_matrix((2, 2)), P, 20.0, steps)
        n1.S_prev[0] = 1.0
        cnt = n1.step()
        return int(round(cnt[1])), list(n1.spike_log)
    c1, log1 = counter(1)
    cf, logf = counter(200)
    check("counterexample: substeps=1 MISSES spikes (undercount)",
          c1 == 1 and cf >= 3, f"(substeps=1 -> {c1}; fine -> {cf})")
    check("counterexample: fine records early spike times",
          len(logf) > 0 and logf[0][1] < 2.0 and len(log1) > 0 and log1[0][1] == 20.0,
          f"(fine_t0={logf[0][1]:.2f}ms, coarse_t={log1[0][1]:.0f}ms)")

    # 8b) 超阈值收敛：固定外层 dt=20ms，**子步细化**（脉冲时刻分辨率 = dts）
    first_t = []
    for sub in (200, 2000, 20000):
        n1 = CobaNetwork(sparse.csr_matrix(Wc), sparse.csr_matrix((2, 2)), P, 20.0, sub)
        n1.S_prev[0] = 1.0
        n1.step()
        first_t.append(n1.spike_log[0][1])
    fd = [abs(first_t[i] - first_t[i + 1]) for i in range(len(first_t) - 1)]
    check("over-threshold convergence (first-spike time by substep refinement)", all(d < 0.05 for d in fd),
          f"(first_t={[round(t,4) for t in first_t]})")

    # 9) 复位正确性：单次发放（substeps=1）后 V == V_reset（**不是**用末值<阈代替）
    Wr = np.zeros((2, 2)); Wr[1, 0] = 50.0     # 强驱动 -> 0.5ms 内触发一次
    n1 = CobaNetwork(sparse.csr_matrix(Wr), sparse.csr_matrix((2, 2)), P, 0.5, 1)
    n1.S_prev[0] = 1.0
    n1.step()
    check("reset: V == V_reset after a single spike", n1.V[1] == P.V_reset_mV and n1.spikes[1] == 1,
          f"(V={n1.V[1]}, spikes={n1.spikes[1]:.0f})")

    # 10) 重复发放
    n1 = CobaNetwork(sparse.csr_matrix(Wc), sparse.csr_matrix((2, 2)), P, 20.0, 200)
    n1.S_prev[0] = 1.0
    check("repeated firing within one step", n1.step()[1] >= 3)

    # 11) 负电导输入拒绝
    try:
        CobaNetwork(sparse.csr_matrix((1, 1)), sparse.csr_matrix((1, 1)),
                    CobaParams(g_unit_nS=-1.0), 1.0)
        check("reject negative g_unit", False)
    except ValueError:
        check("reject negative g_unit", True)
    try:
        CobaNetwork(sparse.csr_matrix(np.array([[-1.0]])), sparse.csr_matrix((1, 1)), P, 1.0)
        check("reject negative weight in We", False)
    except ValueError:
        check("reject negative weight in We", True)

    # 12) 无负电导 / 无 NaN
    check("no negative conductance", (net.g_exc >= 0).all() and (net.g_inh >= 0).all())
    check("no NaN in V", np.isfinite(net.V).all())
    check("reproducible", run(P, We, spikes=[0], T=10, dt=0.5)[1][1] == run(P, We, spikes=[0], T=10, dt=0.5)[1][1])

    # 13) 性能实测：区分 **无脉冲** 与 **有非零脉冲**（dt=20ms, substeps=1）
    big = 166700; nnz = 1000000
    rng = np.random.default_rng(1)
    Wbig = sparse.csr_matrix((rng.random(nnz), (rng.integers(0, big, nnz), rng.integers(0, big, nnz))), shape=(big, big))
    netbig = CobaNetwork(Wbig, sparse.csr_matrix((big, big)), P, 20.0, substeps=1)
    netbig.S_prev[:] = 0.0
    t0 = time.perf_counter()
    for _ in range(10):
        netbig.S_prev[:] = 0.0; netbig.step()
    idle = (time.perf_counter() - t0) / 10 * 1000
    t0 = time.perf_counter()
    for _ in range(10):
        netbig.S_prev[:] = 0.0; netbig.S_prev[rng.integers(0, big, big // 100)] = 1.0  # 1% 发放
        netbig.step()
    active = (time.perf_counter() - t0) / 10 * 1000
    state = (netbig.V.nbytes + netbig.g_exc.nbytes + netbig.g_inh.nbytes + netbig.S_prev.nbytes) / 1e6
    Wmem = (Wbig.data.nbytes + Wbig.indices.nbytes + Wbig.indptr.nbytes) / 1e6
    print(f"  [INFO] perf dt=20ms substeps=1: idle={idle:.1f} ms/step, active(1% spikes)={active:.1f} ms/step")
    print(f"  [INFO] state mem={state:.2f} MB; W(nnz=1e6) mem={Wmem:.2f} MB")
    print(f"  [INFO] 40s sim=2000 步 -> idle≈{idle*2000/1000:.0f}s, active≈{active*2000/1000:.0f}s (nnz=1e6)")
    print(f"  [INFO] 真实连接组 nnz≈2.56e7 (25.6x) -> 成本随 nnz 线性增长，不能以 1e6 基准当作全脑速度")

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
