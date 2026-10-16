"""P9-1a 回归：**COBA-LIF 数值原型**的合成小网络测试（静息/单兴奋/单抑制/分流/时间叠加/衰减/
左右镜像/dt 收敛/子步等价/无负电导·NaN/复位/可复现）。**只评数值实现正确性**；不跑真实全脑、不接车。"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy import sparse

from loop.coba_proto import CobaNetwork, CobaParams, mirror_network

RESULTS = []
P = CobaParams(g_unit_nS=10.0)          # 合成单位尺度（非实测定标），取 10 nS 使效应清晰


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def run(params, We, Wi=None, spikes=None, T=60.0, dt=1.0, substeps=1, V0=None):
    """返回 (net, peak_V)：peak_V = 每步 V 的最大值（**瞬态可观测**，非衰减后的尾值）。"""
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
    print("=========== coba_proto ===========")

    # 1) 静息回归：无输入 -> V -> E_l
    n = 3
    net = CobaNetwork(sparse.csr_matrix((n, n)), sparse.csr_matrix((n, n)), P, 0.1)
    net.reset(V0_mV=-40.0)
    for _ in range(2000):
        net.step()
    check("resting regression V->E_l", np.allclose(net.V, P.E_l_mV, atol=0.2),
          f"(V={net.V[0]:.2f}, E_l={P.E_l_mV})")

    # 2) 单个兴奋突触
    We = np.zeros((2, 2)); We[1, 0] = 1.0
    net_e, pk_e = run(P, We, spikes=[0], T=60, dt=0.5)
    check("single excitatory synapse depolarizes", pk_e[1] > P.E_l_mV + 1.0, f"(peak={pk_e[1]:.2f})")
    check("excitatory g decays to ~0", abs(net_e.g_exc[1]) < 1e-9, f"(g={net_e.g_exc[1]:.2e})")

    # 3) 单个抑制突触
    Wi = np.zeros((2, 2)); Wi[1, 0] = 1.0
    net_i, pk_i = run(P, np.zeros((2, 2)), Wi=Wi, spikes=[0], T=60, dt=0.5)
    check("single inhibitory synapse hyperpolarizes", net_i.V[1] < P.E_l_mV - 1.0 or pk_i[1] < P.E_l_mV - 1.0, f"(V1={net_i.V[1]:.2f})")

    # 4) 分流抑制：E_inh=E_l（纯分流），比较 exc-only vs exc+inh
    ps = CobaParams(E_inh_mV=P.E_l_mV, g_unit_nS=10.0)
    We3 = np.zeros((3, 3)); We3[2, 0] = 1.0
    Wi3 = np.zeros((3, 3)); Wi3[2, 1] = 1.0
    peak_e = run(ps, We3, spikes=[0], T=20, dt=0.5, V0=-60.0)[1][2]
    peak_s = run(ps, We3, Wi=Wi3, spikes=[0, 1], T=20, dt=0.5, V0=-60.0)[1][2]
    check("shunting inhibition reduces depolarization", peak_s < peak_e - 1.0,
          f"(exc={peak_e:.2f}, shunted={peak_s:.2f})")

    # 5) 多脉冲时间叠加（间隔 1 ms < tau_syn_e=2 ms）
    net_t = CobaNetwork(sparse.csr_matrix(We), sparse.csr_matrix((2, 2)), P, 0.5)
    gmax = 0.0
    for k in range(40):
        if k % 2 == 0:
            net_t.S_prev[0] = 1.0
        net_t.step()
        gmax = max(gmax, net_t.g_exc[1])
    check("temporal summation accumulates g", gmax > 1.5 * P.g_unit_nS, f"(gmax={gmax:.2f})")

    # 6) 撤除后电导指数衰减（spike 注入于第 1 步，随后 (T-dt) 衰减）
    T, dt = 1.5, 0.5
    g = run(P, We, spikes=[0], T=T, dt=dt)[0].g_exc[1]
    expect = P.g_unit_nS * np.exp(-(T - dt) / P.tau_syn_e_ms)
    check("conductance decays exponentially after removal", abs(g - expect) < 1e-6,
          f"(g={g:.4f}, expect={expect:.4f})")

    # 7) 左右镜像对称
    m = 4
    Wm = sparse.random(m, m, density=0.4, random_state=0, format="csr")
    side = np.array(["L", "R", "L", "R"])
    Wmir, _ = mirror_network(Wm, side)
    net_a = CobaNetwork(Wm, sparse.csr_matrix((m, m)), P, 0.5); net_a.S_prev[0] = 1.0
    net_b = CobaNetwork(Wmir, sparse.csr_matrix((m, m)), P, 0.5); net_b.S_prev[1] = 1.0
    va, vb = [], []
    for _ in range(20):
        net_a.step(); net_b.step(); va.append(net_a.V.copy()); vb.append(net_b.V.copy())
    check("left/right mirror symmetry", np.allclose(np.array(va)[:, 1], np.array(vb)[:, 0], atol=1e-9))

    # 8) dt 收敛
    T = 4.0   # 固定时刻（各 dt 的公共倍数）；**亚阈**参数避免发放复位造成的 V 跳变
    Psub = CobaParams(g_unit_nS=10.0, V_th_mV=-30.0)
    vals = [run(Psub, We, spikes=[0], T=T, dt=d)[0].V[1] for d in (2.0, 1.0, 0.5, 0.25, 0.1)]
    diffs = [abs(vals[i] - vals[i + 1]) for i in range(len(vals) - 1)]
    check("dt convergence (monotone, ratio>5x)", all(diffs[i+1] < diffs[i] for i in range(len(diffs)-1)) and diffs[-1] < diffs[0] * 0.2,
          f"(diffs={[round(d,4) for d in diffs]})")
    v_sub = run(Psub, We, spikes=[0], T=T, dt=4.0, substeps=40)[0].V[1]
    check("substeps match fine dt", abs(v_sub - vals[-1]) < 0.01, f"(sub={v_sub:.3f}, fine={vals[-1]:.3f})")

    # 9) 无负电导 / 无 NaN
    check("no negative conductance", (net.g_exc >= 0).all() and (net.g_inh >= 0).all())
    check("no NaN in V", np.isfinite(net.V).all())

    # 10) 复位
    pr = CobaParams(g_unit_nS=200.0)
    netr, _ = run(pr, We, spikes=[0], T=20, dt=0.5)
    check("reset to V_reset after spike", netr.V[1] <= pr.V_th_mV)

    # 11) 可复现
    check("reproducible", run(P, We, spikes=[0], T=10, dt=0.5)[1][1] == run(P, We, spikes=[0], T=10, dt=0.5)[1][1])

    # 12) 只有 spike 才产电导
    net0, _ = run(P, We, T=20, dt=0.5)
    check("no spike -> no synaptic conductance", net0.g_exc.max() == 0.0)

    # 性能实测（合成大网络，稀疏 1e6 边）
    big = 166700; nnz = 1000000
    rng = np.random.default_rng(1)
    Wbig = sparse.csr_matrix((rng.random(nnz), (rng.integers(0, big, nnz), rng.integers(0, big, nnz))),
                             shape=(big, big))
    netbig = CobaNetwork(Wbig, sparse.csr_matrix((big, big)), P, 0.1)
    t0 = time.perf_counter()
    for _ in range(20):
        netbig.step()
    per_step = (time.perf_counter() - t0) / 20 * 1000
    mem = (netbig.V.nbytes + netbig.g_exc.nbytes + netbig.g_inh.nbytes + netbig.S_prev.nbytes) / 1e6
    print(f"  [INFO] perf: {per_step:.1f} ms/step (N={big}, nnz={Wbig.nnz}); state mem={mem:.2f} MB")

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
