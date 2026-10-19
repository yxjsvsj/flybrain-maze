"""P10-C1b：真实 MaleCNS 子网 **LIF 前向性能** + 资源预算 + ΔW 数学安全 + 独立进程隔离。

**不训练、不更新 ΔW、不跑虚拟迷宫、不控车。** ΔW 恒为 0。
LIF = 现有 `flybrain.FlyBrain`（归一化 LIF，非 P9 COBA）；真实子网权重。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy
import scipy.sparse as sp

from loop.p100_evidence import audit
from loop.p100b_path_audit import chain_layers
from loop.p10_test_decoder import build_freeze
from loop.visual_motor_audit import load_data

HERE = os.path.dirname(os.path.abspath(__file__))
N_NEURONS_EXPECTED = 23218
NNZ_EXPECTED = 3549643


def _peak_memory_mb():
    """Windows 峰值工作集（ctypes）；失败则返回 None。"""
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(c)
        f = ctypes.windll.psapi.GetProcessMemoryInfo
        f.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(PMC), ctypes.wintypes.DWORD]
        f.restype = ctypes.wintypes.BOOL
        ok = f(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return round(c.PeakWorkingSetSize / 1e6, 2) if ok else None
    except Exception:                            # noqa: BLE001
        return None


def build_subnet(D=None):
    D = D or load_data()
    A = audit(D)
    L = chain_layers(D, A)
    union = np.array(sorted(set(np.concatenate([L[0], L[1], L[2], L[3], L[4], L[5], A["DAN"]]).tolist())),
                     dtype=np.int64)
    W = D["W"].tocsr().copy(); W.eliminate_zeros()
    Wsub = W[union][:, union].tocsr()
    return {"D": D, "A": A, "layers": L, "union": union, "Wsub": Wsub}


def verify_subnet(sub):
    D, A, L, union, Wsub = sub["D"], sub["A"], sub["layers"], sub["union"], sub["Wsub"]
    W = D["W"].tocsr().copy(); W.eliminate_zeros()
    ids = D["ids"]
    idmap = {int(b): i for i, b in enumerate(ids)}
    # 方向：KC 137302 -> MBON 519128 => W[post,pre] > 0
    direction = None
    if 137302 in idmap and 519128 in idmap:
        direction = {"kc_pre": 137302, "mbon_post": 519128,
                     "W_post_pre": float(W[idmap[519128], idmap[137302]]),
                     "W_pre_post": float(W[idmap[137302], idmap[519128]]),
                     "correct": float(W[idmap[519128], idmap[137302]]) > 0}
    n, nnz = int(Wsub.shape[0]), int(Wsub.nnz)
    mem = nnz * 4 + nnz * 4 + (n + 1) * 4
    # 截断输入损失
    incoming_total = int(W[union].nnz)
    incoming_internal = int(W[union][:, union].nnz)
    lost_edges = incoming_total - incoming_internal
    lost_weight = float(W[union].sum() - W[union][:, union].sum())
    sensory = np.char.find(D["superclass"].astype(str), "sensory") >= 0
    return {
        "n_neurons": n, "nnz": nnz, "csr_dtype": str(Wsub.dtype), "csr_format": Wsub.format,
        "csr_float32_MB": round(mem / 1e6, 2),
        "bodyId_unique": bool(len(np.unique(ids[union])) == n),
        "direction_check": direction,
        "n_DN_preserved": int(np.intersect1d(union, A["DN"]).size), "n_DN_total": int(len(A["DN"])),
        "n_DN_outside_subnet": int(len(A["DN"]) - np.intersect1d(union, A["DN"]).size),
        "n_sensory_in_subnet": int(np.intersect1d(union, np.flatnonzero(sensory)).size),
        "truncation": {"incoming_edges_total": incoming_total, "incoming_edges_internal": incoming_internal,
                       "lost_incoming_edges": lost_edges, "lost_incoming_weight": round(lost_weight, 4),
                       "note": "子网外输入被截断；不得宣称为全脑功能"},
        "layer_sizes": [int(len(x)) for x in L], "n_DAN": int(len(A["DAN"])),
        "note": "六层存在节点重合；union 去重后为 unique 神经元数，不能直接相加。",
    }


def run_lif_benchmark(repeats=2, warmup=5, batches=(10, 50, 100), timeout_s=120.0, mem_cap_mb=2048.0):
    from flybrain import FlyBrain
    sub = build_subnet()
    ver = verify_subnet(sub)
    if ver["n_neurons"] != N_NEURONS_EXPECTED or ver["nnz"] != NNZ_EXPECTED:
        return {"ok": False, "error": "subnet size/nnz mismatch", "verification": ver}
    D = sub["D"]; union = sub["union"]; Wsub = sub["Wsub"]
    td = tempfile.mkdtemp(prefix="p10c1b_sub_")
    try:
        np.savez(os.path.join(td, "brain.npz"),
                 ids=D["ids"][union], visual=np.zeros(0, np.int32), azimuth=np.zeros(0, np.float32),
                 cell_type=D["cell_type"][union], side=D["side"][union],
                 positions=(D["positions"][union] if D.get("positions") is not None
                            else np.zeros((len(union), 3), np.float32)),
                 superclass=D["superclass"][union])
        sp.save_npz(os.path.join(td, "weights.npz"), Wsub)
        runs = []
        t0_all = time.perf_counter()
        for rep in range(repeats):
            b = FlyBrain(data=td, device="cpu", batch=1, seed=100 + rep, dt=0.020)
            t_warm0 = time.perf_counter()
            for _ in range(warmup):
                b.step()
            warm_s = time.perf_counter() - t_warm0
            batches_out = {}
            abnormal = []
            for nb in batches:
                ts, fired_counts = [], []
                for _ in range(nb):
                    t = time.perf_counter()
                    fired = b.step()
                    ts.append(time.perf_counter() - t)
                    fired_counts.append(int(len(fired)))
                    if time.perf_counter() - t0_all > timeout_s:
                        abnormal.append("timeout")
                        break
                    if not np.isfinite(b.v).all():
                        abnormal.append("non-finite-v")
                batches_out[str(nb)] = {
                    "median_ms": round(float(np.median(ts)) * 1000, 4),
                    "p95_ms": round(float(np.percentile(ts, 95)) * 1000, 4),
                    "fired_mean": round(float(np.mean(fired_counts)), 2) if fired_counts else None,
                    "fired_max": int(max(fired_counts)) if fired_counts else None,
                    "abnormal_firing": bool(max(fired_counts) > 0.5 * b.n) if fired_counts else False,
                }
                if time.perf_counter() - t0_all > timeout_s:
                    break
            runs.append({"repeat": rep, "warmup_s": round(warm_s, 4),
                         "batches": batches_out, "abnormal": abnormal,
                         "peak_mem_MB": _peak_memory_mb()})
            if time.perf_counter() - t0_all > timeout_s:
                break
        med = [r["batches"].get("100", r["batches"].get("50", {})).get("median_ms") for r in runs]
        med = [m for m in med if m is not None]
        repeatable = (len(med) >= 2 and abs(med[0] - med[1]) <= max(0.05, 0.2 * max(med)))
        peak = max([r.get("peak_mem_MB") or 0 for r in runs] or [0])
        return {"ok": True, "verification": ver, "runs": runs, "repeatable": bool(repeatable),
                "peak_mem_MB": peak, "mem_cap_exceeded": peak > mem_cap_mb,
                "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "scipy": scipy.__version__,
                             "flybrain": getattr(__import__("flybrain"), "__version__", "?")},
                "device": "cpu", "lif_params": {"dt": 0.020, "tau": 0.100, "gain": 3.0,
                                                "noise_hz": 1.2, "noise_amp": 0.22,
                                                "refractory": 0.0, "dW": 0},
                "note": "真实子网 LIF 前向；仅短时性能测量，无学习/无闭环。"}
    finally:
        shutil.rmtree(td, ignore_errors=True)


def resource_budget(per_step_ms, max_steps_per_trial=2000, budget_hours=24.0, peak_mem_mb=None):
    trials_70 = 7 * 10
    trials_1890 = 27 * trials_70
    def hours(n):
        return round(n * max_steps_per_trial * (per_step_ms / 1000.0) / 3600.0, 3)
    est70, est1890 = hours(trials_70), hours(trials_1890)
    mem_ok = peak_mem_mb is None or peak_mem_mb <= 2048.0
    ok = (est1890 <= budget_hours) and mem_ok
    return {"per_step_ms": per_step_ms, "max_steps_per_trial": max_steps_per_trial,
            "trials_70": trials_70, "trials_1890": trials_1890,
            "grid_clarification": ("27 参数点 × 70 trial = 1,890 次试验（若逐条件执行）；"
                                   "预注册未选择子集，**不得**事后缩减网格并声称原预注册通过。"),
            "est_hours_70": est70, "est_hours_1890": est1890,
            "peak_mem_MB": peak_mem_mb, "budget_hours": budget_hours,
            "resource_ready": bool(ok),
            "reason": None if ok else "预算/内存超限或缺失"}


def dw_spec_check():
    """合成稀疏矩阵：验证 clip 是否充分、L1、新边、未批准连接不变。"""
    W = sp.csr_matrix(np.array([[0.0, 0.0, 0.0],
                                [0.5, 0.0, 0.0],     # W[1,0]=0.5 (KC->MBON excitatory)
                                [0.0, 0.3, 0.0]], dtype=np.float32))
    approved = {(1, 0), (2, 1)}
    # 1) 大负 dW：clip(±0.6) 后仍可能 W_eff<0
    dW = sp.csr_matrix(([-0.6], ([1], [0])), shape=(3, 3), dtype=np.float32)
    clipped = dW.copy(); clipped.data = np.clip(clipped.data, -0.6, 0.6)
    Weff = (W + clipped).toarray()
    neg_possible = bool(Weff[1, 0] < 0)
    # 2) 提议修订（待审批，不实现）：W_eff>=0
    Weff_fix = np.maximum(0.0, Weff)
    fixed_ok = bool(Weff_fix[1, 0] >= 0)
    # 3) L1 上界
    W_l1 = np.array([[0.0, 0.0, 0.0], [0.7, 0.3, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    l1_before = float(np.abs(W_l1[1]).sum())
    W_l1[1, 0] += 0.4
    l1_after = float(np.abs(W_l1[1]).sum())
    l1_break = l1_after > 1.0
    # 4) 新边 / 未批准不变：dW 只写 approved mask
    Walone = W.toarray()
    # 未加掩码：连非边 (0,2) 也写 -> 产生新边
    dW_unmasked = np.zeros((3, 3), np.float32); dW_unmasked[1, 0] = 0.2; dW_unmasked[0, 2] = 0.2
    Weff_unmasked = Walone + dW_unmasked
    new_edges = int(((Weff_unmasked != 0) & (Walone == 0)).sum())
    # 正确加掩码：只写 approved 边 -> 未批准连接逐位不变
    dW_masked = np.zeros((3, 3), np.float32); dW_masked[1, 0] = 0.2
    Weff_masked = Walone + dW_masked
    unapproved_unchanged = bool(all(Weff_masked[i, j] == Walone[i, j]
                                    for i in range(3) for j in range(3) if (i, j) not in approved))
    return {
        "current_spec_sufficient": False,
        "issues": [
            "负 ΔW 在 |ΔW|>W 时可令兴奋性 W_eff<0；clip(ΔW,±Δmax) 单独**不充分**",
            "加 ΔW 可破坏逐突触后 L1 上界（无重归一化时）",
            "ΔW 必须只写官方 KC→MBON 边掩码，否则可能产生新边",
        ],
        "synthetic": {"negative_dW_makes_Weff_negative": neg_possible,
                      "proposed_Weff_floor_fixes": fixed_ok,
                      "l1_before": l1_before, "l1_after": l1_after, "l1_break": l1_break,
                      "new_edges_if_unmasked": new_edges,
                      "unapproved_unchanged_with_mask": unapproved_unchanged},
        "proposed_erratum": {
            "status": "PENDING_APPROVAL; NOT implemented",
            "add_bound": "for excitatory edges: ΔW >= -W (equivalently W_eff >= 0); symmetric for inhibitory",
            "l1": "either document that L1 is not preserved, or add a per-post constraint -- but that changes unapproved synapses and needs approval",
            "topology": "apply ΔW ONLY on the official KC->MBON edge mask; no new edges; no transmitter-sign change",
            "note": "不得悄悄修改 P10-C1a 冻结规格；不得自动实现更新。",
        },
    }


_CHILD = r'''
import json, sys
ALLOWED = {"tof","tof_delta","neural_state","motor_feedback"}
FORBIDDEN = {"map","occupancy","pose","global_pose","dijkstra","teacher_action","teacher_label",
             "goal","goal_coord","goal_direction","nav_memory","planner","path_plan",
             "turn_command","manual_turn"}
try:
    payload = json.loads(sys.argv[1])
except Exception:
    print("BAD_JSON"); sys.exit(5)
bad = set(payload) & FORBIDDEN
if bad:
    print("LEAK:" + ",".join(sorted(bad))); sys.exit(2)
unknown = set(payload) - ALLOWED
if unknown:
    print("UNKNOWN:" + ",".join(sorted(unknown))); sys.exit(3)
for m in ("dijkstra", "planner", "networkx", "navigation"):
    if m in sys.modules:
        print("MODULE_LEAK:" + m); sys.exit(4)
print("OK"); sys.exit(0)
'''


def run_isolation_process(payload: dict):
    td = tempfile.mkdtemp(prefix="p10c1b_iso_")
    try:
        p = os.path.join(td, "student_child.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(_CHILD)
        r = subprocess.run([sys.executable, p, json.dumps(payload)], capture_output=True, text=True, timeout=60)
        return {"returncode": r.returncode, "stdout": r.stdout.strip()[-200:], "stderr": r.stderr.strip()[-200:]}
    finally:
        shutil.rmtree(td, ignore_errors=True)


def isolation_suite():
    pos = run_isolation_process({"tof": [1, 2, 3, 4, 5], "tof_delta": [0, 0, 0, 0, 0],
                                 "neural_state": [0.1], "motor_feedback": [0.0, 0.0]})
    neg = {}
    for k in ("map", "pose", "dijkstra", "teacher_action", "teacher_label", "goal", "nav_memory", "planner", "turn_command"):
        neg[k] = run_isolation_process({k: 1})
    ok_pos = pos["returncode"] == 0 and pos["stdout"] == "OK"
    ok_neg = all(v["returncode"] == 2 for v in neg.values())
    return {"mock_process_isolation": bool(ok_pos and ok_neg), "positive": pos, "negative": neg,
            "real_student_isolation": False,
            "note": ("真实子进程边界已验证（mock 学生）；未来真实学生程序的执行路径**尚未**验证，"
                     "故 real_student_isolation=False。")}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    D = load_data()
    freeze = build_freeze(D)
    bench = run_lif_benchmark()
    per_step = None
    if bench.get("ok"):
        b100 = bench["runs"][0]["batches"].get("100") or bench["runs"][0]["batches"].get("50")
        per_step = b100["median_ms"]
    budget = resource_budget(per_step or 0.0, peak_mem_mb=bench.get("peak_mem_MB"))
    dw = dw_spec_check()
    iso = isolation_suite()
    report = {
        "decoder_freeze": freeze, "benchmark": bench, "resource_budget": budget,
        "dw_spec_safety": dw, "isolation": iso,
        "verdicts": {
            "Biological": "NO-GO",
            "Computational_Hypothesis": "GO (conditional, proposal only)",
            "Decoder_Readiness": "NO-GO" if not freeze["decoder_ready"] else "GO",
            "Resource_Readiness": "GO" if budget["resource_ready"] else "NO-GO",
            "Student_Isolation_Readiness": ("mock-only; real student NOT verified"
                                            if not iso["real_student_isolation"] else "GO"),
            "Overall_Execution_Readiness": "NO-GO",
            "Training_Authorization": False,
        },
        "training_authorized": False,
    }
    json.dump(report, open(os.path.join(HERE, "p10c1b_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(freeze, open(os.path.join(HERE, "p10c1b_decoder_freeze.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[P10-C1b] subnet:", bench.get("verification", {}).get("n_neurons"),
          "nnz:", bench.get("verification", {}).get("nnz"),
          "| LIF 100-step median ms:", per_step)
    print("  repeatable:", bench.get("repeatable"), "| peak MB:", bench.get("peak_mem_MB"))
    print("  resource:", budget.get("est_hours_1890"), "h (1890 trials) | ready:", budget.get("resource_ready"))
    print("  dw current sufficient:", dw["current_spec_sufficient"], "| l1_break:", dw["synthetic"]["l1_break"])
    print("  isolation mock:", iso["mock_process_isolation"], "| real:", iso["real_student_isolation"])
    print("  decoder_ready:", freeze["decoder_ready"])
    print("wrote p10c1b_report.json / p10c1b_decoder_freeze.json")
    return 0 if bench.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
