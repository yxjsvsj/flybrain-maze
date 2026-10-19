"""P10-C1a：真实连接组计算学习实验的**执行前门控**（工程/静态/隔离测试）。

**不训练、不更新 ΔW、不跑真实神经动力学、不跑虚拟/实体小车。**
`execution_ready` 只有九项条件全部成立才为 True；`training_authorized` **恒为 False**。
"""
from __future__ import annotations

import json
import os
import sys

from loop.p100e_gate import history_verdicts, w_readonly_check

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 学生允许/禁止输入（严格）
ALLOWED_INPUTS = {"tof", "tof_delta", "neural_state", "motor_feedback"}
FORBIDDEN_INPUTS = {"map", "occupancy", "pose", "global_pose", "dijkstra", "teacher_action",
                    "teacher_label", "goal", "goal_coord", "goal_direction", "nav_memory",
                    "planner", "path_plan", "turn_command", "manual_turn"}
FORBIDDEN_MODULES = {"dijkstra", "networkx", "planner", "navigation", "map_loader"}
FORBIDDEN_FILE_TOKENS = ("map", "teacher", "dijkstra", "planner", "pose", "goal", "occ")


class LeakError(RuntimeError):
    pass


class StudentSandbox:
    """独立学生进程的最小 mock：只接受允许输入；禁止项/未知键/危险依赖/危险文件即失败。"""

    def __init__(self):
        self.log = []

    def step(self, inputs: dict):
        bad = set(inputs) & FORBIDDEN_INPUTS
        if bad:
            raise LeakError(f"forbidden inputs: {sorted(bad)}")
        unknown = set(inputs) - ALLOWED_INPUTS
        if unknown:
            raise LeakError(f"unknown inputs: {sorted(unknown)}")
        self.log.append({"inputs": sorted(inputs)})
        return {"ok": True, "v": 0.0, "omega": 0.0}

    def check_imports(self, modules):
        bad = set(modules) & FORBIDDEN_MODULES
        if bad:
            raise LeakError(f"forbidden modules: {sorted(bad)}")
        return True

    def check_file(self, path: str):
        low = str(path).lower()
        if any(t in low for t in FORBIDDEN_FILE_TOKENS):
            raise LeakError(f"forbidden file: {path}")
        return True


def isolation_selftest():
    s = StudentSandbox()
    results = {"positive": False, "negative": {}, "imports": {}, "files": {}}
    try:
        s.step({"tof": [1, 2, 3, 4, 5], "tof_delta": [0, 0, 0, 0, 0],
                "neural_state": [0.0], "motor_feedback": [0.0, 0.0]})
        results["positive"] = True
    except Exception as e:                        # noqa: BLE001
        results["positive"] = False
        results["positive_error"] = repr(e)[:120]
    for key in ("map", "pose", "dijkstra", "teacher_action", "teacher_label",
                "goal", "nav_memory", "planner", "turn_command"):
        try:
            s.step({key: 1})
            results["negative"][key] = "LEAK-NOT-CAUGHT"
        except LeakError:
            results["negative"][key] = "caught"
    for mod in ("dijkstra", "planner", "networkx"):
        try:
            s.check_imports([mod]); results["imports"][mod] = "LEAK-NOT-CAUGHT"
        except LeakError:
            results["imports"][mod] = "caught"
    for p in ("map.json", "teacher_actions.csv", "planner.npz"):
        try:
            s.check_file(p); results["files"][p] = "LEAK-NOT-CAUGHT"
        except LeakError:
            results["files"][p] = "caught"
    mock_ok = (results["positive"]
               and all(v == "caught" for v in results["negative"].values())
               and all(v == "caught" for v in results["imports"].values())
               and all(v == "caught" for v in results["files"].values()))
    results["mock_process_isolation"] = bool(mock_ok)
    results["real_student_isolation"] = False          # 未来真实学生执行路径尚未验证
    results["ok"] = results["mock_process_isolation"] and results["real_student_isolation"]
    results["note"] = ("mock 沙箱通过；**真实学生进程尚未验证**，故第 8 项仍为 False。"
                       "真实隔离需接口白名单+依赖检查+文件访问限制+运行时断言+日志。")
    return results


def audit_decoder(path=None):
    p = path or os.path.join(HERE, "decode.py")
    if not os.path.exists(p):
        return {"decoder_ready": False, "error": "decode.py missing"}
    text = open(p, encoding="utf-8").read()
    features = {k: (k in text) for k in
                ("memory_gain", "pursuit_turn", "scripted", "readout", "bias", "escape",
                 "stall_turn_mode", "stall_dir", "turn_sign")}
    # 测试期禁止的方向来源（P1 冻结的 decode.py 全部包含）
    risky = [k for k, v in features.items() if v and k in
             ("memory_gain", "pursuit_turn", "scripted", "readout", "bias", "escape",
              "stall_turn_mode", "stall_dir")]
    return {
        "decoder_ready": (len(risky) == 0),      # 当前 decode.py 含 risky -> False
        "features": features,
        "risky_direction_sources": risky,
        "reason": ("P1 冻结的 decode.py 在测试路径仍含外部方向来源：" + ", ".join(risky) +
                   "。测试期方向必须只来自冻结、有神经依据的 DN 输出映射；"
                   "安全层仅限速/急停。"),
        "proposed_interface": {
            "name": "P10TestDecoder (separate, NOT modifying loop/decode.py)",
            "inputs": ["frozen DN group firing rates"],
            "allowed_fixed_gains": ["prespecified, neurally justified, frozen before training"],
            "forbidden": ["memory_gain", "pursuit_turn", "scripted", "readout",
                          "bias", "escape direction", "stall spin", "manual turn"],
            "safety_layer": ["speed limit", "emergency stop (no direction)"],
        },
        "requires": [
            "独立 P10 测试解码接口（不改 P1 decode.py）",
            "固定增益的神经依据与冻结审计",
            "证明测试期无任何外部/人工方向来源",
            "安全层只限速/急停，不决定左右/旋转脱困",
        ],
    }


def check_freeze_spec(path=None):
    p = path or os.path.join(HERE, "p10c1a_freeze_spec.json")
    if not os.path.exists(p):
        return {"ok": False, "error": "freeze spec missing"}
    s = json.load(open(p, encoding="utf-8"))
    need = ["mechanism", "equation", "params", "eligibility", "composition", "bounds",
            "sign_topology", "l1_normalization", "sensitivity", "not_conductance"]
    missing = [k for k in need if not s.get(k)]
    params = s.get("params", {})
    miss_params = [k for k in ("eta", "tau_e", "delta_max", "D0") if k not in params]
    return {"ok": len(missing) == 0 and len(miss_params) == 0,
            "missing": missing, "missing_params": miss_params, "spec": s}


def check_dw_rules(spec):
    s = spec.get("spec", {})
    sign = str(s.get("sign_topology", ""))
    l1 = str(s.get("l1_normalization", ""))
    comp = str(s.get("composition", ""))
    bounds = str(s.get("bounds", ""))
    ok = (("NO new edges" in sign or "不新建" in sign or "不新增" in sign)
          and ("NO renormalization" in l1 or "不得" in l1)
          and ("overlay" in comp or "覆盖" in comp)
          and ("delta_max" in bounds or "Δmax" in bounds))
    return {"ok": ok, "sign_topology": sign, "l1_normalization": l1,
            "composition": comp, "bounds": bounds}


def protocol_rules_check(path=None):
    p = path or os.path.join(HERE, "P10C1_EXPERIMENT_PROTOCOL.md")
    if not os.path.exists(p):
        return {"ok": False, "missing": ["protocol missing"]}
    text = open(p, encoding="utf-8").read()
    need = ["**A**", "**B**", "**C**", "可塑性冻结", "奖励打乱", "神经运动输出消融",
            "无脑", "统计", "固定目标", "导入 Dijkstra"]
    missing = [k for k in need if k not in text]
    return {"ok": len(missing) == 0, "missing": missing}


def resource_budget():
    """静态资源（不跑动力学）；LIF 逐步实测 **MISSING**，不能拿 P9 COBA 数据替代。"""
    return {
        "subnetwork": {"n_neurons": 23218, "nnz": 3549643, "csr_float32_MB": 28.49,
                       "layer_sizes": [311, 10931, 320, 95, 11340, 1285], "n_DAN": 340},
        "plastic_edges": {"KC_to_MBON_in_subnet": 4533, "note": "仅真实 KC→MBON 子集；bodyId 可追踪"},
        "trials": {"planned_per_condition": 10, "conditions": 7, "total": 70,
                   "note": "预注册上限；未执行"},
        "lif_benchmark": "MISSING（本轮不跑 MaleCNS 动力学；P9 COBA 数据仅参考，不能替代 LIF 实测）",
        "resource_ready": False,
        "reason": "缺 LIF 子网逐步耗时实测；仅完成静态 CSR 内存/规模计算。",
        "estimate": {"csr_MB": 28.49, "note": "float32 data+indices+indptr 下界估计；不含状态/突触前迹"},
    }


def run(base=HERE):
    w = w_readonly_check()
    hist = history_verdicts(base)
    hist_ok = (len(hist["problems"]) == 0
               and hist["verdicts"].get("p100", {}).get("verdict") == "NO-GO"
               and all(hist["verdicts"].get(k, {}).get("go") is False for k in ("p100b", "p100c", "p100d")))
    dec = audit_decoder()
    fspec = check_freeze_spec()
    dw = check_dw_rules(fspec)
    prot = protocol_rules_check()
    iso = isolation_selftest()
    res = resource_budget()
    conditions = {
        "1_frozen_assets_complete": w.get("ok", False),
        "2_history_verdicts_correct": hist_ok,
        "3_decoder_ready": dec.get("decoder_ready", False),
        "4_params_preregistered": fspec.get("ok", False),
        "5_dw_rules_clear": dw.get("ok", False),
        "6_abc_independent_failure": prot.get("ok", False),
        "7_controls_and_stats": prot.get("ok", False),
        "8_student_isolation": iso.get("ok", False),
        "9_resource_budget": res.get("resource_ready", False),
    }
    execution_ready = all(conditions.values())
    reasons = [k for k, v in conditions.items() if not v]
    return {
        "checks": {"w_readonly": w, "history": hist, "decoder": dec, "freeze_spec": fspec,
                   "dw_rules": dw, "protocol_rules": prot, "isolation": iso, "resource": res},
        "conditions": conditions, "execution_ready": execution_ready,
        "training_authorized": False,
        "verdicts": {
            "Biological": "NO-GO",
            "Computational_Hypothesis": "GO (conditional, proposal only)",
            "Execution_Readiness": "GO" if execution_ready else "NO-GO",
            "Training_Authorization": False,
        },
        "blocking_reasons": reasons,
        "note": "即使全部通过，training_authorized 仍恒为 False；本轮不训练。",
    }


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    R = run()
    c = R["checks"]
    print("[P10-C1a] w:", c["w_readonly"].get("ok"), "| history:", c["history"].get("problems") == [],
          "| decoder_ready:", c["decoder"].get("decoder_ready"))
    print("  decoder risky:", c["decoder"].get("risky_direction_sources"))
    print("  freeze_spec:", c["freeze_spec"].get("ok"), "| dw_rules:", c["dw_rules"].get("ok"))
    print("  isolation:", c["isolation"].get("ok"), "| resource_ready:", c["resource"].get("resource_ready"))
    print("  conditions:", R["conditions"])
    print("  execution_ready:", R["execution_ready"], "| training_authorized:", R["training_authorized"])
    print("  blocking:", R["blocking_reasons"])
    json.dump(R, open(os.path.join(HERE, "p10c1a_gate.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote p10c1a_gate.json")
    return 0 if R["execution_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
