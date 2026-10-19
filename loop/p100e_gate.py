"""P10-0e：科学收口 + P10-C1 预注册的**静态门控**（不训练、不跑动力学、不控车）。

只做静态检查：历史裁决、预注册要素、无训练产物、官方 W 只读 hash、P1 冻结。
产出**两个独立裁决**：Biological NO-GO；Computational GO（有条件，仅提案）。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W_DIR = os.path.join(os.path.expanduser("~"), "fly-data")
KNOWN_W_HASH = "c29919aa44069a27"
KNOWN_BRAIN_HASH = "cc9bd1ecd00bd703"

ARTIFACTS = ["p100.json", "p100_evidence_review.json", "p100b_path_audit.json",
             "p100c_dan_compartment.json", "p100d_synapse_localization.json"]


def _load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def w_readonly_check():
    out = {"weights_path": os.path.join(W_DIR, "weights.npz"),
           "brain_path": os.path.join(W_DIR, "brain.npz")}
    try:
        out["weights_sha16"] = sha16(out["weights_path"])
        out["brain_sha16"] = sha16(out["brain_path"])
        out["weights_match"] = out["weights_sha16"] == KNOWN_W_HASH
        out["brain_match"] = out["brain_sha16"] == KNOWN_BRAIN_HASH
        out["ok"] = out["weights_match"] and out["brain_match"]
    except Exception as e:                        # noqa: BLE001
        out["ok"] = False
        out["error"] = repr(e)[:160]
    return out


def p1_freeze_check():
    try:
        r = subprocess.run(["git", "diff", "--stat", "P1-mapfix", "--", "nav", "config.py",
                            "loop/decode.py", "loop/encode.py", "loop/run.py"],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
        return {"ok": r.returncode == 0 and r.stdout.strip() == "", "stdout": r.stdout.strip()[:200],
                "returncode": r.returncode}
    except Exception as e:                        # noqa: BLE001
        return {"ok": False, "error": repr(e)[:160]}


def history_verdicts():
    """从历史 JSON 提取裁决，确认未被悄悄改为 GO。"""
    out = {}
    d = _load("p100.json")
    if d:
        out["p100"] = {"go": d.get("verdict", {}).get("go", d.get("go")),
                       "verdict": d.get("verdict", {}).get("verdict") or d.get("verdict")}
    d = _load("p100_evidence_review.json")
    if d:
        out["p100a"] = {"overturned": d.get("verdict", {}).get("overturned"),
                        "verdict": d.get("verdict", {}).get("verdict")}
    d = _load("p100b_path_audit.json")
    if d:
        out["p100b"] = {"go": d.get("verdict", {}).get("go"), "structural": d.get("verdict", {}).get("structural_path")}
    d = _load("p100c_dan_compartment.json")
    if d:
        out["p100c"] = {"go": d.get("go"), "learning": d.get("grading", {}).get("Learning_Readiness")}
    d = _load("p100d_synapse_localization.json")
    if d:
        out["p100d"] = {"go": d.get("go"), "learning": d.get("grading", {}).get("Learning_Readiness")}
    return out


def no_training_check():
    banned = ["p10c1_delta", "p10c1_training", "p10c1_learned", "delta_w.npz", "learned_w.npz"]
    hits = []
    for root, _, files in os.walk(HERE):
        for f in files:
            if any(b in f.lower() for b in banned):
                hits.append(os.path.join(root, f))
    return {"ok": len(hits) == 0, "hits": hits}


PROTOCOL_REQUIRED = [
    ("P10-C1", "P10-C1"), ("eta", "η"), ("tau_e", "τ_e"), ("delta_max", "Δmax"),
    ("level_A", "**A**"), ("level_B", "**B**"), ("level_C", "**C**"),
    ("dijkstra", "Dijkstra"), ("dw0", "ΔW=0"), ("frozen", "可塑性冻结"),
    ("shuffled", "奖励打乱"), ("ablation", "神经运动输出消融"), ("brainless", "无脑"),
    ("biological", "Biological"), ("computational", "Computational"),
    ("no_go", "NO-GO"), ("go", "GO"),
]


def protocol_check(path=None):
    p = path or os.path.join(HERE, "P10C1_EXPERIMENT_PROTOCOL.md")
    if not os.path.exists(p):
        return {"ok": False, "missing": p}
    text = open(p, encoding="utf-8").read()
    missing = [name for name, needle in PROTOCOL_REQUIRED if needle not in text]
    # 旧 LIF 不得被称作已标定 COBA
    not_overclaimed = ("COBA" in text) and ("不得" in text) and ("LIF" in text)
    return {"ok": len(missing) == 0 and not_overclaimed, "missing": missing,
            "not_overclaimed": not_overclaimed}


def evidence_matrix_check(path=None):
    p = path or os.path.join(HERE, "P100E_CLOSEOUT.md")
    if not os.path.exists(p):
        return {"ok": False}
    text = open(p, encoding="utf-8").read()
    keys = ["KC→MBON", "DAN", "g_unit", "叶级", "NO-GO", "逐突触"]
    return {"ok": all(k in text for k in keys), "missing": [k for k in keys if k not in text]}


def dual_verdict():
    return {
        "Biological": "NO-GO",
        "Computational_Hypothesis": "GO (conditional, proposal only)",
        "go_biological": False,
        "go_computational": True,
        "training_authorized": False,
        "meaning": "Computational GO 仅表示值得另行申请有限真实连接组计算假设实验；不授权本轮训练，不推翻 Biological NO-GO。",
    }


def run():
    checks = {
        "w_readonly": w_readonly_check(),
        "p1_freeze": p1_freeze_check(),
        "history_verdicts": history_verdicts(),
        "no_training": no_training_check(),
        "protocol": protocol_check(),
        "evidence_matrix": evidence_matrix_check(),
    }
    hist = checks["history_verdicts"]
    # 历史裁决不得为 GO
    hist_ok = True
    if "p100a" in hist:
        hist_ok = hist_ok and hist["p100a"].get("overturned") is False
    for k in ("p100b", "p100c", "p100d"):
        if k in hist:
            hist_ok = hist_ok and hist[k].get("go") is False
    checks["history_no_go"] = hist_ok
    checks["all_static_ok"] = all([
        checks["w_readonly"].get("ok", False),
        checks["p1_freeze"].get("ok", False),
        checks["no_training"].get("ok", False),
        checks["protocol"].get("ok", False),
        checks["evidence_matrix"].get("ok", False),
        hist_ok,
    ])
    out = {"checks": checks, "dual_verdict": dual_verdict(),
           "note": "静态门控；不训练、不跑动力学、不控车。默认仍 NO-GO（生物）；计算假设仅提案。"}
    return out


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    R = run()
    c = R["checks"]
    print("[P10-0e] w_readonly:", c["w_readonly"].get("ok"), "| p1_freeze:", c["p1_freeze"].get("ok"),
          "| no_training:", c["no_training"].get("ok"))
    print("  protocol:", c["protocol"].get("ok"), c["protocol"].get("missing"),
          "| evidence_matrix:", c["evidence_matrix"].get("ok"),
          "| history_no_go:", c["history_no_go"])
    print("  dual verdict:", R["dual_verdict"])
    print("  all_static_ok:", c["all_static_ok"])
    json.dump(R, open(os.path.join(HERE, "p100e_gate.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote p100e_gate.json")
    return 0 if c["all_static_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
