"""P10-0e/P10-C1a：科学收口 + 预注册的**静态门控**（不训练、不跑动力学、不控车）。

修正（P10-C1a）：
- 正确读取 `p100.json` 的 `decision.verdict`；缺失/空裁决 → 失败。
- **全量 SHA-256**（非前 16 位）。
- 训练产物：**预期清单** + **明确禁止项**。
- **分开**表示“值得做计算实验”（worth_doing）与“已满足启动条件”（execution_ready）。
- `training_authorized` **恒为 False**。
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

KNOWN_SHA256 = {
    "brain.npz": "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca",
    "weights.npz": "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c",
}
EXPECTED_ARTIFACTS = [
    "p100.json", "p100_evidence_review.json", "p100b_path_audit.json",
    "p100c_dan_compartment.json", "p100d_synapse_localization.json",
]
FORBIDDEN_SUBSTR = ["p10c1_delta", "p10c1_training", "p10c1_learned",
                    "delta_w", "learned_w", "p10c1_checkpoint", "p10c1_weights"]


def _load(name, base=HERE):
    p = os.path.join(base, name)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:                            # noqa: BLE001
        return None


def sha256_full(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def w_readonly_check(w_dir=W_DIR):
    out = {"files": {}}
    ok = True
    for name, known in KNOWN_SHA256.items():
        p = os.path.join(w_dir, name)
        try:
            actual = sha256_full(p)
            match = actual == known
            out["files"][name] = {"path": p, "sha256": actual, "match": match}
            ok = ok and match
        except Exception as e:                    # noqa: BLE001
            out["files"][name] = {"path": p, "error": repr(e)[:160], "match": False}
            ok = False
    out["ok"] = ok
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


def history_verdicts(base=HERE):
    """严格读取历史裁决；缺失/空裁决记录到 `problems`。"""
    out, problems = {}, []
    d = _load("p100.json", base)
    if d is None:
        problems.append("missing/invalid p100.json")
    else:
        v = (d.get("decision") or {}).get("verdict")
        if not isinstance(v, str) or not v.strip():
            problems.append("p100.json: decision.verdict missing/empty")
        out["p100"] = {"verdict": v, "go": isinstance(v, str) and v.strip().upper() == "GO"}
    d = _load("p100_evidence_review.json", base)
    if d is None:
        problems.append("missing/invalid p100_evidence_review.json")
    else:
        ov = (d.get("verdict") or {}).get("overturned")
        if ov is None:
            problems.append("p100a: verdict.overturned missing")
        out["p100a"] = {"overturned": ov, "verdict": (d.get("verdict") or {}).get("verdict")}
    for name, key in (("p100b_path_audit.json", "p100b"), ("p100c_dan_compartment.json", "p100c"),
                      ("p100d_synapse_localization.json", "p100d")):
        d = _load(name, base)
        if d is None:
            problems.append(f"missing/invalid {name}")
            continue
        go = d.get("go")
        if go is None:
            go = (d.get("verdict") or {}).get("go")
        if go is None:
            problems.append(f"{key}: go/verdict.go missing")
        out[key] = {"go": go}
    return {"verdicts": out, "problems": problems}


def no_training_check(base=HERE):
    """预期产物清单 + 明确禁止项（不只看几个文件名）。"""
    missing_expected = [f for f in EXPECTED_ARTIFACTS if not os.path.exists(os.path.join(base, f))]
    hits = []
    for root, _, files in os.walk(base):
        for f in files:
            low = f.lower()
            if any(b in low for b in FORBIDDEN_SUBSTR):
                hits.append(os.path.join(root, f))
    return {"ok": len(missing_expected) == 0 and len(hits) == 0,
            "missing_expected": missing_expected, "forbidden_hits": hits}


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
        return {"ok": False, "missing": ["<protocol missing>"]}
    text = open(p, encoding="utf-8").read()
    missing = [name for name, needle in PROTOCOL_REQUIRED if needle not in text]
    not_overclaimed = ("COBA" in text) and ("不得" in text) and ("LIF" in text)
    return {"ok": len(missing) == 0 and not_overclaimed, "missing": missing,
            "not_overclaimed": not_overclaimed}


def evidence_matrix_check(path=None):
    p = path or os.path.join(HERE, "P100E_CLOSEOUT.md")
    if not os.path.exists(p):
        return {"ok": False, "missing": ["<closeout missing>"]}
    text = open(p, encoding="utf-8").read()
    keys = ["KC→MBON", "DAN", "g_unit", "叶级", "NO-GO", "逐突触"]
    return {"ok": all(k in text for k in keys), "missing": [k for k in keys if k not in text]}


def dual_verdict():
    """两个**不同**概念：值得做（proposal） vs 已满足启动条件（execution）。"""
    return {
        "Biological": "NO-GO",
        "Computational_Hypothesis": "GO (conditional, proposal only)",
        "worth_doing_computational": True,
        "execution_ready": False,          # 由 p10c1a_gate 判定；本门控不冒充准入
        "training_authorized": False,      # 恒 False
        "meaning": ("worth_doing=值得另行申请有限计算假设实验；execution_ready 由 P10-C1a 单独判定；"
                    "training_authorized 恒 False；不推翻 Biological NO-GO。"),
    }


def run(base=HERE, w_dir=W_DIR):
    checks = {
        "w_readonly": w_readonly_check(w_dir),
        "p1_freeze": p1_freeze_check(),
        "history": history_verdicts(base),
        "no_training": no_training_check(base),
        "protocol": protocol_check(),
        "evidence_matrix": evidence_matrix_check(),
    }
    hist = checks["history"]
    out = hist["verdicts"]
    hist_ok = (len(hist["problems"]) == 0
               and out.get("p100", {}).get("go") is False
               and out.get("p100a", {}).get("overturned") is False
               and all(out.get(k, {}).get("go") is False for k in ("p100b", "p100c", "p100d")))
    checks["history_no_go"] = hist_ok
    checks["all_static_ok"] = all([
        checks["w_readonly"].get("ok", False),
        checks["p1_freeze"].get("ok", False),
        hist_ok,
        checks["no_training"].get("ok", False),
        checks["protocol"].get("ok", False),
        checks["evidence_matrix"].get("ok", False),
    ])
    return {"checks": checks, "dual_verdict": dual_verdict(),
            "note": "静态门控；不训练、不跑动力学、不控车。execution_ready 由 p10c1a_gate 判定。"}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    R = run()
    c = R["checks"]
    print("[P10-0e] w_readonly:", c["w_readonly"].get("ok"), "| p1_freeze:", c["p1_freeze"].get("ok"),
          "| no_training:", c["no_training"].get("ok"))
    print("  history problems:", c["history"].get("problems"))
    print("  protocol:", c["protocol"].get("ok"), "| evidence_matrix:", c["evidence_matrix"].get("ok"),
          "| history_no_go:", c["history_no_go"])
    print("  dual verdict:", {k: v for k, v in R["dual_verdict"].items() if k != "meaning"})
    print("  all_static_ok:", c["all_static_ok"])
    json.dump(R, open(os.path.join(HERE, "p100e_gate.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote p100e_gate.json")
    return 0 if c["all_static_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
