"""P11-1c 回归：授权、安全、完整 25 seeds、完整脑 KC/MBON 基线、与截断子网差异、裁决边界。"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p11_1c experiment ===========")
    d = json.load(open(os.path.join(HERE, "p11_1c_experiment_report.json"), encoding="utf-8"))
    a = d["authorization"]; cfg = a.get("config", {}); sc = cfg.get("scope", {})
    check("authorization granted", a.get("granted") is True)
    check("authorization text present", "我批准 P11-1c" in (d.get("authorization_text") or ""))
    check("scope: 25 seeds / none / 400 / 10000", sc.get("seeds") == list(range(101, 126))
          and sc.get("conditions") == ["none"] and sc.get("steps") == 400 and sc.get("total_steps") == 10000)
    check("scope: dW=0 / no learning / 2 GiB / 15 min", sc.get("dW") == 0 and sc.get("learning") is False
          and sc.get("mem_cap_GiB") == 2.0 and sc.get("wall_cap_min") == 15)
    check("checksum present", bool(cfg.get("checksum")))
    check("SAFETY_PRECHECK PASS", d["SAFETY_PRECHECK"] == "PASS")
    check("COMPLETENESS 25/25", d["COMPLETENESS"] == "25/25")
    check("RESOURCE INCONCLUSIVE (lost job peak, not rerun)", d["RESOURCE"] == "INCONCLUSIVE"
          and "not captured" in d["resource_note"])
    check("reconstructed wall < 900 s", d["reconstructed_child_wall_s"] < 900)
    c = d["child"]
    check("child ok / 25 seeds", c["ok"] is True and len(c["seeds"]) == 25)
    check("full N/nnz 166700/25582938", c["n_neurons"] == 166700 and c["nnz"] == 25582938)
    check("KC/MBON counts 4064/97; matched 320/95", c["n_KC"] == 4064 and c["n_MBON"] == 97
          and c["n_matched_KC"] == 320 and c["n_matched_MBON"] == 95)
    check("init ~9.75 s", abs(c["init_s"] - 9.7534) < 0.05)
    check("all seeds finite", all(x["finite"] for x in c["seeds"]))
    cmp = d["comparison"]
    check("full 4064 KC mean=1.0; ge0.85/0.95=1.0", cmp["full_4064_KC"]["mean"] == 1.0
          and cmp["full_4064_KC"]["ge_0.85"] == 1.0 and cmp["full_4064_KC"]["ge_0.95"] == 1.0)
    check("full matched 320 KC mean=1.0 (NOT 0.91)", cmp["full_matched_320_KC"]["mean"] == 1.0,
          str(cmp["full_matched_320_KC"]["mean"]))
    check("subnet 320 KC mean ~0.9118", abs(cmp["subnet_320_KC"]["mean"] - 0.911799) < 1e-6)
    check("subnet ge0.85 ~0.86325", abs(cmp["subnet_320_KC"]["ge_0.85"] - 0.86325) < 1e-5)
    df = cmp["difference_matched_320_KC"]
    check("difference +0.0882", abs(df["mean"] - 0.088201) < 1e-5)
    check("difference CI [0.0599,0.1416]", abs(df["ci"]["ci_low"] - 0.05989752) < 1e-6
          and abs(df["ci"]["ci_high"] - 0.14159823) < 1e-6)
    check("25 per-seed differences; seed 114 included", len(df["per_seed"]) == 25
          and abs(df["per_seed"][13] - 0.711781) < 1e-4)
    check("full 97 MBON mean ~0.9226", abs(cmp["full_97_MBON"]["mean"] - 0.922561) < 1e-6)
    check("matched 95 MBON mean ~0.9327", abs(cmp["full_matched_95_MBON"]["mean"] - 0.932691) < 1e-6)
    check("subnet 95 MBON mean ~0.8815", abs(cmp["subnet_95_MBON"]["mean"] - 0.881472) < 1e-6)
    check("startup full KC reaches ~1.0 by bin 2", cmp["startup_bins_KC"]["full_mean"][1] > 0.99
          and cmp["startup_bins_KC"]["full_mean"][2] == 1.0)
    check("FULL_BRAIN_BASELINE HIGH", d["FULL_BRAIN_BASELINE"] == "HIGH")
    check("FULL_VS_SUBNET_DIFFERENCE OBSERVED", d["FULL_VS_SUBNET_DIFFERENCE"] == "OBSERVED")
    check("MODEL_ROOT_CAUSE NOT_ESTABLISHED", d["MODEL_ROOT_CAUSE"] == "NOT_ESTABLISHED")
    check("training not authorized", d["training_authorized"] is False)
    v = d["verdicts"]
    check("Level A/B/C/Biological NO-GO", v["Level_A_Training"] == "NO-GO" and v["Level_B"] == "NO-GO"
          and v["Level_C"] == "NO-GO" and v["Biological"] == "NO-GO")
    # raw per-seed shapes and manifest
    raw = d["raw_dir"]
    z = np.load(os.path.join(raw, "101.npz"), allow_pickle=True)
    check("per-seed kc shape 400x4064", z["kc"].shape == (400, 4064))
    check("per-seed mb shape 400x97", z["mb"].shape == (400, 97))
    ids = np.load(os.path.join(raw, "ids.npz"), allow_pickle=True)
    check("matched id arrays 320/95", ids["kc_matched"].size == 320 and ids["mb_matched"].size == 95)
    check("manifest 29 files / 26 npz", len(d["manifest"]) == 29
          and sum(1 for k in d["manifest"] if k.endswith(".npz")) == 26)
    check("manifest hash present", bool(d["manifest_sha256"]))
    # failed launch preserved
    check("failed launch report preserved", os.path.exists(os.path.join(HERE, "p11_1c_experiment_report_launch_failure.json")))
    # no raw npz tracked
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout
        check("raw full npz not tracked", "101.npz" not in tracked and "125.npz" not in tracked)
    except Exception:                            # noqa: BLE001
        check("git ls-files", False)
    # P1 freeze and preserved dirty files
    try:
        r = subprocess.run(["git", "diff", "--stat", "P1-mapfix", "--", "nav", "config.py",
                            "loop/decode.py", "loop/encode.py", "loop/run.py"],
                           cwd=ROOT, capture_output=True, text=True, timeout=30)
        check("P1 freeze empty", r.returncode == 0 and r.stdout.strip() == "")
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30).stdout
        check("pre-existing dirty files preserved", all(x in st for x in
              ("hardware/tof5_client.py", "loop/diagnose.py", "loop/p10c1b_decoder_freeze.json")))
    except Exception:                            # noqa: BLE001
        check("git checks", False)
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
