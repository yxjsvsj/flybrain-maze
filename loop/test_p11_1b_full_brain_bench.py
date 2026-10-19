"""P11-1b 回归：授权记录、Job Object 自检、完整脑一次性基准、资源上限、10k 估算、训练边界。"""
from __future__ import annotations

import json
import os
import sys

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p11_1b full-brain benchmark ===========")
    p = os.path.join(HERE, "p11_1b_report.json")
    if not os.path.exists(p):
        check("report exists", False); print("0/1"); return 1
    d = json.load(open(p, encoding="utf-8"))
    a = d["authorization"]
    check("authorization granted", a["granted"] is True)
    check("source is chat approval", "批准" in a["source"] and "chat" in a["source"])
    check("exact approval time not fabricated", a["exact_time"] == "NOT RECORDED")
    check("signature not fabricated", "none" in a["signature"])
    check("scope one run / 5+100 / 15min / 2GB",
          a["scope"]["one_run"] is True and a["scope"]["warmup_steps"] == 5
          and a["scope"]["measure_steps"] == 100 and a["scope"]["wall_cap_min"] == 15
          and a["scope"]["mem_cap_MB"] == 2048)
    check("25-seed not authorized", "25-seed model-validity experiment" in a["not_authorized"])
    check("baseline e7bc9fc", d["baseline_commit"] == "e7bc9fc")
    check("worktree not falsely claimed clean", d["worktree_clean"] is False
          and "hardware/tof5_client.py" in d["dirty_files"])
    check("official hashes ok", d["hashes_ok"] is True)
    s = d["job_selftest"]
    check("job limit selftest enforced", s["ok"] is True and "MEMORY_ERROR" in (s["output"] or ""))
    check("2 GB job limit assigned", d["job"]["process_memory_limit_MB"] == 2048.0
          and d["job"]["assigned"] is True)
    check("full brain started", d.get("full_brain_started") is True)
    check("verdict RESOURCE_OK", d["verdict"] == "RESOURCE_OK")
    c = d["child"]
    check("full N/nnz", c["n_neurons"] == 166700 and c["nnz"] == 25582938)
    check("init recorded ~2.63 s", abs(c["init_s"] - 2.6327) < 0.01)
    check("median ~8.22 ms", abs(c["median_ms"] - 8.2223) < 0.01)
    check("p95 ~11.22 ms", abs(c["p95_ms"] - 11.2237) < 0.01)
    check("cumulative 10/50/100", set(c["cumulative_s"]) == {"10", "50", "100"})
    check("finite state", c["finite_state"] is True)
    check("no stimulus / no learning / dW=0",
          c["stimulus"] == "none" and c["learning"] is False and c["dW"] == 0)
    check("params unchanged", c["params"]["gain"] == 3.0 and c["params"]["tonic"] == 0.14
          and c["params"]["noise_hz"] == 1.2 and c["params"]["dt"] == 0.02)
    par = d["parent"]
    check("wall < 15 min", par["wall_s"] < 900 and par["timeout"] is False)
    check("no memory exceed", par["memory_exceeded"] is False)
    check("job peak committed < 2 GB", par["job_peak_committed_MB"] < 2048)
    m = d["memory_reporting"]
    check("working set and committed distinguished",
          m["child_working_set_peak_MB"] < 2048 and m["enforced_job_peak_committed_MB"] < 2048
          and "unreliable" in m["note"] and "NOT used" in m["note"])
    e = d["estimate_10000_steps"]
    check("10k core ~82 s", abs(e["core_s"] - 82.22) < 0.1)
    check("10k memory = measured peak", e["memory_peak_MB"] == m["enforced_job_peak_committed_MB"])
    check("estimate not called full experiment", "not a full experiment" in e["uncertainty"])
    check("training not authorized", d["training_authorized"] is False)
    src = open(os.path.join(HERE, "p11_1b_full_brain_bench.py"), encoding="utf-8").read()
    check("no stimulus injection / no eye drive", "inject" not in src and "eye_drive" not in src)
    check("no training calls", "train" not in src.lower() or "training_authorized" in src)
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
