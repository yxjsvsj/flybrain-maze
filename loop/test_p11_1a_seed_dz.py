"""P11-1a 补充回归：seed-level paired Cohen's dz、三者区分、seed 114、历史不变、P1 freeze。只读。"""
from __future__ import annotations

import json
import os
import subprocess
import sys

RESULTS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok, extra=""):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {extra}")


def main() -> int:
    print("=========== p11_1a seed dz ===========")
    d = json.load(open(os.path.join(HERE, "p11_1a_seed_dz.json"), encoding="utf-8"))
    check("status OK", d["status"] == "OK")
    check("archive verified", d["archive_status"] == "ARCHIVED_VERIFIED")
    check("25 seeds x 320 KC", d["n_seeds"] == 25 and d["n_cells"] == 320)
    s = d["seed_level_paired_dz"]
    check("mean ~ -2.00000377e-5", abs(s["mean"] + 2.0000037693534978e-05) < 1e-15)
    check("sample SD ~ 4.58470662e-4", abs(s["sample_sd_ddof1"] - 0.000458470662124455) < 1e-15)
    check("seed-level dz ~ -0.0436234", abs(s["dz"] + 0.04362337515962108) < 1e-12)
    check("dz = mean/SD", abs(s["dz"] - s["mean"] / s["sample_sd_ddof1"]) < 1e-15)
    check("25 seed means", len(s["seed_mean_values"]) == 25)
    c = d["comparisons"]
    check("cross-cell dz ~ -0.022377", abs(c["cross_cell_dz_P11_0"] + 0.022376861423254013) < 1e-12)
    check("pooled ratio ~ -0.003161", abs(c["pooled_seed_cell_standardized_ratio"] + 0.0031611144077032804) < 1e-12)
    check("all three distinct", c["all_three_distinct"] is True)
    check("cross-cell/pooled not called paired dz", "must NOT be presented" in c["warning"])
    check("seed 114 included", d["seed_114"]["included"] is True and d["seed_114"]["index"] == 13)
    check("seed 114 value +0.000625", abs(d["seed_114"]["value"] - 0.000625000335) < 1e-9)
    check("no seed selection/deletion", d["seed_selection"].startswith("none"))
    check("historical unchanged", d["historical_unchanged"] is True)
    check("no A0 rerun", d["a0_rerun"] is False)
    check("no full-brain init", d["full_brain_initialized"] is False)
    check("training False", d["training_authorized"] is False)
    src = open(os.path.join(HERE, "p11_1a_seed_dz.py"), encoding="utf-8").read()
    check("no FlyBrain / flybrain import", "FlyBrain" not in src and "from flybrain" not in src)
    # P1 freeze
    try:
        r = subprocess.run(["git", "diff", "--stat", "P1-mapfix", "--", "nav", "config.py",
                            "loop/decode.py", "loop/encode.py", "loop/run.py"],
                           cwd=os.path.dirname(HERE), capture_output=True, text=True, timeout=30)
        check("P1 freeze empty", r.returncode == 0 and r.stdout.strip() == "")
    except Exception:                            # noqa: BLE001
        check("P1 freeze check available", False)
    # historical P11-1a report retained
    h = json.load(open(os.path.join(HERE, "p11_1a_report.json"), encoding="utf-8"))
    check("P11-1a historical retained", abs(h["p11_0_dz_errata"]["p11_0_reported_dz"] + 0.022377) < 1e-4)
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
