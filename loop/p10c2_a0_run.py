"""P10-C2 一次性无学习 A0（Mode A）：LPLC2/LC4 固定刺激 -> KC -> KC→MBON 资格迹。

**仅运行一次**；ΔW=0；无训练/迷宫/小车。Mode A：只做探索性统计检测，**永不 A0_PASS**。
raw 数据写入临时目录（不进仓库）；仓库只提交报告/清单。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp

from loop.p100e_gate import w_readonly_check
from loop.p10c1b_lif_benchmark import _peak_memory_mb, build_subnet
from loop.p10c2_0b_adjudication import (APPROVAL_CONFIG_PATH, load_approval_config,
                                        adjudicate_raw, PROTOCOL_VERSION)

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p10c2_a0_report.json")
RAW_ROOT = os.path.join(tempfile.gettempdir(), "opencode", "p10c2_a0")
SEEDS = list(range(101, 126))
CONDITIONS = ("none", "stim", "recover")
STEPS = 400
STIM_WINDOW = (100, 300)
RECOVER_OFF = 200
DT = 0.02
TAU = 0.1
DV = 0.2
AMOUNT = round(DV * (1 - np.exp(-DT / TAU)), 6)
TAU_PRE = 0.05
TAU_POST = 0.05
TAU_E = 0.5
WALL_CAP_S = 15 * 60
MEM_CAP_MB = 2048.0
WINDOW = slice(STIM_WINDOW[0], STIM_WINDOW[1])


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _check_scope(ap):
    cfg = ap.get("config", {})
    sc = cfg.get("scope", {})
    ok = (ap.get("valid") and ap.get("status") == "APPROVED" and ap.get("mode") == "A"
          and ap.get("run_authorized") and ap.get("canonical_pass_allowed") is False
          and ap.get("threshold") is None
          and sc.get("seeds") == SEEDS and sc.get("conditions") == list(CONDITIONS)
          and sc.get("steps") == STEPS and sc.get("entrance") == "LPLC2/LC4"
          and abs(float(sc.get("dv", -1)) - DV) < 1e-12
          and abs(float(sc.get("amount", -1)) - AMOUNT) < 1e-5
          and sc.get("wall_cap_min") == 15 and sc.get("mem_cap_MB") == MEM_CAP_MB
          and sc.get("no_training") and sc.get("no_dijkstra") and sc.get("no_B_C"))
    return ok, sc


def _write_report(obj):
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                            # noqa: BLE001
        pass
    if os.path.exists(REPORT):
        print(f"REFUSING: report already exists ({REPORT}); one run only.")
        return 2
    started = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {"protocol_version": PROTOCOL_VERSION, "mode": "A", "a0_executed": True,
              "training_authorized": False, "started_at": started, "failures": []}
    raw_dir = os.path.join(RAW_ROOT, _dt.datetime.now().strftime("%Y%m%dT%H%M%S"))
    try:
        ap = load_approval_config(APPROVAL_CONFIG_PATH)
        ok, scope = _check_scope(ap)
        report["approval"] = {"valid": ap.get("valid"), "status": ap.get("status"),
                              "mode": ap.get("mode"), "threshold": ap.get("threshold"),
                              "run_authorized": ap.get("run_authorized"),
                              "canonical_pass_allowed": ap.get("canonical_pass_allowed"),
                              "checksum": ap.get("config", {}).get("checksum"),
                              "approved_by": ap.get("config", {}).get("approved_by"),
                              "approved_at": ap.get("config", {}).get("approved_at"),
                              "scope": scope}
        if not ok:
            report.update(state="INCONCLUSIVE", reason="approval scope invalid")
            _write_report(report); return 1
        w = w_readonly_check()
        report["hashes"] = {k: v.get("sha256") for k, v in w.get("files", {}).items()}
        report["hashes_ok"] = bool(w.get("ok"))
        if not w.get("ok"):
            report.update(state="INCONCLUSIVE", reason="official W/brain hash mismatch")
            _write_report(report); return 1
        import subprocess
        try:
            report["run_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(HERE),
                                                  capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:                        # noqa: BLE001
            report["run_commit"] = None
        with open(__file__, "rb") as f:
            report["script_sha256"] = _sha256_bytes(f.read())

        sub = build_subnet()
        D, L, union, Wsub = sub["D"], sub["layers"], sub["union"], sub["Wsub"]
        kc_pos = np.searchsorted(union, L[2]); mb_pos = np.searchsorted(union, L[3])
        lp_pos = np.searchsorted(union, sub["A"]["LP"])
        E = Wsub[mb_pos][:, kc_pos].tocoo()
        edge_kc, edge_mbon = E.col, E.row
        n_edges = int(E.nnz)
        os.makedirs(raw_dir, exist_ok=True)
        td = tempfile.mkdtemp(prefix="p10c2_a0_sub_")
        try:
            np.savez(os.path.join(td, "brain.npz"), ids=D["ids"][union],
                     visual=np.zeros(0, np.int32), azimuth=np.zeros(0, np.float32),
                     cell_type=D["cell_type"][union], side=D["side"][union],
                     positions=(D["positions"][union] if D.get("positions") is not None
                                else np.zeros((len(union), 3), np.float32)),
                     superclass=D["superclass"][union])
            sp.save_npz(os.path.join(td, "weights.npz"), Wsub)
            from flybrain import FlyBrain
            brain = FlyBrain(data=td, device="cpu", batch=1, seed=101, dt=DT)
            dp, dq, de = np.exp(-DT / TAU_PRE), np.exp(-DT / TAU_POST), np.exp(-DT / TAU_E)
            records, completed, manifest = [], [], {}
            failures = []
            t_start = time.perf_counter()
            for seed in SEEDS:
                for cond in CONDITIONS:
                    if time.perf_counter() - t_start > WALL_CAP_S:
                        failures.append("wall cap exceeded"); break
                    brain.reset(seed)
                    lp = np.zeros(STEPS, np.float32)
                    kc = np.zeros((STEPS, len(kc_pos)), np.float32)
                    mb = np.zeros((STEPS, len(mb_pos)), np.float32)
                    elig = np.zeros((STEPS, n_edges), np.float32)
                    p = np.zeros(len(kc_pos)); q = np.zeros(len(mb_pos)); ee = np.zeros(n_edges)
                    for t in range(STEPS):
                        inj = []
                        if cond == "stim" and STIM_WINDOW[0] <= t < STIM_WINDOW[1]:
                            inj = [(lp_pos, AMOUNT)]
                        elif cond == "recover" and STIM_WINDOW[0] <= t < RECOVER_OFF:
                            inj = [(lp_pos, AMOUNT)]
                        fired = np.asarray(brain.step(inject=inj))
                        lmask = np.isin(fired, lp_pos); kmask = np.isin(fired, kc_pos); mmask = np.isin(fired, mb_pos)
                        lp[t] = float(lmask.sum())
                        kc[t, np.searchsorted(kc_pos, fired[kmask])] = 1.0
                        mb[t, np.searchsorted(mb_pos, fired[mmask])] = 1.0
                        if t > 0:
                            p = p * dp + kc[t - 1]; q = q * dq + mb[t - 1]
                            ee = ee * de + p[edge_kc] * q[edge_mbon]
                        elig[t] = ee
                    elig_mean = elig.mean(axis=1)
                    fn = f"{seed}_{cond}.npz"
                    np.savez_compressed(os.path.join(raw_dir, fn), lp=lp, kc=kc, mb=mb,
                                        eligibility=elig, eligibility_mean=elig_mean,
                                        meta=json.dumps({"seed": seed, "condition": cond, "steps": STEPS,
                                                         "stim_window": list(STIM_WINDOW),
                                                         "amount": AMOUNT, "tau_pre": TAU_PRE,
                                                         "tau_post": TAU_POST, "tau_e": TAU_E}))
                    with open(os.path.join(raw_dir, fn), "rb") as f:
                        manifest[fn] = _sha256_bytes(f.read())
                    records.append({"seed": seed, "condition": cond,
                                    "entrance_rate": float(lp[WINDOW].sum() / (STIM_WINDOW[1] - STIM_WINDOW[0])),
                                    "kc_rate": float(kc[WINDOW].mean()),
                                    "mbon_rate": float(mb[WINDOW].mean()),
                                    "eligibility_series": elig_mean.tolist()})
                    completed.append({"seed": seed, "condition": cond,
                                      "wall_s": round(time.perf_counter() - t_start, 2),
                                      "peak_mem_MB": _peak_memory_mb()})
                    if (_peak_memory_mb() or 0) > MEM_CAP_MB:
                        failures.append("memory cap exceeded"); break
                    if time.perf_counter() - t_start > WALL_CAP_S:
                        failures.append("wall cap exceeded"); break
                if failures:
                    break
            wall_s = round(time.perf_counter() - t_start, 2)
            payload = {"protocol_version": PROTOCOL_VERSION,
                       "params": {"dt": DT, "tau_pre": TAU_PRE, "tau_post": TAU_POST, "tau_e": TAU_E,
                                  "dv": DV, "steps": STEPS, "stim_window": list(STIM_WINDOW)},
                       "records": records}
            with open(os.path.join(raw_dir, "records.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            with open(os.path.join(raw_dir, "records.json"), "rb") as f:
                manifest["records.json"] = _sha256_bytes(f.read())
            adj = adjudicate_raw(payload, approval_config_path=APPROVAL_CONFIG_PATH)
            report.update({"raw_dir": raw_dir, "raw_manifest": manifest,
                           "raw_manifest_sha256": _sha256_bytes(json.dumps(manifest, sort_keys=True).encode()),
                           "n_runs_completed": len(completed), "completed": completed,
                           "wall_s": wall_s, "peak_mem_MB": max([c.get("peak_mem_MB") or 0 for c in completed] or [0]),
                           "failures": failures, "adjudication": adj,
                           "seeds_completed": sorted(set(c["seed"] for c in completed)),
                           "state": adj.get("state"),
                           "canonical_a0_pass": False,
                           "note": "Mode A: exploratory detection only; canonical A0_PASS disabled."})
            if failures:
                report["state"] = "INCONCLUSIVE"
            _write_report(report)
            print("[P10-C2 A0] state:", report["state"], "| runs:", len(completed),
                  "| wall_s:", wall_s, "| peak_mem:", report["peak_mem_MB"])
            print("  entrance:", adj.get("entrance"), "| kc:", adj.get("kc"))
            print("  eligibility:", adj.get("eligibility"))
            print("  raw:", raw_dir)
            return 0
        finally:
            shutil.rmtree(td, ignore_errors=True)
    except Exception as e:                       # noqa: BLE001
        report.update(state="INCONCLUSIVE", reason=f"exception: {repr(e)[:200]}")
        report["failures"].append(repr(e)[:200])
        _write_report(report)
        print("EXCEPTION -> INCONCLUSIVE:", repr(e)[:200])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
