"""D1 offline audit for the frozen 1314-DN readout.

Reads only archived CSVs/NPZs and the frozen readout. It never constructs a
FlyBrain, never steps dynamics, and never imports hardware or motor code.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain import Readout

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
READOUT_PATH = DATA / "readout_real.npz"
SENSOR_READOUT_PATH = DATA / "readout_real_sensor.npz"
MAX_RANGE = 5.78
FOLLOW_DISTANCE = 0.9
FOLLOW_GAIN = 0.6
MODES = ("BRAIN", "ZERO", "FLIP")
FIXED_FORWARD_MPS = 0.35
WHEEL_TRACK_M = 0.19
MAX_OMEGA = 1.2
MAX_WHEEL_MPS = 0.6

SCENES: tuple[dict[str, Any], ...] = (
    {"id": "A_OPEN", "class": "A", "label": "A 无遮挡", "csv": "static_open.csv",
     "feats": "static_open_feats.npz", "side": "none"},
    {"id": "C_RIGHT_370", "class": "C", "label": "C 右侧墙 370mm", "csv": "static_right_370.csv",
     "feats": "static_right_370_feats.npz", "side": "right"},
    {"id": "C_RIGHT_PERP_400", "class": "C", "label": "C 右侧墙垂直 400mm",
     "csv": "static_right_perp_400.csv", "feats": "static_right_perp_400_feats.npz",
     "side": "right"},
    {"id": "D_BOTH_350", "class": "D", "label": "D 双侧走廊 350mm", "csv": "static_corr_0350.csv",
     "feats": "static_corr_0350_feats.npz", "side": "both"},
    {"id": "D_BOTH_400", "class": "D", "label": "D 双侧走廊 400mm", "csv": "static_corr_0400.csv",
     "feats": "static_corr_0400_feats.npz", "side": "both"},
    {"id": "D_BOTH_500", "class": "D", "label": "D 双侧走廊 500mm", "csv": "static_corr_0500.csv",
     "feats": "static_corr_0500_feats.npz", "side": "both"},
    {"id": "D_BOTH_1000", "class": "D", "label": "D 双侧走廊 1000mm", "csv": "static_corr_1000.csv",
     "feats": "static_corr_1000_feats.npz", "side": "both"},
    {"id": "FRONT_1000", "class": "EXTRA", "label": "额外：正前方 1000mm", "csv": "static_front_1000.csv",
     "feats": "static_front_1000_feats.npz", "side": "front"},
    {"id": "FR_FRONT_400", "class": "EXTRA", "label": "额外：右前+正前 400mm",
     "csv": "static_fr_front_400.csv", "feats": "static_fr_front_400_feats.npz", "side": "front"},
)

ARCHIVE_FILES = (
    "readout_real.npz",
    "readout_real_sensor.npz",
    "static_open.csv", "static_open_feats.npz",
    "static_right_370.csv", "static_right_370_feats.npz",
    "static_right_perp_400.csv", "static_right_perp_400_feats.npz",
    "static_corr_0350.csv", "static_corr_0350_feats.npz",
    "static_corr_0400.csv", "static_corr_0400_feats.npz",
    "static_corr_0500.csv", "static_corr_0500_feats.npz",
    "static_corr_1000.csv", "static_corr_1000_feats.npz",
    "static_front_1000.csv", "static_front_1000_feats.npz",
    "static_fr_front_400.csv", "static_fr_front_400_feats.npz",
)

SOURCE_FILES = (
    ("loop/train_readout.py", "training code: ridge DN readout imitates scripted follow_side=R"),
    ("loop/brain_readout_compare.py", "offline comparison code; not a natural motor readout"),
    ("loop/physical_static_diag.py", "real-ToF static diagnostic; no motors"),
    ("loop/p2_readout_recoverability.py", "LODO DN-vs-sensor probe code"),
    ("loop/decode.py", "decoder; readout mode and memory_gain"),
    ("loop/encode_tof.py", "real 5-ToF encoder; side masks"),
    ("loop/P2D28_REPORT.md", "archived causal diagnosis; FAIL"),
    ("loop/P2D29_REPORT.md", "archived L50/R50 candidate; FAIL, route closed"),
    ("loop/p2d28_metrics.json", "P2d-28 six-trial metrics"),
    ("loop/p2d28_events.csv", "P2d-28 event table"),
    ("loop/p2d29_base.json", "P2d-29 raw baseline"),
    ("loop/p2d29_offline.json", "P2d-29 raw offline result"),
    ("loop/p2d29_openloop.json", "P2d-29 raw open-loop result"),
    ("loop/p2d29_closedloop.json", "P2d-29 raw exploratory closed-loop result"),
    ("loop/p2d29_gate_confirm.json", "P2d-29 confirm-only gate"),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _num(value: Any) -> float:
    try:
        return float(value) if value not in (None, "") else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _finite(values: np.ndarray, fn) -> float | None:
    if values.size == 0 or not np.any(np.isfinite(values)):
        return None
    out = float(fn(values))
    return round(out, 1) if np.isfinite(out) else None


def _longest_same_sign(values: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    best = cur = 1
    for a, b in zip(values[:-1], values[1:]):
        if np.signbit(a) == np.signbit(b):
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _readout_meta(readout: Readout, path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "kind": readout.kind,
        "components": readout.components,
        "lambda": readout.lam,
        "bias": float(readout.b),
        "cv_score": float(readout.cv_score),
        "input": "1314 descending-neuron Trace values",
        "output": "single scalar turn",
        "trained_target": "clip(0.6 * (0.9 - dmin_R), -1, 1) for scripted follow_side=R",
        "training_code": "loop/train_readout.py",
        "training_note": "linear ridge imitates a hand-written teacher; not a natural fly motor program",
        "natural_motor_circuit": False,
    }


def load_scene(spec: dict[str, Any], readout: Readout) -> dict[str, Any]:
    csv_path = DATA / spec["csv"]
    feats_path = DATA / spec["feats"]
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows_all = list(csv.DictReader(fh))
    z = np.load(feats_path, allow_pickle=False)
    feats_all = np.asarray(z["feats"], dtype=np.float32)
    seq = np.asarray(z["tof_seq"]).astype(int).ravel()
    if len(rows_all) != len(feats_all):
        raise ValueError(f"{spec['id']}: CSV rows {len(rows_all)} != feats {len(feats_all)}")
    by_seq = {int(r["tof_seq"]): (i, r) for i, r in enumerate(rows_all)}
    aligned_rows: list[dict[str, str]] = []
    aligned_feats: list[np.ndarray] = []
    for j, sq in enumerate(seq):
        if int(sq) not in by_seq:
            raise ValueError(f"{spec['id']}: missing CSV row for tof_seq={sq}")
        _, row = by_seq[int(sq)]
        aligned_rows.append(row)
        aligned_feats.append(feats_all[j])
    pred_all = np.asarray(readout.predict(np.stack(aligned_feats)), dtype=np.float64).ravel()
    raw_all = np.asarray([_num(r.get("brain_turn_raw")) for r in aligned_rows], dtype=np.float64)
    if not np.all(np.isfinite(pred_all)) or not np.all(np.isfinite(raw_all)):
        raise ValueError(f"{spec['id']}: non-finite readout output")
    max_error = float(np.max(np.abs(pred_all - raw_all)))
    if max_error > 2e-4:
        raise ValueError(f"{spec['id']}: frozen readout does not reproduce CSV ({max_error})")

    active = np.asarray([int(r.get("DN_fired_count", "0") or 0) > 0 for r in aligned_rows])
    rows = [r for r, keep in zip(aligned_rows, active) if keep]
    idx = np.flatnonzero(active)
    feats = np.stack([aligned_feats[i] for i in idx])
    pred = pred_all[idx]
    raw = raw_all[idx]
    if len(rows) == 0:
        raise ValueError(f"{spec['id']}: no active frames")
    times = np.asarray([_num(r.get("t")) for r in rows], dtype=np.float64)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    sign = np.sign(raw)
    pos = float(np.mean(raw > 0))
    neg = float(np.mean(raw < 0))
    teacher = np.clip(FOLLOW_GAIN * (FOLLOW_DISTANCE - np.asarray(
        [_num(r.get("dmin_R")) for r in rows], dtype=np.float64)), -1.0, 1.0)
    corr = float(np.corrcoef(raw, teacher)[0, 1]) if np.std(raw) > 1e-12 and np.std(teacher) > 1e-12 else 0.0
    dmin_l = np.asarray([_num(r.get("dmin_L")) for r in rows], dtype=np.float64)
    dmin_r = np.asarray([_num(r.get("dmin_R")) for r in rows], dtype=np.float64)
    dmin_f = np.asarray([_num(r.get("dmin_F")) for r in rows], dtype=np.float64)
    tof_mm = {name: np.asarray([_num(r.get(f"{name}_mm")) for r in rows], dtype=np.float64)
              for name in ("L", "FL", "F", "FR", "R")}
    status = {name: [r.get(f"{name}_status", "") for r in rows] for name in ("L", "FL", "F", "FR", "R")}
    valid_counts = {name: int(sum(v == "VALID" for v in vals)) for name, vals in status.items()}
    outer_l_ignored = bool(valid_counts["L"] > 0 and np.nanmean(dmin_l) >= MAX_RANGE - 0.01)
    outer_r_ignored = bool(valid_counts["R"] > 0 and np.nanmean(dmin_r) >= MAX_RANGE - 0.01)
    longest = _longest_same_sign(raw)
    sustained = bool(
        abs(float(raw.mean())) >= 0.10
        and max(pos, neg) >= 0.80
        and longest * dt >= 1.0
    )
    return {
        "id": spec["id"],
        "class": spec["class"],
        "label": spec["label"],
        "side": spec["side"],
        "source_csv": str(csv_path),
        "source_feats": str(feats_path),
        "csv_sha256": sha256_file(csv_path),
        "feats_sha256": sha256_file(feats_path),
        "frames_total": int(len(rows_all)),
        "frames_active": int(len(rows)),
        "duration_s": round(float(times[-1] - times[0]), 3) if len(times) > 1 else 0.0,
        "max_readout_recompute_error": max_error,
        "neural_turn": {
            "mean": round(float(raw.mean()), 6),
            "median": round(float(np.median(raw)), 6),
            "std": round(float(raw.std()), 6),
            "min": round(float(raw.min()), 6),
            "max": round(float(raw.max()), 6),
            "positive_fraction": round(pos, 4),
            "negative_fraction": round(neg, 4),
            "sign_changes": int(np.sum(np.diff(sign) != 0)) if len(sign) > 1 else 0,
            "longest_same_sign_s": round(longest * dt, 3),
            "sustained": sustained,
            "mean_abs": round(float(np.mean(np.abs(raw))), 6),
            "teacher_corr": round(corr, 4),
        },
        "tof": {
            "valid_counts": valid_counts,
            "ranges_mm": {
                name: {
                    "mean": _finite(v, np.nanmean),
                    "min": _finite(v, np.nanmin),
                } for name, v in tof_mm.items()
            },
            "dmin_L_mean": round(float(np.nanmean(dmin_l)), 4),
            "dmin_R_mean": round(float(np.nanmean(dmin_r)), 4),
            "dmin_F_mean": round(float(np.nanmean(dmin_f)), 4),
            "outer_L_valid_but_ignored_by_neural_side": outer_l_ignored,
            "outer_R_valid_but_ignored_by_neural_side": outer_r_ignored,
        },
        "features": {
            "shape": list(feats.shape),
            "mean_l2": round(float(np.mean(np.linalg.norm(feats, axis=1))), 4),
            "mean": round(float(feats.mean()), 6),
            "max_abs": round(float(np.max(np.abs(feats))), 6),
        },
        "brain_fired_count": {
            "median": int(np.median([int(r.get("DN_fired_count", 0) or 0) for r in rows])),
            "max": int(np.max([int(r.get("DN_fired_count", 0) or 0) for r in rows])),
            "note": "archived field is total brain fired events, not a DN-only spike count",
        },
        "_frames": [
            {
                "i": int(i),
                "t": round(float(times[i]), 4),
                "tof_mm": {name: (None if not np.isfinite(tof_mm[name][i]) else round(float(tof_mm[name][i]), 1))
                           for name in ("L", "FL", "F", "FR", "R")},
                "tof_status": {name: status[name][i] for name in ("L", "FL", "F", "FR", "R")},
                "dmin_F": round(float(dmin_f[i]), 4),
                "feature_l2": round(float(np.linalg.norm(feats[i])), 4),
                "feature_bins": [round(float(v), 4) for v in
                                 np.mean(np.abs(feats[i]).reshape(18, -1), axis=1)],
                "neural_turn_raw": round(float(raw[i]), 6),
            } for i in range(len(rows))
        ],
    }


def _archive_manifest(readout: Readout) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in ARCHIVE_FILES:
        snap = DATA / name
        if not snap.is_file():
            raise FileNotFoundError(f"missing archived snapshot: {snap}")
        origin = ROOT / name
        origin_hash = sha256_file(origin) if origin.is_file() else None
        snap_hash = sha256_file(snap)
        if origin_hash is not None and origin_hash != snap_hash:
            raise ValueError(f"snapshot differs from origin: {name}")
        out.append({
            "name": name,
            "origin": str(origin) if origin.is_file() else None,
            "snapshot": str(snap),
            "bytes": snap.stat().st_size,
            "sha256": snap_hash,
            "origin_sha256": origin_hash,
            "byte_identical_to_origin": (origin_hash is None) or (origin_hash == snap_hash),
        })
    for name, role in SOURCE_FILES:
        p = ROOT / name
        if not p.is_file():
            raise FileNotFoundError(f"missing source: {p}")
        out.append({
            "name": name,
            "origin": str(p),
            "snapshot": None,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "origin_sha256": sha256_file(p),
            "role": role,
            "byte_identical_to_origin": True,
        })
    return out


def build_report() -> dict[str, Any]:
    if not READOUT_PATH.is_file():
        raise FileNotFoundError(READOUT_PATH)
    readout = Readout.load(READOUT_PATH)
    readout_meta = _readout_meta(readout, READOUT_PATH)
    sensor = Readout.load(SENSOR_READOUT_PATH)
    scenes: dict[str, Any] = {}
    for spec in SCENES:
        scenes[spec["id"]] = load_scene(spec, readout)
    scenes["B_LEFT"] = {
        "id": "B_LEFT",
        "class": "B",
        "label": "B 左侧障碍",
        "status": "MISSING",
        "reason": "no archived left-only real-ToF/DN recording was found; nothing was synthesized",
    }
    manifest = _archive_manifest(readout)
    open_scene = scenes["A_OPEN"]
    c_scenes = [scenes["C_RIGHT_370"], scenes["C_RIGHT_PERP_400"]]
    c_outer_ignored = all(s["tof"]["outer_R_valid_but_ignored_by_neural_side"] for s in c_scenes)
    c_no_side_input = all(
        abs(s["tof"]["dmin_R_mean"] - MAX_RANGE) < 0.02 and
        abs(s["tof"]["dmin_L_mean"] - MAX_RANGE) < 0.02
        for s in c_scenes
    )

    q1 = {
        "answer": "INSUFFICIENT_EVIDENCE",
        "reason": "no left-only archive; right-only archives do not enter the neural side input",
        "opposite_direction_not_testable": True,
    }
    q2 = {
        "answer": "FAIL",
        "reason": "open scene has a sustained nonzero neural turn",
        "mean": open_scene["neural_turn"]["mean"],
        "negative_fraction": open_scene["neural_turn"]["negative_fraction"],
        "longest_same_sign_s": open_scene["neural_turn"]["longest_same_sign_s"],
        "sustained": open_scene["neural_turn"]["sustained"],
    }
    q3 = {
        "answer": "INSUFFICIENT_EVIDENCE",
        "reason": "all archived scenes are constant; no readout_real approach/removal sequence exists",
    }
    q4 = {
        "answer": "INSUFFICIENT_EVIDENCE",
        "reason": "one recording per scene/class; no repeated independent runs of the same condition",
    }
    q5 = {
        "answer": "DOMAIN_MISMATCH_EVIDENCE",
        "reason": "readout is a simulated hand-teacher imitation; real outer R is valid but excluded "
                  "from the neural right-side input; real open-loop output is strongly biased",
        "outer_R_ignored_for_C": c_outer_ignored,
        "C_has_no_neural_side_input": c_no_side_input,
        "readout_training": readout_meta["training_note"],
    }
    verdict = "NO_GO"
    reasons = [
        "OPEN_LOOP_PERSISTENT_TURN",
        "C_RIGHT_OUTER_SENSOR_NOT_IN_NEURAL_SIDE",
        "READOUT_IS_HAND_TEACHER_IMITATION",
        "NO_LEFT_ARCHIVE",
        "NO_RECOVERY_ARCHIVE",
        "NO_REPEATED_RUN_STABILITY",
    ]
    if not (open_scene["neural_turn"]["sustained"] and c_outer_ignored and c_no_side_input):
        raise AssertionError("verdict preconditions changed; refusing to emit a hard NO_GO")
    return {
        "task": "DEMO-BRAIN-TURN-D1",
        "baseline": "1855e91",
        "scope": "offline audit and replay only; no new full-brain dynamics, no motors, no training",
        "verdict": verdict,
        "reason_codes": reasons,
        "readout": readout_meta,
        "sensor_baseline": {
            "path": str(SENSOR_READOUT_PATH),
            "sha256": sha256_file(SENSOR_READOUT_PATH),
            "kind": sensor.kind,
            "components": sensor.components,
            "note": "2-feature sensor baseline; not used as the brain source",
        },
        "tof_chain": {
            "sensor_order": ["L", "FL", "F", "FR", "R"],
            "sensor_angles_deg": {"L": 70.0, "FL": 30.0, "F": 0.0, "FR": -30.0, "R": -70.0},
            "avoidance_fov_deg": 120.0,
            "neural_side_groups": {"left": ["FL", "F"], "right": ["F", "FR"]},
            "outer_LR_used_for_neural_side": False,
            "note": "outer L/R (±70°) are outside the ±60° neural avoidance FOV; "
                    "they remain available for mapping but do not drive the DN side input",
        },
        "archive_manifest": manifest,
        "scenes": scenes,
        "coverage": {
            "A": "AVAILABLE",
            "B": "MISSING",
            "C": "AVAILABLE_BUT_SENSOR_NOT_IN_NEURAL_SIDE",
            "D": "AVAILABLE_BUT_NOT_LEFT_ONLY",
        },
        "questions": {"Q1_opposite_direction": q1, "Q2_open_sustained": q2,
                      "Q3_recovery": q3, "Q4_stability": q4, "Q5_domain_shift": q5},
        "next_authorization": {
            "experiment": "D1-NEXT-STATIC-TOF",
            "authorized_now": False,
            "scope": "one static real 5-ToF brain-response test; no robot motion and no training",
            "requires": ["NEW_FULL_BRAIN_AUTHORIZATION=True", "one explicit static-run approval"],
            "fixed": ["MaleCNS", "BASE PhysicalToFEncoder", "readout_real.npz", "memory_gain=0"],
            "scenes": ["A open", "B left-only", "C right-only", "D both"],
            "repeats_per_scene": 3,
            "sequence": "open -> side -> removal -> open",
            "duration_s": 10,
            "pre_registered_gates": [
                "opposite sign for left vs right",
                "open sustained |mean| < 0.05",
                "return to baseline within 1 s after removal",
                "same sign in at least 2/3 repeats",
            ],
            "does_not_fix": ["open-loop bias", "outer L/R neural-side coverage"],
        },
        "replay": {
            "modes": list(MODES),
            "memory_gain": 0.0,
            "pursuit_turn": False,
            "dijkstra": False,
            "map": False,
            "fixed_forward_mps": FIXED_FORWARD_MPS,
            "wheel_track_m": WHEEL_TRACK_M,
            "max_omega_rad_s": MAX_OMEGA,
            "actuator": "DISCONNECTED",
            "safety": "REPLAY_ONLY",
        },
        "limits": [
            "No new MaleCNS/FlyBrain run was performed.",
            "No natural motor program is claimed; the readout is a linear imitation of a hand controller.",
            "B-left and recovery/stability are missing or insufficient; no result was fabricated.",
            "P2d-29 L50/R50 route remains FAIL and closed; it is not reused as a success.",
        ],
    }


def write_outputs(report: dict[str, Any] | None = None) -> dict[str, Any]:
    report = report or build_report()
    public = dict(report)
    public["scenes"] = {k: {kk: vv for kk, vv in v.items() if kk != "_frames"}
                        for k, v in report["scenes"].items()}
    public["frames_note"] = "raw frames remain in the archived CSV/NPZ snapshots; not duplicated here"
    jpath = HERE / "D1_ACCEPTANCE.json"
    jpath.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mpath = HERE / "D1_ACCEPTANCE.md"
    mpath.write_text(render_markdown(report), encoding="utf-8")
    return report


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:+.4f}"
    return str(v)


def render_markdown(r: dict[str, Any]) -> str:
    readout = r["readout"]
    lines = [
        "# DEMO-BRAIN-TURN-D1 — 第一关验收",
        "",
        f"**基线** `{r['baseline']}`",
        f"**判定** `{r['verdict']}`",
        f"**readout** `{readout['sha256']}`",
        "",
        "本轮只读归档与冻结 readout；未创建完整脑、未跑 MaleCNS 动力学、未训练、未启动电机。",
        "",
        "## 结论",
        "",
        "现有真实五路 ToF 归档**不足以证明**左右相反方向；且已有数据已经显示无遮挡持续偏转、"
        "右侧最近传感器未进入神经侧向输入。按第一关规则判 **NO_GO**，不是 CANDIDATE_GO。",
        "",
        "## A/B/C/D",
        "",
        "| 场景 | 归档 | 神经转向均值 | 正/负比例 | 持续同号 | 关键限制 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for sid in ("A_OPEN", "B_LEFT", "C_RIGHT_370", "C_RIGHT_PERP_400", "D_BOTH_350", "D_BOTH_400"):
        s = r["scenes"][sid]
        if s.get("status") == "MISSING":
            lines.append(f"| {s['label']} | MISSING | - | - | - | 无左向归档；未合成 |")
            continue
        nt = s["neural_turn"]
        note = "outer R valid but not in neural side" if s["tof"]["outer_R_valid_but_ignored_by_neural_side"] else "archived static"
        lines.append(
            f"| {s['label']} | `{Path(s['source_csv']).name}` | {nt['mean']:+.4f} | "
            f"{nt['positive_fraction']:.3f}/{nt['negative_fraction']:.3f} | "
            f"{nt['longest_same_sign_s']:.2f}s | {note} |"
        )
    lines += [
        "",
        "B（左侧障碍）没有归档。TOF 链的 120° 神经避障 FOV 只覆盖 `FL/F/FR`；"
        "外层 `L(+70°)`、`R(-70°)` 不进入 DN 侧向输入。C 的 `R` 380–450mm 有效，"
        "但 `dmin_R` 仍为 5.78，所以 C 不是有效的“右障碍神经响应”测试。",
        "",
        "## 五个问题",
        "",
        f"1. 左右刺激是否相反？ **{r['questions']['Q1_opposite_direction']['answer']}** —— "
        f"{r['questions']['Q1_opposite_direction']['reason']}。",
        f"2. 无障碍是否持续偏转？ **{r['questions']['Q2_open_sustained']['answer']}** —— "
        f"均值 `{r['questions']['Q2_open_sustained']['mean']:+.4f}`，"
        f"负向 `{r['questions']['Q2_open_sustained']['negative_fraction']:.1%}`，"
        f"最长同号 `{r['questions']['Q2_open_sustained']['longest_same_sign_s']:.2f}s`。",
        f"3. 撤除后能否恢复？ **{r['questions']['Q3_recovery']['answer']}** —— "
        f"{r['questions']['Q3_recovery']['reason']}。",
        f"4. 不同运行是否稳定？ **{r['questions']['Q4_stability']['answer']}** —— "
        f"{r['questions']['Q4_stability']['reason']}。",
        f"5. 真实五路 ToF 域偏移？ **{r['questions']['Q5_domain_shift']['answer']}** —— "
        f"{r['questions']['Q5_domain_shift']['reason']}。",
        "",
        "## readout 身份",
        "",
        f"- `{readout['path']}` / `{readout['sha256']}`。",
        f"- `{readout['kind']}`，{readout['components']} PCs，λ={readout['lambda']}，"
        f"CV score={readout['cv_score']:+.6f}。",
        f"- 目标：`{readout['trained_target']}`。",
        "- 它是模仿手写贴墙控制器的线性解码器，**不是**天然果蝇运动神经回路，也没有独立左右输出。",
        "- P2d-29 的 L50/R50 方案仍为 FAIL_OFFLINE/FAIL，已关闭；本报告没有重启或调参。",
        "",
        "## 回放",
        "",
        "`python -m demo.brain_turn_7d.replay` 启动本地只读回放。BRAIN 使用冻结 readout 的原始输出；"
        "ZERO 置零；FLIP 仅为消融反转。左/右轮速是公式预览，不发送，界面没有驱动/手动驱动/Arduino 接口。",
        "",
        "## 下一项需要授权的范围",
        "",
        "仅当用户明确批准时，才可做一次 `D1-NEXT-STATIC-TOF`：固定现有 MaleCNS、BASE 编码器、"
        "`readout_real.npz`、`memory_gain=0`；A/B/C/D 各 3 次独立、每次 10 s、"
        "“无遮挡→单侧→撤除→无遮挡”；只记录 ToF/DN/readout，不接电机、不训练。"
        "预注册门槛为左右反号、无遮挡 `|mean|<0.05`、撤除 1 s 内回基线、重复同号 ≥2/3。"
        "该测试只能补 B/恢复/稳定性，不能修复 A 的持续偏置或 C 的外层传感器覆盖；未获授权前不运行。",
        "",
        "## 限制",
        "",
    ]
    lines += [f"- {x}" for x in r["limits"]]
    lines += [
        "",
        "`FULL_BRAIN=NO`, `MOTOR=NO`, `TRAINING=NO`, `BIOLOGICAL_VALIDITY=UNVALIDATED`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    report = write_outputs()
    print(f"D1 {report['verdict']} readout={report['readout']['sha256']}")
    print(f"wrote {HERE / 'D1_ACCEPTANCE.json'}")
    print(f"wrote {HERE / 'D1_ACCEPTANCE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
