"""D2 offline review and pre-registration report.

Nothing here loads a brain, trains a readout, or opens hardware. It audits the
archived D1 evidence and freezes the future training/validation contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from demo.brain_turn_d2 import gates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D1_DIR = ROOT / "demo" / "brain_turn_7d"
D1_JSON = D1_DIR / "D1_ACCEPTANCE.json"
READOUT = ROOT / "readout_real.npz"
P2_TRAIN = ROOT / "loop" / "train_p2_readout.py"
P2_RECOVER = ROOT / "loop" / "p2_readout_recoverability.py"
P2D29_REPORT = ROOT / "loop" / "P2D29_REPORT.md"
P2D29_OFFLINE = ROOT / "loop" / "p2d29_offline.json"

OLD_READOUT_SHA = "632cddb1c9ae43de7bde63a5bd8851036009c4625d12c113bf0e96119ace083e"

TRAIN_SEEDS = (3001, 3002, 3003)
VAL_SEEDS = (3051, 3052)
REPEATS = 3
TRAIN_DISTANCE_MM = 400
VAL_DISTANCE_MM = 700
SECONDS = 10.0
TOF_HZ = 10.0
BRAIN_HZ = 50.0
RUNS = 4 * 2 * REPEATS * len(TRAIN_SEEDS + VAL_SEEDS)
BRAIN_STEPS = int(RUNS * SECONDS * BRAIN_HZ)
TEACHER_GAIN = gates.TEACHER_GAIN
RIDGE_LAMBDA = 0.01
TRACE_TAU = 0.2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _old_route(d1: dict[str, Any]) -> dict[str, Any]:
    p2d29 = _load_json(P2D29_OFFLINE)
    return {
        "d1_verdict": d1["verdict"],
        "d1_report": str(D1_JSON),
        "readout_sha256": OLD_READOUT_SHA,
        "readout_identity": d1["readout"]["training_note"] + "; not a natural motor readout",
        "why_failed": [
            "readout_real imitates a hand-written follow_side=R teacher; it is not a natural motor readout",
            "D1 A_OPEN showed a persistent negative turn (mean -0.2773, 96.5% negative)",
            "D1 C_RIGHT used an outer R valid at 380-450mm but dmin_R remained 5.78; it never entered the neural side input",
            "D1 B_LEFT is missing; no left-only real ToF/DN archive exists",
            "P2d-29 L50/R50 remains FAIL_OFFLINE / FAIL and is closed",
        ],
        "p2d29_offline_verdict": p2d29.get("verdict"),
        "p2d29_closed": True,
        "reopen_forbidden": True,
    }


def _candidate(d1: dict[str, Any]) -> dict[str, Any]:
    scenes = d1["scenes"]
    left = [k for k, v in scenes.items() if v.get("class") == "B" and v.get("status") != "MISSING"]
    right_neural = []
    for k, v in scenes.items():
        if v.get("class") != "C":
            continue
        t = v["tof"]
        if not t["outer_R_valid_but_ignored_by_neural_side"]:
            right_neural.append(k)
    both = [k for k, v in scenes.items() if v.get("class") == "D"]
    candidate_files = sorted(str(p) for p in ROOT.glob("readout_real_p2_5tof_candidate*.npz"))
    return {
        "status": "INSUFFICIENT_EVIDENCE",
        "architecture": [
            "real FL/F/FR ToF",
            "PhysicalToFEncoder",
            "MaleCNS with fixed W (no synaptic update)",
            "1314 descending-neuron Trace",
            "new external linear motion readout",
            "omega and left/right motor preview",
        ],
        "feature_dim": 1314,
        "left_only_archives": left,
        "right_only_valid_neural_archives": right_neural,
        "both_sides_archives": both,
        "open_archive": [k for k, v in scenes.items() if v.get("class") == "A"],
        "candidate_npz_present": candidate_files,
        "candidate_trained": False,
        "training_run": False,
        "full_brain_run": False,
        "recoverability_tool": {
            "path": str(P2_RECOVER),
            "expected_inputs": "static_A/B/C/D feats",
            "current_inputs": sorted(k for k, v in scenes.items() if v.get("status") != "MISSING"),
            "compatible": False,
            "reason": "no left-only B and no static_A/B/C/D archive; static_corr* would be mislabeled C by _label",
            "run": False,
        },
        "sufficient_for_bilateral_decision": False,
        "reason": "left-only neural response data is absent; right-only C is not in the neural side input; "
                  "D both-sides data cannot establish opposite left/right signs; no repeated runs or removal data",
        "not_a_no_go_for_a_future_study": True,
        "not_a_candidate_go": True,
    }


def _prereg() -> dict[str, Any]:
    return {
        "protocol": "D2-STATIC-EXTERNAL-READOUT/v1",
        "status": "PREREGISTERED; NOT_AUTHORIZED; NOT_RUN",
        "fixed_biology": {
            "connectome_W": "official MaleCNS, read-only",
            "synaptic_dW": 0.0,
            "training_scope": "external linear readout only",
            "trace_tau_s": TRACE_TAU,
            "brain_hz": BRAIN_HZ,
            "device": "CPU",
            "batch": 1,
            "teacher": "engineering rule, not biological ground truth",
            "teacher_formula": "clip(3.0 * (prox_R - prox_L), -1, +1), positive = left",
            "teacher_used_at_inference": False,
            "inference_inputs": ["1314 DN Trace values"],
            "forbidden_at_inference": ["raw ToF", "teacher", "map", "Dijkstra", "goal bearing",
                                       "temporary bias subtraction", "sign flip", "manual left/right switch"],
            "outer_LR": "excluded from neural side input; must not be used as FL/FR substitute",
        },
        "data_collection": {
            "name": "D2-STATIC-v1",
            "scenes": ["A_open", "B_FL_only", "C_FR_only", "D_FL_FR_symmetric"],
            "distances_mm": {"train": [TRAIN_DISTANCE_MM], "validation": [VAL_DISTANCE_MM]},
            "train_seeds": list(TRAIN_SEEDS),
            "validation_seeds": list(VAL_SEEDS),
            "repeats_per_scene_seed": REPEATS,
            "seconds_per_run": SECONDS,
            "timeline_s": {"open_pre": 2.0, "side": 4.0, "open_post": 4.0},
            "windows_s": {"side": [2.0, 6.0], "recovery": [6.0, 7.0]},
            "tof_hz": TOF_HZ,
            "runs": RUNS,
            "brain_steps": BRAIN_STEPS,
            "motor": "OFF; static rig only",
            "scene_purity": {
                "B_FL_only": "FL valid; F/FR no target",
                "C_FR_only": "FR valid; F/FL no target",
                "D": "FL and FR valid and symmetric",
            },
            "right_outer_R": "not used",
        },
        "readout": {
            "type": "zero-bias linear ridge on z-scored 1314 DN Trace",
            "lambda": RIDGE_LAMBDA,
            "components": "none (all 1314 features)",
            "fit_intercept": False,
            "turn_clip": [-1.0, 1.0],
            "training_split": "train seeds 3001-3003 at 400mm",
            "validation_split": "validation seeds 3051-3052 at 700mm; never used for fitting",
            "hyperparameter_search": False,
            "feature_selection": False,
            "random_seed": 20261022,
        },
        "gates": {
            "open_zero_bias": {"max_abs_mean": gates.OPEN_MAX_ABS_MEAN},
            "unilateral_opposite": {"fl_max": -gates.SIDE_MIN_ABS_MEAN,
                                    "fr_min": gates.SIDE_MIN_ABS_MEAN,
                                    "fr_minus_fl_min": gates.SIDE_MARGIN_MIN},
            "seed_repeatability": {"fraction": gates.MIN_SEED_FRACTION},
            "removal_recovery": {"max_abs_mean": gates.RECOVERY_MAX_ABS_MEAN},
            "zero_ablation": {"omega": "exactly 0"},
            "shuffle": {"original_min_accuracy": gates.ORIGINAL_MIN_ACC,
                        "shuffled_max_accuracy": gates.SHUFFLE_MAX_ACC,
                        "direction_accuracy": "fraction of B_FL_only + C_FR_only validation runs "
                                              "whose mean turn sign is correct",
                        "shuffle": "permute validation DN feature rows across runs; do not refit"},
        },
        "judgement_rule": "all gates must pass; report every failure; no threshold changes",
        "stopping_rule": "any gate fails -> INSUFFICIENT/NO_GO as applicable; no tuning, no seed selection, no route switch",
    }


def _src_manifest(d1: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for p, role in ((READOUT, "old frozen readout (DO NOT REUSE)"),
                    (D1_JSON, "D1 permanent NO_GO report"),
                    (D1_DIR / "D1_ACCEPTANCE.md", "D1 human report"),
                    (P2_TRAIN, "future external teacher/readout reference; not run"),
                    (P2_RECOVER, "old recoverability tool; incompatible inputs"),
                    (P2D29_REPORT, "closed L50/R50 route"),
                    (P2D29_OFFLINE, "raw closed-route result")):
        if p.is_file():
            entries.append({"path": str(p), "sha256": _sha(p), "role": role})
    for scene in d1["scenes"].values():
        if scene.get("status") == "MISSING":
            continue
        entries.append({"path": scene["source_csv"], "sha256": scene["csv_sha256"],
                        "role": "real-ToF archive"})
        entries.append({"path": scene["source_feats"], "sha256": scene["feats_sha256"],
                        "role": "DN features archive"})
    return entries


def build_review() -> dict[str, Any]:
    d1 = _load_json(D1_JSON)
    old = _old_route(d1)
    cand = _candidate(d1)
    return {
        "task": "DEMO-BRAIN-TURN-D2",
        "baseline": "0d1d845",
        "scope": "offline review, pre-registration, synthetic gates; no full brain, no training, no motor",
        "conclusion": "INSUFFICIENT_EVIDENCE",
        "old_route": old,
        "candidate": cand,
        "prereg": _prereg(),
        "source_manifest": _src_manifest(d1),
        "authorization": {
            "NEW_FULL_BRAIN_AUTHORIZATION": False,
            "READOUT_TRAINING_AUTHORIZATION": False,
            "ROBOT_MOTION_AUTHORIZATION": False,
            "REAL_SYNAPTIC_LEARNING_READY": False,
            "D2_NEXT_EXPERIMENT_AUTHORIZED": False,
        },
        "next_experiment": {
            "name": "D2-AUTH-1-STATIC-TRAIN-AND-HELDOUT",
            "authorized_now": False,
            "authorization": "one explicit approval for D2-STATIC-v1 data collection plus external "
                             "linear-readout fitting and held-out evaluation",
            "full_brain": "static ToF runs under existing safety framework; no robot; W read-only; dW=0",
            "training": "external linear readout only; no connectome training",
            "runs": RUNS,
            "brain_steps": BRAIN_STEPS,
            "resource_status": "UNVERIFIED; must be measured under the existing Job limit and reported",
            "stop_conditions": [
                "resource or Job failure",
                "any gate fails after frozen evaluation",
                "inference would need raw ToF/teacher/bias/sign/manual switching",
            ],
        },
        "demo": {
            "reuse": "demo/brain_turn_7d/replay.py",
            "launcher": "demo/brain_turn_d2/dashboard.py",
            "read_only": True,
            "actuators": "DISCONNECTED",
            "post_drive_route": "NONE",
            "storyboard": "STORYBOARD.md",
        },
        "limits": [
            "No new full-brain dynamics were run.",
            "No readout was trained; no candidate weights exist.",
            "No left-only real ToF/DN data exists; this is the primary evidence gap.",
            "The right outer R-only archives are not valid neural right-side stimuli.",
            "P2d-29 remains FAIL and closed; D1 remains NO_GO.",
        ],
    }


def render_md(r: dict[str, Any]) -> str:
    c = r["candidate"]
    o = r["old_route"]
    p = r["prereg"]
    lines = [
        "# DEMO-BRAIN-TURN-D2 — 双向神经读出可行性准备",
        "",
        f"**基线** `{r['baseline']}`",
        f"**D2 结论** `{r['conclusion']}`",
        f"**D1 结论** `{o['d1_verdict']}`（永久保留，未改写）",
        "",
        "本轮只做离线审查、预注册和合成门控测试；未运行完整脑、未训练、未接电机。",
        "",
        "## 1. 旧路线为什么失败",
        "",
        f"- `readout_real.npz` SHA `{o['readout_sha256']}`；{o['readout_identity']}。",
        "- D1 A 无遮挡：`mean=-0.2773`，`96.5%` 负向，持续同号 `8.48 s`；不是零偏置。",
        "- D1 C 右侧：外层 `R` 有效但 `dmin_R` 仍为 `5.78`，没有进入神经侧向输入；"
        "不能把 R-only 数据当 FR 神经刺激。",
        "- D1 B 左侧：无归档；没有可用的左向真实 ToF/DN 反应。",
        f"- P2d-29 L50/R50：`{o['p2d29_offline_verdict']}`，已关闭；不能重开或调参。",
        "",
        "## 2. 候选双向架构",
        "",
        "```text",
        "真实 FL/F/FR ToF",
        "  → PhysicalToFEncoder",
        "  → MaleCNS（W 固定，ΔW=0）",
        "  → 1314 DN Trace",
        "  → 外部零偏置线性读出",
        "  → omega",
        "  → 左右轮（未来仅预览/受控授权）",
        "```",
        "",
        f"- DN 特征维数：**{c['feature_dim']}**。",
        f"- 左向专用归档：**{len(c['left_only_archives'])}**；"
        f"有效神经右向归档：**{len(c['right_only_valid_neural_archives'])}**；"
        f"双侧归档：**{len(c['both_sides_archives'])}**。",
        "- 现有 `static_corr_*` 是双侧，不是 FL-only/FR-only；不能替代左右独立刺激。",
        "- `p2_readout_recoverability.py` 期望 `static_A/B/C/D`；当前无 B，且 `static_corr*` 会被其 "
        "`_label` 误标成 C，因此不能直接运行它当成双向证据。",
        f"- 候选 `.npz` 是否存在：**{len(c['candidate_npz_present'])}**；未训练、未产生结果。",
        "",
        f"**候选结论：`{c['status']}`**。没有左侧数据，无法判断 1314 DN 是否足够支持双向识别；"
        "这不是 CANDIDATE_GO，也不是对现有 readout 的翻案。",
        "",
        "## 3. 预注册的训练/验证边界",
        "",
        f"- 协议：`{p['protocol']}`；状态：`{p['status']}`。",
        "- 连接组 W 只读；生物突触 ΔW=0；只训练外部线性读出。",
        "- 老师：`clip(3.0*(prox_R-prox_L), -1, +1)`，正=左；仅训练时使用，推理绝不读取。",
        "- 推理输入只有 1314 DN Trace；禁止原始 ToF、老师、地图、Dijkstra、目标方位、"
        "临时偏置、反转符号、手动左右切换。",
        f"- 训练：seeds {p['data_collection']['train_seeds']}，{TRAIN_DISTANCE_MM}mm，"
        f"{REPEATS} 次；验证：seeds {p['data_collection']['validation_seeds']}，"
        f"{VAL_DISTANCE_MM}mm，独立距离。",
        f"- 数据量：{RUNS} 次 × {SECONDS:g}s，约 {BRAIN_STEPS:,} 完整脑步；只做静态 ToF，不接电机。",
        "- 读出：1314 特征 z-score、零截距 ridge、λ=0.01；不搜索、不选特征。",
        "",
        "## 4. 冻结成功门槛",
        "",
        "| 门槛 | 固定值 |",
        "|---|---|",
        f"| 无障碍 | 所有验证 seed `|mean| ≤ {gates.OPEN_MAX_ABS_MEAN}` |",
        f"| 左右相反 | FL `≤ {-gates.SIDE_MIN_ABS_MEAN}`，FR `≥ {gates.SIDE_MIN_ABS_MEAN}`，"
        f"FR−FL `≥ {gates.SIDE_MARGIN_MIN}` |",
        f"| 独立重复 | 至少 {gates.MIN_SEED_FRACTION:.0%} 验证 seed 同号 |",
        f"| 撤除恢复 | 撤除后 1s 内 `|mean| ≤ {gates.RECOVERY_MAX_ABS_MEAN}` |",
        "| ZERO 消融 | neural turn=0，omega=0 |",
        f"| 打乱特征 | 原始方向准确率 `≥ {gates.ORIGINAL_MIN_ACC}`，打乱后 `≤ {gates.SHUFFLE_MAX_ACC}` |",
        "",
        "全部门槛必须在前瞻性冻结的验证集上通过；任何失败均报告，不调门槛、不挑 seed、不换路线。",
        "",
        "## 5. 一次性下一步授权范围",
        "",
        f"- `{r['next_experiment']['name']}`：`authorized_now={r['next_experiment']['authorized_now']}`。",
        "- 授权范围：`D2-STATIC-v1` 数据采集 + 外部线性读出训练 + 留出验证；",
        "  A/B/C/D，400mm 训练、700mm 验证，3 次重复，5 seeds。",
        "- 完整脑只在静态 ToF 采样中运行；不接电机；不训练连接组；",
        "  资源/Job 必须实测，未验证前不得声称安全。",
        "- 任何门失败即停止并保留失败素材；不自动换读出/神经群/增益/滤波/老师。",
        "",
        "## 6. 演示因果链与视频",
        "",
        "- 复用 D1 REPLAY 页面：`python -m demo.brain_turn_7d.replay --open`；页面只读，"
        "无 `/api/drive`，无电机接口。",
        "- 分镜与诚实失败备选见 `STORYBOARD.md`。",
        "- 当前页面只能展示历史 readout/失败证据，不能展示实体车运动。",
        "",
        "## 7. 限制",
        "",
    ]
    lines += [f"- {x}" for x in r["limits"]]
    lines += [
        "",
        "`FULL_BRAIN=NO`, `TRAINING=NO`, `ROBOT_MOTION=NO`, `BIOLOGICAL_VALIDITY=UNVALIDATED`。",
    ]
    return "\n".join(lines) + "\n"


def write_outputs() -> dict[str, Any]:
    review = build_review()
    (HERE / "D2_REVIEW.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "D2_REVIEW.md").write_text(render_md(review), encoding="utf-8")
    return review


def main() -> int:
    review = write_outputs()
    print(f"D2 {review['conclusion']} "
          f"left={len(review['candidate']['left_only_archives'])} "
          f"right_neural={len(review['candidate']['right_only_valid_neural_archives'])}")
    print(f"wrote {HERE / 'D2_REVIEW.json'}")
    print(f"wrote {HERE / 'D2_REVIEW.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
