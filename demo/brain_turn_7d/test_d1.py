"""Offline/synthetic regression tests for DEMO-BRAIN-TURN-D1.

No FlyBrain is constructed, no dynamics are stepped, and no hardware is opened.
"""
from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from demo.brain_turn_7d import audit, replay

HERE = Path(__file__).resolve().parent


class D1AuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit.build_report()
        cls.data = replay.ReplayData()

    def test_readout_hash_and_identity(self):
        self.assertEqual(self.report["readout"]["sha256"],
                         "632cddb1c9ae43de7bde63a5bd8851036009c4625d12c113bf0e96119ace083e")
        self.assertFalse(self.report["readout"]["natural_motor_circuit"])
        self.assertEqual(self.report["readout"]["output"], "single scalar turn")
        self.assertIn("hand-written teacher", self.report["readout"]["training_note"])

    def test_verdict_and_reasons(self):
        self.assertEqual(self.report["verdict"], "NO_GO")
        for code in ("OPEN_LOOP_PERSISTENT_TURN",
                     "C_RIGHT_OUTER_SENSOR_NOT_IN_NEURAL_SIDE",
                     "NO_LEFT_ARCHIVE",
                     "READOUT_IS_HAND_TEACHER_IMITATION"):
            self.assertIn(code, self.report["reason_codes"])

    def test_all_archive_hashes_are_frozen_snapshots(self):
        names = {x["name"] for x in self.report["archive_manifest"]}
        for name in audit.ARCHIVE_FILES:
            self.assertIn(name, names)
        for item in self.report["archive_manifest"]:
            self.assertEqual(len(item["sha256"]), 64)
            self.assertTrue(item["byte_identical_to_origin"], item["name"])
        self.assertEqual(self.report["readout"]["sha256"], audit.sha256_file(audit.READOUT_PATH))

    def test_a_open_is_persistently_biased(self):
        a = self.report["scenes"]["A_OPEN"]
        self.assertTrue(a["neural_turn"]["sustained"])
        self.assertLess(a["neural_turn"]["mean"], -0.20)
        self.assertGreater(a["neural_turn"]["negative_fraction"], 0.90)
        self.assertGreater(a["neural_turn"]["longest_same_sign_s"], 4.0)
        self.assertEqual(self.report["questions"]["Q2_open_sustained"]["answer"], "FAIL")

    def test_tof_neural_side_fov_excludes_outer_lr(self):
        chain = self.report["tof_chain"]
        self.assertFalse(chain["outer_LR_used_for_neural_side"])
        self.assertEqual(chain["neural_side_groups"], {"left": ["FL", "F"], "right": ["F", "FR"]})
        self.assertEqual(chain["sensor_angles_deg"]["R"], -70.0)

    def test_c_right_outer_sensor_is_not_neural_side_input(self):
        for sid in ("C_RIGHT_370", "C_RIGHT_PERP_400"):
            s = self.report["scenes"][sid]
            self.assertTrue(s["tof"]["outer_R_valid_but_ignored_by_neural_side"])
            self.assertAlmostEqual(s["tof"]["dmin_R_mean"], audit.MAX_RANGE, places=2)
            self.assertGreater(s["tof"]["valid_counts"]["R"], 0)
        self.assertEqual(self.report["coverage"]["C"],
                         "AVAILABLE_BUT_SENSOR_NOT_IN_NEURAL_SIDE")

    def test_b_left_is_missing_not_fabricated(self):
        b = self.report["scenes"]["B_LEFT"]
        self.assertEqual(b["status"], "MISSING")
        self.assertNotIn("B_LEFT", self.data.scenes)
        self.assertEqual(self.report["coverage"]["B"], "MISSING")

    def test_recovery_and_stability_are_insufficient(self):
        self.assertEqual(self.report["questions"]["Q3_recovery"]["answer"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(self.report["questions"]["Q4_stability"]["answer"], "INSUFFICIENT_EVIDENCE")

    def test_frozen_readout_reproduces_every_csv_scene(self):
        for sid, s in self.report["scenes"].items():
            if s.get("status") == "MISSING":
                continue
            self.assertLess(s["max_readout_recompute_error"], 2e-4, sid)
            self.assertGreater(s["frames_active"], 0, sid)
            self.assertEqual(s["frames_active"], len(s["_frames"]), sid)

    def test_modes_share_raw_but_change_final_turn(self):
        sid = "A_OPEN"
        brain = self.data.replay(sid, "BRAIN")
        zero = self.data.replay(sid, "ZERO")
        flip = self.data.replay(sid, "FLIP")
        self.assertEqual(len(brain), len(zero))
        self.assertEqual(len(brain), len(flip))
        for b, z, f in zip(brain, zero, flip):
            self.assertEqual(b["brain_turn_raw"], z["brain_turn_raw"])
            self.assertEqual(b["brain_turn_raw"], f["brain_turn_raw"])
            self.assertAlmostEqual(z["brain_turn"], 0.0)
            self.assertAlmostEqual(z["omega_rad_s"], 0.0)
            self.assertAlmostEqual(f["brain_turn"], -b["brain_turn_raw"], places=6)
            self.assertAlmostEqual(b["brain_turn"], b["brain_turn_raw"], places=6)
            self.assertFalse(b["turn_ablation"])
            self.assertTrue(z["turn_ablation"])
            self.assertTrue(f["turn_ablation"])
            self.assertEqual(b["memory_gain"], 0.0)
            self.assertEqual(b["pursuit_turn"], 0.0)

    def test_wheel_preview_is_only_preview(self):
        f = self.data.replay("A_OPEN", "BRAIN")[0]
        p = f["motor_preview"]
        self.assertFalse(p["sent"])
        self.assertEqual(f["safety"]["actuators"], "DISCONNECTED")
        self.assertEqual(f["safety"]["drive_api"], "NOT_PRESENT")
        self.assertAlmostEqual(
            p["left_mps"],
            audit.FIXED_FORWARD_MPS - f["omega_rad_s"] * audit.WHEEL_TRACK_M / 2,
            places=4,
        )
        self.assertAlmostEqual(
            p["right_mps"],
            audit.FIXED_FORWARD_MPS + f["omega_rad_s"] * audit.WHEEL_TRACK_M / 2,
            places=4,
        )

    def test_report_outputs_exist_and_are_json_serializable(self):
        j = HERE / "D1_ACCEPTANCE.json"
        md = HERE / "D1_ACCEPTANCE.md"
        self.assertTrue(j.is_file())
        self.assertTrue(md.is_file())
        loaded = json.loads(j.read_text(encoding="utf-8"))
        self.assertEqual(loaded["verdict"], "NO_GO")
        self.assertIn("NO_GO", md.read_text(encoding="utf-8"))


class D1HttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = replay.ReplayData()
        cls.server = replay.create_server("127.0.0.1", 0, cls.data)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return r.status, r.read()

    def test_page_has_replay_and_chain(self):
        code, body = self._get("/")
        text = body.decode("utf-8")
        self.assertEqual(code, 200)
        self.assertIn("REPLAY", text)
        self.assertIn("NO MOTOR", text)
        self.assertIn("Neural activity", text)
        self.assertIn("PREVIEW ONLY", text)

    def test_meta_and_replay(self):
        code, body = self._get("/api/meta")
        self.assertEqual(code, 200)
        meta = json.loads(body)
        self.assertEqual(meta["verdict"], "NO_GO")
        self.assertEqual(meta["safety"]["actuators"], "DISCONNECTED")
        code, body = self._get("/api/replay?scene=A_OPEN&mode=BRAIN")
        self.assertEqual(code, 200)
        frames = json.loads(body)["frames"]
        self.assertGreater(len(frames), 10)
        self.assertEqual(frames[0]["mode"], "BRAIN")

    def test_zero_mode_over_http(self):
        code, body = self._get("/api/frame?scene=A_OPEN&mode=ZERO&i=3")
        self.assertEqual(code, 200)
        f = json.loads(body)
        self.assertEqual(f["brain_turn"], 0.0)
        self.assertEqual(f["omega_rad_s"], 0.0)
        self.assertNotEqual(f["brain_turn_raw"], 0.0)

    def test_no_drive_routes_and_post_forbidden(self):
        for path in ("/api/drive", "/api/manual/drive"):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self._get(path)
            self.assertEqual(cm.exception.code, 404)
        req = urllib.request.Request(self.base + "/api/drive", data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 405)

    def test_local_only_bind(self):
        with self.assertRaises(ValueError):
            replay.create_server("0.0.0.0", 0, self.data)


if __name__ == "__main__":
    unittest.main()
