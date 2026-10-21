"""P11-2-S5-0 static semantics report regression; no dynamics."""
from __future__ import annotations

import hashlib
import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "p11_2_s5_0_report.json")
S4 = os.path.join(HERE, "p11_2_s4_report.json")


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class S50Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _load(REPORT)

    def test_status_and_source(self):
        self.assertEqual(self.r["status"], "S5_0_SOURCE_SEMANTICS_COMPLETE; NO_NEW_DYNAMICS")
        self.assertEqual(self.r["source"]["sha256"],
                         "142ecfffd5d32e30655829e34b38b04024e49beae662bd078279e9610c789b0c")
        self.assertEqual(self.r["source"]["version"], "0.1.0")
        self.assertFalse(self.r["gates"]["NEW_FULL_BRAIN_AUTHORIZATION"])
        self.assertTrue(self.r["no_new_dynamics"])

    def test_step_order(self):
        s = self.r["step_semantics"]
        l = s["lines"] if "lines" in s else self.r["source"]["lines"]
        self.assertLess(l["current"], l["leak"])
        self.assertLess(l["leak"], l["tonic"])
        self.assertLess(l["tonic"], l["noise"])
        self.assertLess(l["noise"], l["inject"])
        self.assertLess(l["inject"], l["threshold"])
        self.assertLess(l["threshold"], l["reset"])
        self.assertIn("dimensionless", s["v_units"])
        self.assertEqual(s["inject"]["unit"].split(";")[0], "voltage increment in the same dimensionless v units")
        self.assertIn("not nA", s["inject"]["unit"])
        self.assertIn("W[post, pre]", s["synaptic_weights"]["orientation"])
        self.assertIn("none", s["synaptic_weights"]["normalization"])

    def test_static_math(self):
        m = self.r["static_math"]
        self.assertAlmostEqual(m["sham"], 0.772332, places=6)
        self.assertAlmostEqual(m["low"], 0.872332, places=6)
        self.assertAlmostEqual(m["primary"], 0.972332, places=6)
        self.assertTrue(m["matches"])
        self.assertTrue(m["amount_primary_formula_match"])
        self.assertTrue(m["amount_low_formula_match"])
        self.assertIn("does not predict firing-rate", m["caveat"])

    def test_indirect_feedback(self):
        f = self.r["step_semantics"]["indirect_feedback"]
        self.assertEqual(f["direct_kc_kc_edges_in_B"], 0)
        self.assertGreater(f["nonKC_2hop_kc_to_kc_endpoints"], 0)
        self.assertGreater(f["nonKC_2hop_kc_to_kc_paths"], 0)
        self.assertTrue(f["exists"])
        self.assertIn("static", f["note"])

    def test_api_limits(self):
        a = self.r["api_access"]
        self.assertTrue(a["instrumentation_required"])
        self.assertFalse(a["package_modification_required"])
        self.assertIn("post-reset", a["v_after_step"])
        self.assertIn("not stored", a["pre_reset_v"])
        self.assertIn("does not store", a["component_breakdown"])
        self.assertIn("subclass/wrapper", a["proposed"])

    def test_biological_evidence(self):
        e = self.r["biological_evidence"]
        self.assertGreaterEqual(len(e), 5)
        statuses = {x["status"] for x in e}
        self.assertIn("ENGINEERING_HAND_CALIBRATED", statuses)
        self.assertIn("CONNECTOME_STRUCTURE_ONLY", statuses)
        self.assertTrue(any("hand-calibrated, not measured" in x["biological_basis"] for x in e))
        self.assertTrue(all("not_supported" in x and x["not_supported"] for x in e))

    def test_s4_correction(self):
        c = self.r["s4_correction"]
        self.assertEqual(c["full_kc_negative"], 2737)
        self.assertEqual(c["other_kc_negative"], 2734)
        self.assertEqual(c["full_kc_positive"], 1146)
        self.assertEqual(c["other_kc_positive"], 835)
        self.assertEqual(c["positive_sum_check"], 1146)
        self.assertEqual(c["negative_sum_check"], 2737)
        self.assertEqual(c["zero_sum_check"], 181)
        self.assertTrue(c["s4_markdown_has_2737_full"])
        self.assertTrue(c["s4_markdown_has_2734_other"])
        self.assertEqual(c["s4_json_sha256"], _sha(S4))
        s4 = _load(S4)
        self.assertEqual(s4["cell_level"]["full_kc_primary"]["n_negative"], 2737)
        self.assertEqual(s4["cell_level"]["other_kc_primary"]["n_negative"], 2734)

    def test_decision(self):
        d = self.r["decision"]
        self.assertEqual(d["priority"], "B")
        a = self.r["answers"]
        self.assertIn("YES at code level", a["q1_lif_vs_prereg"])
        self.assertIn("NO validated biological calibration", a["q2_biological_parameters"])
        self.assertIn("cannot prove", a["q3_s3_evidence"])
        self.assertIn("insufficient", a["q4_reliable_voltage_logging"])
        self.assertIn("B first", a["q5_next"])
        self.assertEqual(self.r["gates"]["REAL_SYNAPTIC_LEARNING_READY"], "NO")
        self.assertEqual(self.r["gates"]["Level_A/B/C/Biological"], "NO-GO")


if __name__ == "__main__":
    unittest.main()
