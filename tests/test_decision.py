"""Decision output tests use synthetic fixture data and no network."""
import json
import unittest
import tempfile

import jsonschema

from fundkit import decision


class DecisionTests(unittest.TestCase):
    def test_schema_valid_and_research_only(self):
        d = decision.build_decision()
        clean = {k: v for k, v in d.items() if not k.startswith("_")}
        with open("schemas/decision.schema.json", encoding="utf-8") as fh:
            schema = json.load(fh)
        jsonschema.validate(clean, schema)
        self.assertIn(d["decision_status"],
                      ("RESEARCH_ONLY", "NO_FEASIBLE_STRATEGY", "DATA_BLOCKED"))
        self.assertEqual(d["evidence_status"], "INSUFFICIENT")
        self.assertEqual(d["decision_status"], "DATA_BLOCKED")
        self.assertTrue(all(f["subscribe_cny"] == 0 for f in d["fund_decisions"]))
        self.assertEqual(d["recommended_cash_hold_cny"], d["cash_budget_cny"])
        self.assertEqual(len(d["fund_decisions"]), 13)
        equity = [f for f in d["fund_decisions"] if f["fund_code"] == "110020"]
        self.assertEqual(equity[0]["action"], "PAUSE_NEW_BUY")
        self.assertEqual(equity[0]["subscribe_cny"], 0)
        total_w = sum(f["current_weight"] for f in d["fund_decisions"])
        self.assertAlmostEqual(total_w, 1.0, places=3)
        for f in d["fund_decisions"]:
            self.assertIsNone(f["redeemable_lots"])
            self.assertIsNone(f["style_confidence"])

    def test_decision_file_written_clean(self):
        d = decision.build_decision()
        with tempfile.TemporaryDirectory() as tmp:
            path = decision.write_decision(d, out_dir=tmp)
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
        self.assertFalse(any(k.startswith("_") for k in saved))


if __name__ == "__main__":
    unittest.main()