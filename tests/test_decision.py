"""决策输出测试：schema 校验与字段语义（使用本地缓存数据，无网络）。"""
import json
import os
import unittest

import jsonschema

from fundkit import decision


class DecisionTests(unittest.TestCase):
    def test_schema_valid_and_research_only(self):
        d = decision.build_decision()
        clean = {k: v for k, v in d.items() if not k.startswith("_")}
        schema = json.load(open("schemas/decision.schema.json", encoding="utf-8"))
        jsonschema.validate(clean, schema)  # 不抛异常即通过
        self.assertIn(d["decision_status"],
                      ("RESEARCH_ONLY", "NO_FEASIBLE_STRATEGY", "DATA_BLOCKED"))
        self.assertEqual(d["evidence_status"], "PARTIAL")
        self.assertEqual(len(d["fund_decisions"]), 13)
        # 股指基金默认禁新买
        equity = [f for f in d["fund_decisions"] if f["fund_code"] == "110020"]
        self.assertEqual(equity[0]["action"], "PAUSE_NEW_BUY")
        self.assertEqual(equity[0]["subscribe_cny"], 0)
        # 权重合计合理
        total_w = sum(f["current_weight"] for f in d["fund_decisions"])
        self.assertAlmostEqual(total_w, 1.0, places=3)
        # 不伪造字段
        for f in d["fund_decisions"]:
            self.assertIsNone(f["redeemable_lots"])
            self.assertIsNone(f["style_confidence"])

    def test_decision_file_written_clean(self):
        import tempfile
        d = decision.build_decision()
        with tempfile.TemporaryDirectory() as tmp:
            path = decision.write_decision(d, out_dir=tmp)
            saved = json.load(open(path, encoding="utf-8"))
        self.assertFalse(any(k.startswith("_") for k in saved))


if __name__ == "__main__":
    unittest.main()
