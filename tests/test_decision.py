"""决策输出测试：schema 校验与 fail-closed 语义（使用本地缓存，无网络）。"""
import json
import unittest
import tempfile

import jsonschema

from fundkit import decision


class DecisionTests(unittest.TestCase):
    def test_schema_valid_and_research_only(self):
        # 固定指向不存在的验证报告，保证 verified=False，断言与环境无关
        d = decision.build_decision(oos_summary_path="reports/__nonexistent__.json")
        clean = {k: v for k, v in d.items() if not k.startswith("_")}
        with open("schemas/decision.schema.json", encoding="utf-8") as fh:
            schema = json.load(fh)
        jsonschema.validate(clean, schema)
        # 净值缺失→DATA_BLOCKED；净值齐全但未用当前算法重跑验证→MANUAL_REVIEW
        self.assertIn(d["decision_status"], ("DATA_BLOCKED", "MANUAL_REVIEW"))
        self.assertEqual(d["evidence_status"],
                         "INSUFFICIENT" if d["decision_status"] == "DATA_BLOCKED"
                         else "PARTIAL")
        self.assertTrue(all(f["subscribe_cny"] == 0 for f in d["fund_decisions"]))
        self.assertTrue(all(f["target_weight"] is None for f in d["fund_decisions"]))
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