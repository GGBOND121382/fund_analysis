"""Regression tests for the 2026-11-27 NAV and EB ranking repairs."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from fundkit.industry_rank import NavPath, nav_quality_issue
from scripts.rank_models_m123 import eb_posteriors, process_period, EB_Z


class TestQualityAndPooling(unittest.TestCase):
    def test_flat_and_invalid_not_roundtrip(self):
        self.assertEqual(nav_quality_issue(np.ones(250)), "FLAT_NAV_REVIEW")
        self.assertEqual(nav_quality_issue([1, np.nan, 1]), "INVALID_NAV_PATH")
        self.assertEqual(nav_quality_issue([1, 0, 1]), "INVALID_NAV_PATH")
        self.assertEqual(nav_quality_issue([1, .98, 1.01, 1]), "")
        self.assertEqual(nav_quality_issue([1, 1.000001, 1.000002]), "")

    def test_zero_dispersion_does_not_replace_observations(self):
        d = pd.DataFrame({"code": [str(i) for i in range(12)], "group": "g",
                          "observed_log_cagr": np.linspace(.02, .03, 12), "annual_se": .10})
        p = eb_posteriors(d)
        self.assertTrue(p.eb_status.eq("NO_POOL_ZERO_DISPERSION").all())
        np.testing.assert_allclose(p.eb_log_cagr, d.observed_log_cagr)
        np.testing.assert_allclose(p.mu_lower, np.expm1(d.observed_log_cagr - EB_Z*d.annual_se))
        self.assertEqual(p.mu_lower.nunique(), 12)

    def test_duplicate_share_does_not_change_prior(self):
        d = pd.DataFrame({"code": [f"{i:06d}" for i in range(12)], "group": "g",
                          "product_key": [str(i) for i in range(12)],
                          "observed_log_cagr": np.linspace(.01, .15, 12), "annual_se": .005})
        a = eb_posteriors(d)
        extra = d.iloc[[0]].copy()
        extra["code"] = "999999"
        extra["observed_log_cagr"] = .5
        b = eb_posteriors(pd.concat([d, extra], ignore_index=True))
        self.assertTrue(b.peer_count.eq(12).all())
        self.assertTrue(b.eb_status.eq("EB_CONDITIONAL").all())
        np.testing.assert_allclose(a.eb_tau2, b.eb_tau2.iloc[:12])
        np.testing.assert_allclose(a.eb_log_cagr, b.eb_log_cagr.iloc[:12])
        np.testing.assert_allclose(a.eb_se, b.eb_se.iloc[:12])

    def test_sparse_group_does_not_borrow_unrelated_asset_prior(self):
        d = pd.DataFrame({"code": [str(i) for i in range(12)], "group": ["bond"] + ["equity"]*11,
                          "observed_log_cagr": [.02] + [.3]*11, "annual_se": .01})
        a = eb_posteriors(d).iloc[0]
        self.assertEqual(a.eb_status, "NO_POOL_SMALL_GROUP")
        self.assertAlmostEqual(a.eb_log_cagr, .02)
        self.assertAlmostEqual(a.eb_se, .01)

    def test_eb_uses_same_month_observation_as_se(self):
        d = pd.DataFrame({"code": ["x"], "group": ["g"], "annual_se": [.01],
                          "observed_log_cagr": [.50], "eb_observed_log_mean": [.03]})
        a = eb_posteriors(d).iloc[0]
        self.assertAlmostEqual(a.eb_log_cagr, .03)

    def test_old_scored_flat_path_cannot_enter_any_model(self):
        end = pd.Timestamp("2026-11-20")
        dates = pd.bdate_range("2025-10-01", end)
        t = np.arange(len(dates))
        values = {"003184": np.ones(len(t)),
                  "000001": np.exp(.0001*t + .003*np.sin(t/5)),
                  "000002": np.exp(.0002*t + .003*np.sin(t/4))}
        paths = {c: NavPath(dates,v) for c,v in values.items()}
        rows = []
        for c,v in values.items():
            rows.append(dict(code=c,name=c,product_key=c,group="g",status="SCORED",
                             window_start="2025-11-20",window_end=str(end.date()),
                             period_return=v[-1]/v[0]-1,R=v[-1]/v[0]-1,
                             D=.01,L=.001,Aret=.001,Add=.001,SCORE_M0=90.))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            pd.DataFrame(rows).to_csv(p/"all_scored_1y.csv",index=False)
            result, audit = process_period("1y",end,p,paths,boot_scenarios=20)
        flat=result.loc[result.code.eq("003184")].iloc[0]
        self.assertEqual(flat.model_quality_status,"FLAT_NAV_REVIEW")
        self.assertTrue(pd.isna(flat.mu_lower))
        self.assertTrue(pd.isna(flat.SCORE_M0))
        for m in ("M1","M2","M3"):
            self.assertTrue(pd.isna(flat["SCORE_"+m]))
            self.assertEqual(result["SCORE_"+m].notna().sum(), 2)
        self.assertEqual(audit["eligible_common"],2)


if __name__ == "__main__":
    unittest.main()
