"""Regression tests for the reconstructed historical bootstrap experiment."""
import unittest
import numpy as np
import pandas as pd
from scripts.audit_bootstrap_reconstruction import fast_d95, reconstruct, SCENARIOS, BLOCK, SEED, BOOT_STEPS
from fundkit.opportunity_rank import stationary_bootstrap, max_drawdown


class BootstrapReconstruction(unittest.TestCase):
    def setUp(self):
        self.t=pd.Timestamp("2025-05-20")
        dates=pd.bdate_range(self.t-pd.DateOffset(years=1),self.t)
        v=np.exp(np.cumsum(np.r_[0,np.random.default_rng(32).normal(.0002,.003,len(dates)-1)]))
        self.v=v
        self.hist=pd.DataFrame({"code":"000001","date":dates.strftime("%Y-%m-%d"),"nav":v})
        ret=v[-1]/v[0]-1
        years=(dates[-1]-dates[0]).days/365.2425
        self.d=pd.DataFrame([dict(code="000001",period_return=ret,
            annualized_return=(1+ret)**(1/years)-1,D=np.max(1-v/np.maximum.accumulate(v)),
            L_daily=np.sqrt(252*np.mean(np.minimum(v[1:]/v[:-1]-1,0)**2)))])
    def test_fast_bootstrap_matches_reference(self):
        ref=np.quantile(max_drawdown(stationary_bootstrap(self.v,steps=BOOT_STEPS,
            scenarios=SCENARIOS,block_length=BLOCK,seed=SEED)),.95)
        self.assertAlmostEqual(fast_d95(self.v),ref,places=12)
    def test_reconstruction_with_identical_overlap(self):
        a,c=reconstruct(self.d,pd.concat([self.hist,self.hist.iloc[[120]]]),self.t)
        self.assertEqual(c["verified"],1)
        self.assertEqual(a.history_end.iloc[0],"2025-05-20")
        self.assertAlmostEqual(a.D95.iloc[0],fast_d95(self.v))
    def test_mismatched_pre_feature_rejected(self):
        d=self.d.copy();d["D"]+=.001
        a,c=reconstruct(d,self.hist,self.t)
        self.assertTrue(a.empty)
        self.assertEqual(c["pre_feature_mismatch"],1)
    def test_conflicting_overlap_rejected(self):
        q=self.hist.iloc[[120]].copy();q["nav"]*=1.01
        a,c=reconstruct(self.d,pd.concat([self.hist,q]),self.t)
        self.assertTrue(a.empty)
        self.assertEqual(c["overlap_conflict"],1)
    def test_future_rejected(self):
        q=self.hist.iloc[[-1]].copy();q["date"]="2025-05-21"
        with self.assertRaises(ValueError):
            reconstruct(self.d,pd.concat([self.hist,q]),self.t)


if __name__=="__main__":
    unittest.main()
