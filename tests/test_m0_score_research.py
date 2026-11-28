import unittest
import numpy as np
import pandas as pd
from fundkit.m0_score_research import (WEIGHTS, monthly_metrics, build_scores,
    product_representatives, future_matrix, portfolio_stats)


class TestM0Research(unittest.TestCase):
    def test_weights(self):
        for w in WEIGHTS.values():
            self.assertAlmostEqual(sum(w),1)
            self.assertTrue(all(x>=0 for x in w))

    def test_monthly_metrics_and_no_future(self):
        r=np.array([.01,-.02,.01,-.01,.03])
        m=pd.DataFrame(dict(code="a",month=pd.period_range("2025-01",periods=5,freq="M").astype(str),monthly_return=r))
        z=monthly_metrics(m,"2025-06-20").iloc[0]
        self.assertAlmostEqual(z.sharpe,np.sqrt(12)*r.mean()/r.std(ddof=1))
        self.assertAlmostEqual(z.sortino,np.sqrt(12)*r.mean()/np.sqrt(np.mean(np.minimum(r,0)**2)))
        self.assertAlmostEqual(z.stress_proxy,min(np.prod(1+r[i:i+3])-1 for i in range(3)))
        self.assertAlmostEqual(z.consistency,.6)
        self.assertAlmostEqual(z.recovery,.6)
        self.assertTrue(z.complete)
        self.assertFalse(monthly_metrics(m.drop(index=2),"2025-06-20").iloc[0].complete)
        with self.assertRaises(ValueError):
            monthly_metrics(m,"2025-05-20")
        with self.assertRaises(ValueError):
            monthly_metrics(pd.concat([m,m.iloc[[0]]]),"2025-06-20")

    def test_ratio_floors(self):
        m=pd.DataFrame(dict(code="a",month=pd.period_range("2025-01",periods=5,freq="M").astype(str),monthly_return=.001))
        z=monthly_metrics(m,"2025-06-20").iloc[0]
        self.assertTrue(z.sharpe_floor and z.sortino_floor)
        self.assertTrue(np.isfinite(z.sharpe) and np.isfinite(z.sortino))

    def test_product_and_baseline(self):
        d=pd.DataFrame(dict(code=["01","02","03"],name=["基金C","基金A","另一基金A"],
            product_key=["p","p","q"],annualized_return=[.04,.03,.02],D=[.01,.02,.01],
            L=[.01,.02,.01],L_daily=[.1,.2,.1],stress_proxy=[.01,-.01,.02],
            sharpe=[1,2,3],sortino=[1,2,3],recovery=[.2,.4,.1],consistency=[.8,.6,.9],
            Aret=[.01,.01,.02],Add=[.01,.02,.01]))
        z=product_representatives(d)
        self.assertEqual(set(z.code),{"02","03"})
        a=build_scores(z,True)
        np.testing.assert_allclose(a.BASE,.4*a.P_R+.35*a.P_D+
            .15*(a.Aret.rank(pct=False).sub(1)*100+(100-a.Add.rank(pct=False).sub(1)*100))/2+.1*a.P_monthly)
        np.testing.assert_allclose(a.M0_3,(.4*a.P_R+.35*a.P_D+.1*a.P_monthly)/.85)

    def test_strict_future_and_suspension(self):
        days=pd.bdate_range("2025-05-21",periods=70)
        d=pd.concat([pd.DataFrame(dict(code=c,date=days.strftime("%Y-%m-%d"),nav=np.linspace(1,1.02,len(days)))) for c in ["a","b"]])
        missing=days[2].strftime("%Y-%m-%d")
        d=d.loc[~((d.code=="b")&(d.date==missing))]
        q=future_matrix(d,["a","b"],days,{})
        self.assertEqual(portfolio_stats(q,["a","b"])["status"],"REJECTED_MISSING_OUTCOME")
        q=future_matrix(d,["a","b"],days,{"b":[missing]})
        self.assertAlmostEqual(portfolio_stats(q,["a","b"])["net_return"],1.02*(1-.0005)**2-1)
        d=d.loc[~((d.code=="b")&(d.date==days[-1].strftime("%Y-%m-%d")))]
        q=future_matrix(d,["a","b"],days,{"b":[missing,days[-1].strftime("%Y-%m-%d")]})
        self.assertEqual(portfolio_stats(q,["a","b"])["status"],"REJECTED_MISSING_OUTCOME")


if __name__=="__main__":
    unittest.main()
