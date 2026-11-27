"""Regression tests for independent ablations and currency-safe identity."""
import unittest
import numpy as np
import pandas as pd
from fundkit.model_ablation import (canonical_product_name, identity_keys,
    cny_comparable, category_residual_features, add_return_ablation, Z)
from scripts.rank_models_m123 import eb_posteriors


class TestAblation(unittest.TestCase):
    def fixture(self, n=12):
        month=pd.period_range("2025-01",periods=11,freq="M").astype(str)
        common=np.linspace(-.015,.025,len(month))
        m=[]; d=[]
        for i in range(n):
            c=f"{i:06d}"
            x=common + .001*i + .0001*np.sin(np.arange(len(month))*(i+1))
            m.extend(dict(code=c,month=k,monthly_return=np.expm1(y)) for k,y in zip(month,x))
            d.append(dict(code=c,name=f"基金{i}A",group="g",product_key=f"id:{i}",
                eb_observed_log_mean=12*x.mean(),annual_se=12*x.std(ddof=1)/np.sqrt(len(x)),
                complete_months=len(month)))
        return pd.DataFrame(d),pd.DataFrame(m)

    def test_currency_qualified_same_product(self):
        names=["易方达全球成长精选混合(QDII)美元现汇A","易方达全球成长精选混合(QDII)人民币A",
               "美元债券基金A","甲混合A","甲混合C"]
        d=pd.DataFrame({"name":names,"group":["QDII · USD","QDII","债券型","混合","混合"],
            "code":["a","b","c","d","e"],"product_key":["name:x","name:y","name:z","id:1","id:2"]})
        k=identity_keys(d)
        self.assertEqual(k.iloc[0],k.iloc[1])
        self.assertNotEqual(k.iloc[3],k.iloc[4])
        self.assertEqual(cny_comparable(d).tolist(),[False,True,True,True,True])
        self.assertEqual(canonical_product_name("美元债券基金A"),"美元债券基金")

    def test_m4_exact_and_thin_group_fallback(self):
        d,m=self.fixture(2)
        result=add_return_ablation(d,m,eb_posteriors)
        expected=np.expm1(d.eb_observed_log_mean-Z*d.annual_se)
        np.testing.assert_allclose(result.return_M4,expected)
        np.testing.assert_allclose(result.return_M5,expected)
        self.assertTrue(result.factor_status.eq("FALLBACK_NO_FACTOR").all())

    def test_factor_excludes_own_product_and_removes_shared_variation(self):
        d,m=self.fixture()
        x=m.assign(x=np.log1p(m.monthly_return)).pivot(index="code",columns="month",values="x")
        r=category_residual_features(d,m).set_index("code")
        c=d.code.iloc[0]
        benchmark=x.drop(c).mean(axis=0)
        residual=x.loc[c]-benchmark
        self.assertAlmostEqual(r.loc[c,"factor_mean"],12*benchmark.mean())
        self.assertAlmostEqual(r.loc[c,"residual_mean"],12*residual.mean())
        self.assertLess(r.loc[c,"residual_se"],d.annual_se.iloc[0]/20)
        result=add_return_ablation(d,m,eb_posteriors)
        self.assertTrue(result.factor_status.eq("FACTOR_EB").all())
        self.assertTrue(np.isfinite(result.return_M5).all())

    def test_duplicate_share_does_not_modify_other_factor(self):
        d,m=self.fixture()
        q=d.iloc[[0]].copy(); q["code"]="999999"
        qm=m.loc[m.code.eq(d.code.iloc[0])].copy()
        qm["code"]="999999"; qm["monthly_return"]+=.01
        a=category_residual_features(d,m).set_index("code")
        b=category_residual_features(pd.concat([d,q]),pd.concat([m,qm])).set_index("code")
        np.testing.assert_allclose(a.factor_mean,b.loc[a.index,"factor_mean"])
        np.testing.assert_allclose(a.residual_mean,b.loc[a.index,"residual_mean"])

    def test_factor_missing_month_falls_back(self):
        d,m=self.fixture(10)
        m=m.loc[~((m.code=="000000")&(m.month==m.month.iloc[0]))]
        result=add_return_ablation(d,m,eb_posteriors)
        self.assertTrue(result.factor_status.str.startswith("FALLBACK").all())
        np.testing.assert_allclose(result.return_M4,result.return_M5)


if __name__=="__main__":
    unittest.main()
