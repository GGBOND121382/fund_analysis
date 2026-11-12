"""研究用：风格窗口内的目标权重情景筛选器（不执行买卖）。

风险优先：每条候选权重同时计算完整历史路径 MDD、历史连续窗口及
联合块自助样本的未来一年 MDD 95% 分位。只在风险可行集内最大化
情景下四分位的年化净收益。未命中 5% 目标仍保留不可行状态。

重要局限：以日度恒定权重再平衡组合近似资产配置，不等于模拟逐笔定投、
场外基金 T+N 申赎与实际持仓。不得以此模型声称可保证 4% 的回撤硬上限。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd

from fundkit.risk_engine import estimate_robust_returns, portfolio_return_lcb


@dataclass(frozen=True)
class WeightSearchConfig:
    target_annual_return: float = .05
    max_drawdown: float = .04
    lookback_days: int = 756
    horizon: int = 252
    bootstrap_paths: int = 192
    block_lengths: tuple[int, ...] = (20, 60)
    historical_windows: int = 18
    candidate_count: int = 2400
    batch_size: int = 80
    min_observations: int = 252
    max_single_fund_weight: float = .40  # 集中度治理上限；不是文献给出的最优比例
    min_cash_weight: float = 0.0
    annual_cash_return: float = 0.0
    annual_risk_free_rate: float = 0.02
    turnover_cost_rate: float = 0.003  # 30bp单边压力代理；真实费用应由逐基金费率表替换
    return_prior_shrinkage_days: int = 504
    return_uncertainty_z: float = 1.0
    seed: int = 42

    def validate(self):
        if not (0 < self.max_drawdown < 1 and -1 < self.target_annual_return < 2):
            raise ValueError("目标收益和回撤参数非法")
        if not (0 <= self.min_cash_weight <= 1 and 0 < self.max_single_fund_weight <= 1):
            raise ValueError("仓位上限/现金下限非法")
        if self.horizon < 2 or self.min_observations < 2:
            raise ValueError("风险窗口长度非法")
        if any(b < 1 or b > self.horizon for b in self.block_lengths):
            raise ValueError("bootstrap 块长度非法")
        if self.candidate_count < 0 or self.bootstrap_paths < 1 or self.batch_size < 1:
            raise ValueError("样本个数非法")
        if not (0 <= self.turnover_cost_rate < 1):
            raise ValueError("交易成本参数非法")
        if self.return_prior_shrinkage_days < 0 or self.return_uncertainty_z < 0:
            raise ValueError("收益收缩/估计不确定性参数非法")
        if not np.isfinite([self.return_prior_shrinkage_days,
                            self.return_uncertainty_z,
                            self.annual_risk_free_rate]).all() or self.annual_risk_free_rate <= -1:
            raise ValueError("收益收缩/估计不确定性/无风险利率参数非法")


def _validate_returns(returns: pd.DataFrame, cfg: WeightSearchConfig) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("DATA_BLOCKED: 缺少基金日收益矩阵")
    if returns.columns.duplicated().any() or not returns.index.is_unique:
        raise ValueError("DATA_BLOCKED: 重复基金或日期")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("DATA_BLOCKED: 日期索引必须为 DatetimeIndex")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("DATA_BLOCKED: 收益日期未排序")
    out = returns.tail(cfg.lookback_days).copy()
    if len(out) < max(cfg.min_observations, cfg.horizon):
        raise ValueError("DATA_BLOCKED: 共同交易日期数量不足")
    try:
        arr = out.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("DATA_BLOCKED: 收益含非法类型") from exc
    if not np.isfinite(arr).all() or np.any(arr <= -1):
        raise ValueError("DATA_BLOCKED: 收益矩阵含缺失/非法涨跌幅；禁止以零填充")
    return out


def _scenarios(R: np.ndarray, cfg: WeightSearchConfig) -> np.ndarray:
    """多基金在同一交易日联合抽取；保留整块收益而非逐基金独立重采样。"""
    rng = np.random.default_rng(cfg.seed)
    n, d = R.shape
    all_paths = []
    historical = np.unique(np.linspace(0, n - cfg.horizon,
                                        max(1, cfg.historical_windows), dtype=int))
    for start in historical:
        all_paths.append(R[start:start + cfg.horizon])
    for b in cfg.block_lengths:
        for _ in range(cfg.bootstrap_paths):
            starts = rng.integers(0, n - b + 1, size=int(np.ceil(cfg.horizon / b)))
            idx = np.concatenate([np.arange(s, s + b) for s in starts])[:cfg.horizon]
            all_paths.append(R[idx])
    return np.stack(all_paths, axis=0)


def _risk_metrics(returns: np.ndarray, weights: np.ndarray,
                  initial_current: np.ndarray, cfg: WeightSearchConfig,
                  scenarios: np.ndarray):
    """批量复利路径；把初始换仓成本计入净值路径和最大回撤。"""
    n_assets = returns.shape[1]
    w = weights[:, :n_assets]
    cash = weights[:, n_assets]
    turnover = np.sum(np.abs(weights - initial_current[None, :]), axis=1) / 2.0
    cost = np.minimum(turnover * cfg.turnover_cost_rate, 0.95)

    r = np.einsum("std,bd->bst", scenarios, w)
    r += cash[:, None, None] * ((1 + cfg.annual_cash_return) ** (1 / 252) - 1)
    nav = np.cumprod(1 + r, axis=2) * (1.0 - cost[:, None, None])
    peak = np.maximum(1.0, np.maximum.accumulate(nav, axis=2))
    mdd = np.max(1.0 - nav / peak, axis=2)
    quantile_mdd = np.quantile(mdd, .95, axis=1)

    hist_r = returns @ w.T
    hist_r += np.broadcast_to(
        cash[None, :] * ((1 + cfg.annual_cash_return) ** (1 / 252) - 1),
        hist_r.shape,
    )
    hist_nav = np.cumprod(1 + hist_r, axis=0) * (1.0 - cost[None, :])
    hist_peak = np.maximum(1.0, np.maximum.accumulate(hist_nav, axis=0))
    hist_mdd = np.max(1.0 - hist_nav / hist_peak, axis=0)

    annualized = np.maximum(nav[:, :, -1], 1e-10) ** (252 / cfg.horizon) - 1
    q25 = np.quantile(annualized, .25, axis=1)

    # 费后Sharpe：首日把初始换仓成本计入收益路径。
    rf_daily = (1.0 + cfg.annual_risk_free_rate) ** (1.0 / 252.0) - 1.0
    scenario_net = np.empty_like(nav)
    scenario_net[:, :, 0] = nav[:, :, 0] - 1.0
    if cfg.horizon > 1:
        scenario_net[:, :, 1:] = nav[:, :, 1:] / np.maximum(nav[:, :, :-1], 1e-12) - 1.0
    scenario_excess = scenario_net - rf_daily
    scenario_std = np.std(scenario_net, axis=2, ddof=1)
    scenario_mean = np.mean(scenario_excess, axis=2)
    scenario_sharpe = np.where(
        scenario_std > 1e-10,
        scenario_mean / np.maximum(scenario_std, 1e-10) * np.sqrt(252.0),
        np.where(scenario_mean > 1e-12, 10.0,
                 np.where(scenario_mean < -1e-12, -10.0, 0.0)),
    )
    scenario_sharpe = np.clip(scenario_sharpe, -10.0, 10.0)
    scenario_sharpe_q25 = np.quantile(scenario_sharpe, .25, axis=1)
    scenario_sharpe_median = np.quantile(scenario_sharpe, .50, axis=1)

    hist_net = np.empty_like(hist_nav)
    hist_net[0, :] = hist_nav[0, :] - 1.0
    if len(hist_nav) > 1:
        hist_net[1:, :] = hist_nav[1:, :] / np.maximum(hist_nav[:-1, :], 1e-12) - 1.0
    hist_std = np.std(hist_net, axis=0, ddof=1)
    hist_mean = np.mean(hist_net - rf_daily, axis=0)
    historical_sharpe = np.where(
        hist_std > 1e-10,
        hist_mean / np.maximum(hist_std, 1e-10) * np.sqrt(252.0),
        np.where(hist_mean > 1e-12, 10.0,
                 np.where(hist_mean < -1e-12, -10.0, 0.0)),
    )
    historical_sharpe = np.clip(historical_sharpe, -10.0, 10.0)
    return (hist_mdd, quantile_mdd, q25, turnover, historical_sharpe,
            scenario_sharpe_q25, scenario_sharpe_median)

def optimize_target_weights(returns: pd.DataFrame,
                            current_weights: Mapping[str, float] | None = None,
                            caps: Mapping[str, float] | None = None,
                            fixed_weights: Mapping[str, float] | None = None,
                            cfg: WeightSearchConfig | None = None,
                            categories: Mapping[str, str] | None = None) -> dict:
    """固定随机种子、多情景风险优先的权重搜索；结果始终是研究候选。

    returns: 已按披露时点切割、已严格对齐交易日期、已处理分红的日收益。
    外层负责识别当前风格/选择同一 regime 窗口，禁止跨风格盲目拼接。
    caps 允许根据季报风格及申购资格设为 0，默认每只基金40%为治理约束。
    收益/MDD达标后按情景Sharpe Q25排序；Sharpe仍是历史估计，不是保证。
    """
    cfg = cfg or WeightSearchConfig()
    cfg.validate()
    Rdf = _validate_returns(returns, cfg)
    codes = list(Rdf.columns)
    R = Rdf.to_numpy(float)
    n = len(codes)
    robust_estimates = estimate_robust_returns(
        Rdf, categories=categories,
        prior_shrinkage_days=cfg.return_prior_shrinkage_days,
        uncertainty_z=cfg.return_uncertainty_z,
    )
    if caps or fixed_weights:
        unknown = (set(caps or {}) | set(fixed_weights or {})).difference(codes)
        if unknown:
            raise ValueError(f"仓位上限/冻结仓位存在未知基金：{sorted(unknown)}")
    base = np.array([float((fixed_weights or {}).get(c, 0)) for c in codes])
    upper = np.array([float((caps or {}).get(c, cfg.max_single_fund_weight)) for c in codes])
    for i, code in enumerate(codes):
        if code in (fixed_weights or {}):
            upper[i] = base[i]  # 存量冻结，无新增无擅自赎回
    if (not np.isfinite(upper).all() or not np.isfinite(base).all() or
            (base < 0).any() or (upper < base).any() or (upper > 1).any() or
            base.sum() + cfg.min_cash_weight > 1 + 1e-10):
        raise ValueError("基金仓位上限/冻结权重不合法")
    free_budget = 1 - cfg.min_cash_weight - base.sum()
    cur = np.array([float((current_weights or {}).get(c, 0)) for c in codes] +
                   [float((current_weights or {}).get('cash', 0))])
    if (cur < 0).any() or not np.isfinite(cur).all() or cur.sum() > 1.000001:
        raise ValueError("账户持仓权重非法")
    if cur.sum() < .999999:
        cur[-1] += 1 - cur.sum()  # 非优化资产计入现金仅用于初始换仓成本估计，不用于持仓风险保证

    # 核心是风险优先的可行域搜索，而非手填 45/35/5。风险可行性使用严格复利路径重新检查。
    rng = np.random.default_rng(cfg.seed)
    candidates = [np.r_[base, 1.0 - base.sum()]]  # 真实存量冻结后的现金基准
    for i in range(n):
        row = base.copy()
        row[i] += min(upper[i] - base[i], free_budget)
        candidates.append(np.r_[row, 1 - row.sum()])
    extra = np.minimum(np.maximum(cur[:n] - base, 0), upper - base)
    extra *= min(1, free_budget / max(1e-12, extra.sum()))
    row = base + extra
    candidates.append(np.r_[row, 1.0 - row.sum()])
    active = np.flatnonzero(upper > base + 1e-12)
    row = base.copy()
    if len(active):
        row[active] += np.minimum(upper[active] - base[active], free_budget / len(active))
    candidates.append(np.r_[row, 1 - row.sum()])
    if cfg.candidate_count and len(active):
        # 冻结的存量权重保持不变；随机候选只分配可交易部分。
        for alpha in (.35, 1.0, 2.5):
            z = rng.dirichlet(np.full(len(active) + 1, alpha),
                              size=max(1, cfg.candidate_count // 3))
            x = np.tile(base, (len(z), 1))
            add = np.minimum(free_budget * z[:, :-1], (upper - base)[active][None, :])
            x[:, active] += add
            candidates.extend(np.column_stack([x, 1 - x.sum(axis=1)]))
    candidate_array = np.asarray(candidates, dtype=float)
    valid = ((candidate_array[:, -1] >= cfg.min_cash_weight - 1e-9) &
             (candidate_array[:, :n] <= upper + 1e-9).all(axis=1) &
             (candidate_array[:, :n] >= base - 1e-9).all(axis=1))
    candidate_array = candidate_array[valid]
    if not len(candidate_array):
        raise ValueError("候选集为空")
    paths = _scenarios(R, cfg)
    best_feasible, best_target_feasible, best_risk = None, None, None
    feasible_count = 0
    checked = 0
    # 风险约束优先；同时满足收益目标时最大化保守Sharpe，再比较收益、回撤和换手。
    for start in range(0, len(candidate_array), cfg.batch_size):
        batch = candidate_array[start:start + cfg.batch_size]
        (hdd, qdd, qret, turn, hist_sharpe,
         scenario_sharpe_q25, scenario_sharpe_median) = _risk_metrics(R, batch, cur, cfg, paths)
        for j, w in enumerate(batch):
            checked += 1
            weight_map = {**{c: float(w[i]) for i, c in enumerate(codes)},
                          "cash": float(w[-1])}
            robust_lcb = portfolio_return_lcb(
                weight_map, robust_estimates, cash_return=cfg.annual_cash_return)
            score = min(float(qret[j]), robust_lcb)
            item = {'weights': w, 'historical_mdd': float(hdd[j]),
                    'scenario_mdd_p95': float(qdd[j]), 'scenario_return_q25': float(qret[j]),
                    'robust_return_lcb': float(robust_lcb), 'robust_return_score': float(score),
                    'historical_sharpe': float(hist_sharpe[j]),
                    'scenario_sharpe_q25': float(scenario_sharpe_q25[j]),
                    'scenario_sharpe_median': float(scenario_sharpe_median[j]),
                    'turnover_proxy': float(turn[j])}
            loss = max(item['historical_mdd'], item['scenario_mdd_p95'])
            risk_key = (loss, -item['scenario_sharpe_q25'],
                        -item['robust_return_score'], item['turnover_proxy'])
            if best_risk is None or risk_key < best_risk[0]:
                best_risk = (risk_key, item)
            if loss > cfg.max_drawdown + 1e-12:
                continue
            feasible_count += 1
            if item['robust_return_score'] > cfg.target_annual_return:
                target_key = (-item['scenario_sharpe_q25'],
                              -item['robust_return_score'], loss,
                              item['turnover_proxy'])
                if best_target_feasible is None or target_key < best_target_feasible[0]:
                    best_target_feasible = (target_key, item)
            else:
                feasible_key = (-item['robust_return_score'],
                                -item['scenario_sharpe_q25'], loss,
                                item['turnover_proxy'])
                if best_feasible is None or feasible_key < best_feasible[0]:
                    best_feasible = (feasible_key, item)
    chosen = (best_target_feasible or best_feasible or best_risk)[1]
    # 第二次独立随机种子情景重检；仍共享同一历史经验分布，不等于时间样本外。
    holdout = _scenarios(R, replace(cfg, seed=cfg.seed + 1000003))
    (_, independent_mdd, independent_q25, _, independent_hist_sharpe,
     independent_sharpe_q25, independent_sharpe_median) = _risk_metrics(
        R, chosen['weights'][None, :], cur, cfg, holdout)
    independent_risk = float(independent_mdd[0])
    independent_return = float(independent_q25[0])
    status = ('RISK_CONSTRAINT_UNMET' if feasible_count == 0 else
              'INDEPENDENT_SCENARIO_REJECTED' if independent_risk > cfg.max_drawdown + 1e-12 else
              'RETURN_TARGET_UNMET' if min(chosen['scenario_return_q25'], independent_return,
                  chosen['robust_return_lcb']) <= cfg.target_annual_return
              else 'SCENARIO_CRITERIA_MET_UNVALIDATED')
    return {
        'status': status, 'trade_ready': False,
        'research_only': True,
        'data_start': str(Rdf.index[0].date()), 'data_end': str(Rdf.index[-1].date()),
        'n_joint_observations': len(Rdf), 'codes': codes,
        'weights': {**{c: round(float(v), 10) for c, v in zip(codes, chosen['weights'][:-1])},
                    'cash': round(float(chosen['weights'][-1]), 10)},
        'candidate_count_evaluated': checked,
        'risk_feasible_candidate_count': feasible_count,
        'historical_mdd': chosen['historical_mdd'],
        'scenario_mdd_p95': chosen['scenario_mdd_p95'],
        'scenario_return_q25': chosen['scenario_return_q25'],
        'robust_return_lcb': chosen['robust_return_lcb'],
        'robust_return_score': chosen['robust_return_score'],
        'historical_sharpe': chosen['historical_sharpe'],
        'scenario_sharpe_q25': chosen['scenario_sharpe_q25'],
        'scenario_sharpe_median': chosen['scenario_sharpe_median'],
        'independent_scenario_sharpe_q25': float(independent_sharpe_q25[0]),
        'independent_scenario_sharpe_median': float(independent_sharpe_median[0]),
        'annual_risk_free_rate_assumption': cfg.annual_risk_free_rate,
        'robust_return_estimates': robust_estimates,
        'concentration_guardrail': {
            'max_single_fund_weight': cfg.max_single_fund_weight,
            'status': 'GOVERNANCE_CAP_NOT_EMPIRICALLY_OPTIMIZED',
        },
        'independent_scenario_mdd_p95': independent_risk,
        'independent_scenario_return_q25': independent_return,
        'turnover_proxy': chosen['turnover_proxy'],
        'risk_budget': cfg.max_drawdown,
        'frozen_current_weights': {c: float(v) for c, v in zip(codes, base) if v > 0},
        'return_target': cfg.target_annual_return,
        'method': 'fixed-weight joint historical + moving-block bootstrap, risk-first reproducible search',
        'limits': [
            '不是全局最优保证；候选网格与样本模拟均有有限采样误差',
            '情景检验不是未来4%硬回撤保证，也不是95%收益置信下界',
            '不同随机种子仅验证模拟稳定性，不是时间样本外；需独立滚动样本外回测',
            '实际交易须另行模拟逐笔申购、赎回、锁定期、分红与真实费用',
            '只识别当前风格后选取历史窗口，旧风格风险需通过现时因子敞口压力测试',
        ],
    }


def build_returns_asof(adapter, codes: list[str], until_ts,
                       cfg: WeightSearchConfig | None = None) -> pd.DataFrame:
    """从已有数据适配器构建共同日期总收益；以实际 published_at <= 决策时刻截断。"""
    cfg = cfg or WeightSearchConfig()
    df = adapter.nav(codes, until_ts)
    if 'published_at' in df.columns and len(df):
        publication = pd.to_datetime(df['published_at'])
        cutoff = pd.Timestamp(until_ts)
        if publication.dt.tz is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(publication.dt.tz)
        if (publication > cutoff).any():
            raise ValueError('LOOKAHEAD_BLOCKED: 未来公开净值进入收益矩阵')
    frames = []
    for code in codes:
        g = df[df['fund_code'] == code].sort_values('nav_date')
        if g.empty or g['nav_date'].duplicated().any():
            raise ValueError(f'DATA_BLOCKED: {code} 净值缺失或重复')
        nav = pd.to_numeric(g['adj_nav_total_return'], errors='coerce')
        if (nav <= 0).any() or nav.isna().any():
            raise ValueError(f'DATA_BLOCKED: {code} 非法复权净值')
        s = pd.Series(nav.to_numpy(dtype=float),
                      index=pd.DatetimeIndex(g['nav_date']), name=code)
        frames.append(s.pct_change(fill_method=None).dropna())
    if not frames:
        raise ValueError('DATA_BLOCKED: 无可用收益')
    common = pd.concat(frames, axis=1, join='inner').dropna()
    return _validate_returns(common, cfg)


def load_frozen_research_weights(path: str, backtest_start,
                                 required_codes: list[str]) -> dict[str, float]:
    """把经过时间边界检查的研究候选传入模拟器；绝不写回交易配置。"""
    import json
    with open(path, encoding='utf-8') as fh:
        report = json.load(fh)
    if report.get('status') != 'SCENARIO_CRITERIA_MET_UNVALIDATED' or report.get('trade_ready'):
        raise ValueError('DATA_BLOCKED: 未取得满足情景条件的研究候选权重')
    if not report.get('data_end') or pd.Timestamp(report['data_end']) >= pd.Timestamp(backtest_start):
        raise ValueError('LOOKAHEAD_BLOCKED: 权重拟合日期不得晚于/等于回测开始日')
    weights = report.get('weights') or {}
    if set(weights) != set(required_codes) | {'cash'}:
        raise ValueError('DATA_BLOCKED: 权重代码与全部持仓不一致')
    v = np.asarray([weights[c] for c in required_codes + ['cash']], dtype=float)
    if not np.isfinite(v).all() or (v < -1e-10).any() or abs(v.sum() - 1) > 1e-6:
        raise ValueError('DATA_BLOCKED: 冻结权重不满足守恒/非负约束')
    return {c: float(weights[c]) for c in required_codes + ['cash']}


def generate_weight_candidates(returns: pd.DataFrame,
                               current_weights: Mapping[str, float] | None = None,
                               caps: Mapping[str, float] | None = None,
                               fixed_weights: Mapping[str, float] | None = None,
                               cfg: WeightSearchConfig | None = None,
                               top_k: int = 8,
                               min_l1_distance: float = 0.04,
                               categories: Mapping[str, str] | None = None) -> dict:
    """Solve a deterministic continuous weight frontier; NEVER sample weights.

    Each solution is the global optimum of a finite-scenario LP subject to
    historical buy-and-hold MDD <=4% and empirical scenario signed-gap CVaR<=0.
    Exposure floors index a research frontier; they are NOT fitted investment
    parameters. Cash-only is returned as a separate diagnostic benchmark.
    """
    from fundkit.convex_portfolio import PortfolioLPConfig, solve_continuous_weights

    cfg = cfg or WeightSearchConfig()
    cfg.validate()
    if top_k < 1 or min_l1_distance < 0:
        raise ValueError("top_k/min_l1_distance invalid")
    df = _validate_returns(returns, cfg)
    codes = list(df.columns)
    R = df.to_numpy(float)
    n = len(codes)
    estimates = estimate_robust_returns(
        df, categories=categories,
        prior_shrinkage_days=cfg.return_prior_shrinkage_days,
        uncertainty_z=cfg.return_uncertainty_z)
    unknown = (set(caps or {}) | set(fixed_weights or {})).difference(codes)
    if unknown:
        raise ValueError(f"仓位上限/冻结仓位存在未知基金：{sorted(unknown)}")
    lower = np.array([float((fixed_weights or {}).get(c,0.)) for c in codes])
    upper = np.array([float((caps or {}).get(c,cfg.max_single_fund_weight))
                      for c in codes])
    for i, c in enumerate(codes):
        if c in (fixed_weights or {}):
            upper[i] = lower[i]
    if (not np.isfinite(lower).all() or not np.isfinite(upper).all() or
        (lower<0).any() or (upper<lower).any() or (upper>1).any() or
        lower.sum()+cfg.min_cash_weight>1+1e-10):
        raise ValueError("DATA_BLOCKED: invalid frozen positions or asset caps")
    current = np.array([float((current_weights or {}).get(c,0.)) for c in
                        codes+["cash"]])
    if not np.isfinite(current).all() or (current<-1e-9).any() or current.sum()>1+1e-8:
        raise ValueError("DATA_BLOCKED: invalid current portfolio")
    if current.sum()<1-1e-8:
        current[-1]+=1-current.sum()
    lower_returns=np.array([float(estimates[c]["robust_lcb_annual"]) for c in codes])
    scenarios=_scenarios(R,cfg)
    # Require distinct, non-cash investment portfolios for strategy ablation.
    # Do not silently replace failures with a riskless 100% cash allocation.
    floors=[0., .10, .20, .30, .40, .50, .60, .75, .90, .98]
    floors=[x for x in floors if x<=1-cfg.min_cash_weight+1e-10]
    if lower.sum() > 0:
        floors=[max(x,float(lower.sum())) for x in floors]
    floors=list(dict.fromkeys(floors))
    generated=[]; rejected=[]
    for floor in floors:
        lp=solve_continuous_weights(
            R,scenarios,lower_returns,current_weights=current,
            fund_lower=lower,fund_upper=upper,
            config=PortfolioLPConfig(
                drawdown_budget=cfg.max_drawdown,cvar_alpha=.95,
                turnover_cost=cfg.turnover_cost_rate,
                cash_annual_return=cfg.annual_cash_return,
                min_risky_weight=floor,min_cash_weight=cfg.min_cash_weight))
        if lp["status"]!="OPTIMAL":
            rejected.append({"min_risky_weight":floor,"status":lp["status"],
                             "reason":lp.get("reason")})
            continue
        w=np.asarray(lp["weights"],float)
        if w[:-1].sum() < 1e-6:
            continue
        # Independently review the daily constant-weight approximation used
        # by the legacy portfolio risk estimator; do not hide discrepancies.
        hdd,qdd,qret,turn,hsh,sq25,smed=_risk_metrics(
            R,w[None,:],current,cfg,scenarios)
        risk_bound=max(float(hdd[0]),float(qdd[0]),
                       float(lp["historical_mdd"]),
                       float(lp["scenario_mdd_quantile"]))
        entry={
            "weights":{**{c:float(w[i]) for i,c in enumerate(codes)},
                       "cash":float(w[-1])},
            "historical_mdd":float(hdd[0]),
            "scenario_mdd_p95":float(qdd[0]),
            "scenario_return_q25":float(qret[0]),
            "historical_sharpe":float(hsh[0]),
            "scenario_sharpe_q25":float(sq25[0]),
            "scenario_sharpe_median":float(smed[0]),
            "robust_return_lcb":float(lower_returns @ w[:-1]+
                                       cfg.annual_cash_return*w[-1]),
            "turnover_proxy":float(turn[0]),
            "static_risk_feasible":bool(risk_bound<=cfg.max_drawdown+1e-7),
            "lp_buy_hold_historical_mdd":float(lp["historical_mdd"]),
            "lp_buy_hold_scenario_mdd_p95":float(lp["scenario_mdd_quantile"]),
            "lp_scenario_signed_gap_cvar":float(lp["scenario_cvar_violation"]),
            "lp_optimality":lp["status"],
            "lp_iterations":lp["iteration_count"],
            "lp_cuts":lp["cut_count"],
            "frontier_min_risky_weight":floor,
        }
        if not entry["static_risk_feasible"]:
            rejected.append({"min_risky_weight":floor,
                             "status":"REBALANCED_RISK_REVIEW_FAILED",
                             "worst_mdd":risk_bound})
            continue
        if any(np.abs(w-np.array([r["weights"][c] for c in codes+["cash"]])).sum()
               < min_l1_distance for r in generated):
            continue
        generated.append(entry)
    # Keep genuinely different risk/return positions, with the best robust
    # lower return as the primary candidate. No cash endpoint consumes a slot.
    generated.sort(key=lambda r:(-r["robust_return_lcb"],
                                 r["scenario_mdd_p95"],
                                 -r["scenario_sharpe_q25"]))
    selected=generated[:top_k]
    return {
        "status":"CONVEX_RISK_FEASIBLE_FRONTIER" if selected else
                 "NO_NONCASH_RISK_FEASIBLE_CANDIDATE",
        "candidate_count_generated":len(floors),
        "candidate_count_returned":len(selected),
        "frontier_risk_feasible_count":len(generated),
        "static_risk_feasible_count":len(generated),
        "candidate_count_requested":top_k,
        "training_data_start":str(df.index[0].date()),
        "training_data_end":str(df.index[-1].date()),
        "robust_return_estimates":estimates,
        "solver":"HiGHS LP / exact affine drawdown cut separation",
        "model":"BUY_AND_HOLD_FINITE_SCENARIO_CVAR; REBALANCING_REVIEW",
        "rejected":rejected,
        "cash_benchmark":{"weights":{**{c:0. for c in codes},"cash":1.},
                          "research_only":True},
        "concentration_guardrail":{"max_single_fund_weight":cfg.max_single_fund_weight,
                                   "status":"GOVERNANCE_CAP_NOT_EMPIRICALLY_OPTIMIZED"},
        "candidates":selected,
        "research_only":True,
    }
