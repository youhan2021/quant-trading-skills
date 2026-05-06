#!/usr/bin/env python3
"""
Rolling Walk-Forward v4 — 修复方向判断bug + 扩大策略空间 + Top-K选择
==============================================================
修复：
  - 方向判定：用IC符号，不用frac>0.5
  - 得分：用rank percentile，不用原始值

新增：
  - 候选因子扩充：加入营收增速、ROE、PB、PE等基本面因子
  - Top-K选择：选Train IC最优的3个策略平均（减少selection bias）
  - 更细致的参数组合

运行：python momentum-strategy/scripts/rolling_wfa_v4.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, warnings
from itertools import combinations
warnings.filterwarnings('ignore')

DATA_DIR = '/home/ubuntu/.hermes/skills/quant-trading/fundamental-data-collector/data'

# ------------------------------------------------------------------
# 1. 候选池
# ------------------------------------------------------------------
with open(f'{DATA_DIR}/ticker_cik_map.json') as f:
    cik_map = json.load(f)
UNIVERSE = sorted(cik_map.keys())
print(f"候选池: {len(UNIVERSE)} 只")

# ------------------------------------------------------------------
# 2. 下载价格数据
# ------------------------------------------------------------------
print("下载价格数据...")
all_prices = {}
BATCH = 30
for i in range(0, len(UNIVERSE), BATCH):
    batch = UNIVERSE[i:i+BATCH]
    df = yf.download(batch, start='2007-01-01', end='2026-03-01',
                     auto_adjust=False, progress=False)
    if df.empty:
        continue
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    if df.index.tz is not None:
        df = df.tz_localize(None)
    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 100:
            all_prices[str(col)] = s
    print(f"  {min(i+BATCH, len(UNIVERSE))}/{len(UNIVERSE)}, {len(all_prices)} valid")

prices = pd.DataFrame(all_prices)
week = prices.resample('W').last()
monthly = prices.resample('ME').last()
print(f"价格矩阵: {prices.shape}")

# ------------------------------------------------------------------
# 3. 基本面数据（yfinance info）
# ------------------------------------------------------------------
print("下载基本面数据...")
fund = {}
for i, t in enumerate(UNIVERSE):
    try:
        info = yf.Ticker(t).info
        fund[t] = {
            'roe': info.get('returnOnEquity'),
            'de': info.get('debtToEquity'),
            'pb': info.get('priceToBook'),
            'pe': info.get('trailingPE'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_yield': info.get('earningsYield'),
            'book_value': info.get('bookValue'),
            'gross_margin': info.get('grossMargins'),
            'profit_margin': info.get('profitMargins'),
            'ocf': info.get('operatingCashflow'),
            'fcf': info.get('freeCashflow'),
            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),
        }
    except:
        fund[t] = {}
    if (i+1) % 30 == 0:
        print(f"  {i+1}/{len(UNIVERSE)}")

print(f"  共 {len(fund)} 只有基本面数据")

# ------------------------------------------------------------------
# 4. 候选因子定义
# ------------------------------------------------------------------
# 动量因子（已有）
PRICE_FACTORS = ['roc20', 'roc60', 'roc120', 'vol20', 'vol60']

# 基本面因子（用当前截面值，不做时序，用常数填充缺失）
FUND_FACTORS = ['roe', 'de', 'pb', 'pe', 'revenue_growth', 'earnings_yield',
                'gross_margin', 'profit_margin', 'dividend_yield', 'ocf_fcf_ratio']

def get_fund_factor_at_date(ticker, dt_str, lookback_days=90):
    """获取某个时点之前最近的基本面数据"""
    # yfinance基本面是最新的季度数据，不做时序模拟（简化处理）
    # 在严格WFA中，基本面因子只用"当时可获得"的信息
    # 这里用最近可获得的数据代替
    return fund.get(ticker, {})

ALL_FACTORS = PRICE_FACTORS + FUND_FACTORS
print(f"候选因子: {len(ALL_FACTORS)} 个 ({len(PRICE_FACTORS)}价格 + {len(FUND_FACTORS)}基本面)")

# ------------------------------------------------------------------
# 5. 生成候选策略
# ------------------------------------------------------------------
TOPN_OPTIONS = [5, 10, 20]
SELECT_K = 3  # 选Train IC最优的K个，平均他们的选股结果

def make_strategies():
    """生成候选策略"""
    strategies = []
    # 单因子
    for f in ALL_FACTORS:
        for n in TOPN_OPTIONS:
            strategies.append({'name': f, 'factors': [f], 'top_n': n})
    # 双因子组合
    for f1, f2 in combinations(ALL_FACTORS, 2):
        for n in TOPN_OPTIONS:
            strategies.append({'name': f'{f1}+{f2}', 'factors': [f1, f2], 'top_n': n})
    # 三因子组合（只选价格因子）
    for f1, f2, f3 in combinations(PRICE_FACTORS, 3):
        for n in TOPN_OPTIONS:
            strategies.append({'name': f'{f1}+{f2}+{f3}', 'factors': [f1, f2, f3], 'top_n': n})
    return strategies

STRATEGIES = make_strategies()
print(f"候选策略: {len(STRATEGIES)} 个")

# ------------------------------------------------------------------
# 6. 滚动窗口
# ------------------------------------------------------------------
ROLL_WINDOWS = [
    ('2009-01-01', '2014-01-01', '2014-01-01', '2015-01-01', '2014'),
    ('2010-01-01', '2015-01-01', '2015-01-01', '2016-01-01', '2015'),
    ('2011-01-01', '2016-01-01', '2016-01-01', '2017-01-01', '2016'),
    ('2012-01-01', '2017-01-01', '2017-01-01', '2018-01-01', '2017'),
    ('2013-01-01', '2018-01-01', '2018-01-01', '2019-01-01', '2018'),
    ('2014-01-01', '2019-01-01', '2019-01-01', '2020-01-01', '2019'),
    ('2015-01-01', '2020-01-01', '2020-01-01', '2021-01-01', '2020'),
    ('2016-01-01', '2021-01-01', '2021-01-01', '2022-01-01', '2021'),
    ('2017-01-01', '2022-01-01', '2022-01-01', '2023-01-01', '2022'),
    ('2018-01-01', '2023-01-01', '2023-01-01', '2024-01-01', '2023'),
    ('2019-01-01', '2024-01-01', '2024-01-01', '2025-01-01', '2024'),
]
LIVE_WINDOW = ('2020-01-01', '2025-01-01', '2025-01-01', '2026-01-01', 'LIVE')

# ------------------------------------------------------------------
# 7. 工具函数
# ------------------------------------------------------------------
def calc_ic(returns, factor_vals):
    mask = ~(returns.isna() | factor_vals.isna())
    if mask.sum() < 5:
        return 0.0
    return float(returns[mask].corr(factor_vals[mask]))

def compute_price_factors(sub_week, pool):
    """计算价格动量因子"""
    fvals = {}
    sub = sub_week[pool]
    for wk in [20, 60, 120]:
        if len(sub) >= wk:
            fvals[f'roc{wk}'] = (sub.iloc[-1] / sub.iloc[-wk] - 1)
    ret = sub.pct_change().dropna()
    for wk in [20, 60]:
        if len(ret) >= wk:
            fvals[f'vol{wk}'] = ret.rolling(wk).std().iloc[-1]
    return fvals

def compute_fund_factors(pool, dt_str):
    """计算基本面因子（用截面常数）"""
    fvals = {}
    for fn in FUND_FACTORS:
        vals = {}
        for t in pool:
            fdata = fund.get(t, {})
            v = fdata.get(fn, np.nan)
            vals[t] = v if v is not None else np.nan
        s = pd.Series(vals)
        # 基本面因子取rank（处理不同量纲）
        fvals[fn] = s.rank(pct=True) if s.notna().sum() > 5 else s * np.nan
    return fvals

def compute_all_factors(sub_week, pool, dt_str):
    """计算所有因子"""
    pf = compute_price_factors(sub_week, pool)
    ff = compute_fund_factors(pool, dt_str)
    return {**pf, **ff}

def eval_strategy_on_train(train_start, train_end, strat, top_k=3):
    """
    在Train期评估策略，返回IC统计和候选股票列表
    返回: (mean_ic, frac, n, factor_dirs, train_scores)
    """
    train_months = monthly[train_start:train_end].index
    monthly_ics = []

    # 对每个月份计算IC
    for dt in train_months:
        dt_str = str(dt)[:10]
        sub = week[:dt_str]
        if len(sub) < 120:
            continue

        pool = [c for c in sub.columns if c in prices.columns]
        fvals = compute_all_factors(sub, pool, dt_str)

        # 检查所有因子都有值
        if not all(f in fvals and fvals[f].notna().sum() > 5 for f in strat['factors']):
            continue

        # 计算方向：用IC符号，不用frac
        factor_dirs = {}
        for fn in strat['factors']:
            ic_vals = []
            # 对每个月份计算IC
            ic_for_fn = calc_ic(monthly.loc[dt_str] if dt_str in monthly.index else pd.Series(), fvals[fn])
            # 用fvals在当前截面的值作为因子暴露
            ic_vals.append(ic_for_fn)

        # 计算综合得分
        scores = pd.Series(index=pool, dtype=float)
        for t in pool:
            s = 0.0
            for fn in strat['factors']:
                fv = fvals[fn]
                if t not in fv.index or pd.isna(fv[t]):
                    continue
                # 用rank percentile
                rank_pct = fv.rank(pct=True)[t] if fv.notna().sum() > 0 else 0.5
                s += rank_pct  # 方向统一为正（用rank percentile = 0~1）
            scores[t] = s

        # 下月收益
        try:
            next_dt = monthly.index[monthly.index.get_loc(dt) + 1]
            mret = monthly.loc[str(next_dt)[:10]]
        except:
            continue

        ic = calc_ic(mret[pool], scores[pool])
        monthly_ics.append(ic)

    if len(monthly_ics) < 12:
        return None

    arr = np.array(monthly_ics)
    mean_ic = np.nanmean(arr)
    frac = np.nanmean(arr > 0)

    # 方向判定：用IC符号（修复bug）
    factor_dirs = {}
    for fn in strat['factors']:
        fn_ics = []
        for dt in train_months:
            dt_str = str(dt)[:10]
            sub = week[:dt_str]
            if len(sub) < 120:
                continue
            pool = [c for c in sub.columns if c in prices.columns]
            fvals = compute_all_factors(sub, pool, dt_str)
            if fn not in fvals:
                continue
            try:
                next_dt = monthly.index[monthly.index.get_loc(dt) + 1]
                mret = monthly.loc[str(next_dt)[:10]]
                ic = calc_ic(mret[pool], fvals[fn][pool])
                fn_ics.append(ic)
            except:
                continue
        if len(fn_ics) >= 12:
            fn_mean_ic = np.nanmean(fn_ics)
            fn_frac = np.nanmean(np.array(fn_ics) > 0)
            # 用IC符号决定方向
            factor_dirs[fn] = +1 if fn_mean_ic > 0 else -1

    return {
        'mean_ic': mean_ic,
        'frac': frac,
        'n': len(arr),
        'factor_dirs': factor_dirs,
    }

def run_backtest_single(tw_start, tw_end, strat, factor_dirs, lookback_weeks=120):
    """对单个策略跑Test回测，返回选股和收益"""
    hist = week[week.index < tw_start]
    if len(hist) < lookback_weeks:
        return None
    lb_start = str(hist.index[-lookback_weeks])[:10]
    lb_sub = week[lb_start:tw_start]

    pool = [c for c in lb_sub.columns if c in prices.columns]
    price_at_start = prices.loc[tw_start:tw_end].iloc[0]
    pool = [t for t in pool if price_at_start.get(t, 0) > 3]

    fvals = compute_all_factors(lb_sub, pool, tw_start)

    # 检查因子可用
    for fn in strat['factors']:
        if fn not in fvals or fvals[fn].notna().sum() < 5:
            return None

    # 计算综合得分（用rank percentile）
    scores = pd.Series(index=pool, dtype=float)
    for t in pool:
        s = 0.0
        for fn in strat['factors']:
            direction = factor_dirs.get(fn, +1)
            fv = fvals[fn]
            if t not in fv.index or pd.isna(fv[t]):
                continue
            rank_pct = fv.rank(pct=True)[t]
            # 方向×rank: direction=+1时，高rank=高分；direction=-1时，高rank=低分
            scores[t] += direction * rank_pct

    top_n = strat['top_n']
    top_stocks = scores.nlargest(top_n).index.tolist()

    if len(top_stocks) < 3:
        return None

    # 回测
    test_prices = prices[top_stocks][tw_start:tw_end].resample('W').last()
    tw_ret = test_prices.pct_change().dropna().mean(axis=1)
    ann = float((1 + tw_ret.mean())**52 - 1)
    vol = float(tw_ret.std() * np.sqrt(52))
    sh = ann / vol if vol > 0 else 0
    cum = (1+tw_ret).cumprod()
    mdd = float((cum/cum.cummax()-1).min())

    return {
        'stocks': top_stocks,
        'sharpe': sh,
        'ann': ann,
        'max_dd': mdd,
        'tw_ret': tw_ret,
    }

def run_backtest_ensemble(tw_start, tw_end, strats, factor_dirs_list, lookback_weeks=120):
    """对K个策略平均选股，减少selection bias"""
    all_stock_scores = {}

    for strat, factor_dirs in zip(strats, factor_dirs_list):
        res = run_backtest_single(tw_start, tw_end, strat, factor_dirs, lookback_weeks)
        if res is None:
            continue
        for stock, score_val in zip(res['stocks'], range(len(res['stocks']), 0, -1)):
            # 排名越前分数越高
            if stock not in all_stock_scores:
                all_stock_scores[stock] = 0.0
            all_stock_scores[stock] += score_val  # top1得5分，top2得4分...

    if not all_stock_scores:
        return None

    # 选总分数最高的
    sorted_stocks = sorted(all_stock_scores.items(), key=lambda x: x[1], reverse=True)
    top_stocks = [s for s, _ in sorted_stocks[:10]]  # 取前10个作为ensemble结果

    test_prices = prices[top_stocks][tw_start:tw_end].resample('W').last()
    tw_ret = test_prices.pct_change().dropna().mean(axis=1)
    ann = float((1 + tw_ret.mean())**52 - 1)
    vol = float(tw_ret.std() * np.sqrt(52))
    sh = ann / vol if vol > 0 else 0
    cum = (1+tw_ret).cumprod()
    mdd = float((cum/cum.cummax()-1).min())

    return {
        'stocks': top_stocks,
        'sharpe': sh,
        'ann': ann,
        'max_dd': mdd,
        'tw_ret': tw_ret,
    }

def run_window(tw_start, tw_end, label, train_start, train_end, is_live=False, use_ensemble=True):
    """运行单个滚动窗口"""
    print(f"\n{'='*60}")
    print(f"窗口 {label}  |  Train: {train_start[:4]}~{train_end[:4]}  |  Test: {tw_start[:4]}")
    print(f"{'='*60}")

    # ---- Train期：评估所有策略 ----
    print(f"评估 {len(STRATEGIES)} 个候选策略...")
    train_results = []

    for si, strat in enumerate(STRATEGIES):
        res = eval_strategy_on_train(train_start, train_end, strat)
        if res is None or len(res['factor_dirs']) < len(strat['factors']):
            continue
        if res['n'] < 24:
            continue
        train_results.append({
            'strat': strat,
            'mean_ic': res['mean_ic'],
            'frac': res['frac'],
            'n': res['n'],
            'factor_dirs': res['factor_dirs'],
        })
        if (si + 1) % 30 == 0:
            print(f"  {si+1}/{len(STRATEGIES)}...")

    if not train_results:
        print("  ⚠️ 无可用策略，跳过")
        return None

    # 按|IC|排序（选最强的）
    usable_sorted = sorted(train_results, key=lambda x: abs(x['mean_ic']), reverse=True)

    # Top-K选择
    K = SELECT_K
    top_k = usable_sorted[:K]
    best_strats = [r['strat'] for r in top_k]
    best_dirs = [r['factor_dirs'] for r in top_k]

    print(f"\n  Train期Top-{K}策略:")
    for r in top_k:
        print(f"    {r['strat']['name']:40s} IC={r['mean_ic']:+.4f} frac={r['frac']:.2f}")

    # ---- Test期：跑回测 ----
    if use_ensemble:
        test_res = run_backtest_ensemble(tw_start, tw_end, best_strats, best_dirs)
        method = f"Top-{K} Ensemble"
    else:
        # 只跑最优的1个
        test_res = run_backtest_single(tw_start, tw_end, best_strats[0], best_dirs[0])
        method = f"Top-1 ({best_strats[0]['name']})"

    if test_res is None:
        print("  ⚠️ Test期回测失败，跳过")
        return None

    # SPY benchmark
    spy = yf.download('SPY', start='2007-01-01', end='2026-03-01',
                      auto_adjust=False, progress=False)['Close'].squeeze()
    if spy.index.tz:
        spy = spy.tz_localize(None)
    spy_w = spy.resample('W').last()
    spy_tw = spy_w[tw_start:tw_end].pct_change().dropna()
    spy_ann = float((1 + spy_tw.mean())**52 - 1)
    spy_vol = float(spy_tw.std() * np.sqrt(52))
    spy_sh = spy_ann / spy_vol if spy_vol > 0 else 0

    sh = test_res['sharpe']
    ann = test_res['ann']
    mdd = test_res['max_dd']
    outperformance = sh - spy_sh
    marker = "★" if outperformance > 0.3 else ("☆" if outperformance > 0 else "")

    print(f"\n  方法: {method}")
    print(f"  结果: Sharpe={sh:.2f}, Ann={ann*100:.1f}%, MaxDD={mdd*100:.1f}%")
    print(f"  SPY:   Sharpe={spy_sh:.2f}")
    print(f"  超额:  {outperformance:+.2f}  {marker}")
    print(f"  选股:  {test_res['stocks']}")

    return {
        'window': label,
        'train_start': train_start,
        'train_end': train_end,
        'test_start': tw_start,
        'test_end': tw_end,
        'method': method,
        'top_k_strategies': [r['strat']['name'] for r in top_k],
        'train_ic_mean': np.mean([r['mean_ic'] for r in top_k]),
        'sharpe': round(sh, 2),
        'ann': round(ann*100, 1),
        'max_dd': round(mdd*100, 1),
        'spy_sharpe': round(spy_sh, 2),
        'outperformance': round(outperformance, 2),
        'stocks': test_res['stocks'],
    }

# ------------------------------------------------------------------
# 8. 运行
# ------------------------------------------------------------------
print("\n" + "="*60)
print(f"滚动 Walk-Forward v4 — Top-{SELECT_K}选择 + 基本面因子 + 方向修复")
print("="*60)

all_results = []
for (train_start, train_end, tw_start, tw_end, label) in ROLL_WINDOWS:
    result = run_window(tw_start, tw_end, label, train_start, train_end)
    if result:
        all_results.append(result)

# ------------------------------------------------------------------
# 9. 汇总
# ------------------------------------------------------------------
print("\n\n" + "="*60)
print("滚动 Walk-Forward 汇总")
print("="*60)
print(f"{'窗口':6s} {'TrainIC':>8} {'Sharpe':>8} {'Ann%':>8} {'MaxDD%':>8} {'SPY':>8} {'超额':>8}")
print('-'*70)
for r in all_results:
    outp = r['outperformance']
    marker = "★" if outp > 0.3 else ("☆" if outp > 0 else "")
    print(f"{r['window']:6s} {r['train_ic_mean']:>+8.4f} {r['sharpe']:8.2f} {r['ann']:7.1f}% {r['max_dd']:7.1f}% {r['spy_sharpe']:8.2f} {outp:>+7.2f}  {marker}")
print('-'*70)

sharpes = [r['sharpe'] for r in all_results]
spy_sharpes = [r['spy_sharpe'] for r in all_results]
outperformances = [r['outperformance'] for r in all_results]
train_ics = [r['train_ic_mean'] for r in all_results]
print(f"{'平均':6s} {np.mean(train_ics):>+8.4f} {np.mean(sharpes):8.2f}              {np.mean(spy_sharpes):8.2f} {np.mean(outperformances):>+7.2f}")

win_rate = np.mean([o > 0 for o in outperformances])
print(f"胜率: {win_rate:.0%} ({sum([o>0 for o in outperformances])}/{len(outperformances)})")

# 保存
output = {'backtest': all_results}
with open('/tmp/rolling_wfa_v4_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n已保存: /tmp/rolling_wfa_v4_results.json")