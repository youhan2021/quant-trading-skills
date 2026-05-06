#!/usr/bin/env python3
"""
Rolling Walk-Forward v3 — Train期选策略，Test期验证
===================================================
每个滚动窗口：
  1. Train期 → 对所有候选策略算IC → 选中IC最优的策略
  2. Test期 → 用选中的策略跑（完全LOCKED）
  3. 汇总所有窗口的Test结果

候选策略生成（75个）：
  - 5单因子 × 3持仓 = 15
  - 10双因子组合 × 3持仓 = 30
  - 10三因子组合 × 3持仓 = 30
  = 75个策略
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
# 3. 生成75个候选策略
# ------------------------------------------------------------------
FACTORS = ['roc20', 'roc60', 'roc120', 'vol20', 'vol60']
TOPN_OPTIONS = [5, 10, 20]

def make_strategies():
    """生成所有候选策略"""
    strategies = []
    # 单因子
    for f in FACTORS:
        for n in TOPN_OPTIONS:
            strategies.append({'name': f, 'factors': [f], 'top_n': n})
    # 双因子
    for f1, f2 in combinations(FACTORS, 2):
        for n in TOPN_OPTIONS:
            strategies.append({'name': f'{f1}+{f2}', 'factors': [f1, f2], 'top_n': n})
    # 三因子
    for f1, f2, f3 in combinations(FACTORS, 3):
        for n in TOPN_OPTIONS:
            strategies.append({'name': f'{f1}+{f2}+{f3}', 'factors': [f1, f2, f3], 'top_n': n})
    return strategies

STRATEGIES = make_strategies()
print(f"候选策略: {len(STRATEGIES)} 个")

# ------------------------------------------------------------------
# 4. 滚动窗口定义
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
# 5. 工具函数
# ------------------------------------------------------------------
def calc_ic(returns, factor_vals):
    mask = ~(returns.isna() | factor_vals.isna())
    if mask.sum() < 5:
        return 0.0
    return float(returns[mask].corr(factor_vals[mask]))

def compute_factor_values(sub_week, pool):
    """计算池子里所有股票在某个时间点的因子值"""
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

def score_strategy(fvals, strat, factor_dirs):
    """计算某只股票的策略得分"""
    score = 0.0
    for fn in strat['factors']:
        if fn not in fvals:
            return 0.0
        direction = factor_dirs.get(fn, 0)
        if direction == 0:
            return 0.0
        # 用rank来避免极值影响
        rank_val = fvals[fn].rank(pct=True)
        score += direction * rank_val
    return score

def eval_strategy_on_train(train_start, train_end, strat):
    """
    在Train期评估一个策略，返回IC均值和胜率
    """
    train_months = monthly[train_start:train_end].index
    monthly_ics = []

    for dt in train_months:
        dt_str = str(dt)[:10]
        sub = week[:dt_str]
        if len(sub) < 120:
            continue

        pool = [c for c in sub.columns if c in prices.columns]
        fvals = compute_factor_values(sub, pool)

        if not all(f in fvals for f in strat['factors']):
            continue

        # 方向判定（基于整个Train期统一方向）
        factor_dirs = {}
        for fn in strat['factors']:
            arr = np.array(fvals[fn])
            frac = np.nanmean(arr > 0) if len(arr) > 0 else 0.5
            factor_dirs[fn] = +1 if frac > 0.5 else -1

        # 计算每只股票的得分
        scores = pd.Series(index=pool, dtype=float)
        for t in pool:
            s = 0.0
            for fn in strat['factors']:
                direction = factor_dirs[fn]
                rank_val = fvals[fn][t] if t in fvals[fn].index else 0
                # 用原始值，不用rank（因子方向已经处理）
                s += direction * (fvals[fn][t] if t in fvals[fn].index else 0)
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

    return {
        'mean_ic': mean_ic,
        'frac': frac,
        'n': len(arr),
        'factor_dirs': factor_dirs,
    }

def run_backtest(tw_start, tw_end, strat, factor_dirs, lookback_weeks=120):
    """在Test期跑回测，返回结果"""
    # 截面因子：tw_start前lookback_weeks周
    hist = week[week.index < tw_start]
    if len(hist) < lookback_weeks:
        return None
    lb_start = str(hist.index[-lookback_weeks])[:10]
    lb_sub = week[lb_start:tw_start]

    pool = [c for c in lb_sub.columns if c in prices.columns]
    price_at_start = prices.loc[tw_start:tw_end].iloc[0]
    pool = [t for t in pool if price_at_start.get(t, 0) > 3]

    fvals = compute_factor_values(lb_sub, pool)
    if not all(f in fvals for f in strat['factors']):
        return None

    # 计算得分
    scores = pd.Series(index=pool, dtype=float)
    for t in pool:
        s = 0.0
        for fn in strat['factors']:
            direction = factor_dirs[fn]
            val = fvals[fn][t] if t in fvals[fn].index else 0
            s += direction * val
        scores[t] = s

    top_n = strat['top_n']
    top_stocks = scores.nlargest(top_n).index.tolist()

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

def run_window(tw_start, tw_end, label, train_start, train_end, is_live=False):
    """运行单个滚动窗口：Train选策略，Test跑回测"""
    print(f"\n{'='*60}")
    print(f"窗口 {label}  |  Train: {train_start[:4]}~{train_end[:4]}  |  Test: {tw_start[:4]}")
    print(f"{'='*60}")

    # ---- Train期：评估所有策略 ----
    print(f"评估 {len(STRATEGIES)} 个候选策略...")
    train_results = []

    for si, strat in enumerate(STRATEGIES):
        res = eval_strategy_on_train(train_start, train_end, strat)
        if res is None:
            continue
        # 策略必须所有因子都有明确方向（frac != 0.5）
        all_known = all(
            (res['frac'] > 0.55 or res['frac'] < 0.45)
            if fn.startswith('roc') else True
            for fn in strat['factors']
        )
        if not all_known and len(strat['factors']) == 1:
            # 单因子必须有明确方向
            train_results.append({
                'strat': strat,
                'mean_ic': res['mean_ic'],
                'frac': res['frac'],
                'n': res['n'],
                'factor_dirs': res['factor_dirs'],
                'usable': res['frac'] > 0.55 or res['frac'] < 0.45,
            })
        else:
            train_results.append({
                'strat': strat,
                'mean_ic': res['mean_ic'],
                'frac': res['frac'],
                'n': res['n'],
                'factor_dirs': res['factor_dirs'],
                'usable': True,
            })

        if (si + 1) % 15 == 0:
            print(f"  {si+1}/{len(STRATEGIES)}...")

    # 过滤可用的策略（必须有有效IC）
    usable = [r for r in train_results if r['usable'] and r['n'] >= 24]
    if not usable:
        print("  ⚠️ 无可用策略，跳过")
        return None

    # 按|IC|排序（既要IC为正，又要有强度）
    # 选IC最强（正或负都行，但负的要用对方向）
    usable_sorted = sorted(usable, key=lambda x: abs(x['mean_ic']), reverse=True)

    # 取Train期IC最优的策略
    best = usable_sorted[0]
    best_strat = best['strat']
    best_dirs = best['factor_dirs']

    print(f"\n  Train期最优策略: {best_strat['name']} (top{best_strat['top_n']})")
    print(f"  Train IC: {best['mean_ic']:+.4f}, 胜率: {best['frac']:.2f}, n={best['n']}")
    print(f"  因子方向: {best_dirs}")

    # ---- Test期：用选中策略跑回测 ----
    test_res = run_backtest(tw_start, tw_end, best_strat, best_dirs)
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

    print(f"\n  结果: Sharpe={sh:.2f}, Ann={ann*100:.1f}%, MaxDD={mdd*100:.1f}%")
    print(f"  SPY:   Sharpe={spy_sh:.2f}")
    print(f"  超额:  {outperformance:+.2f}  {marker}")
    print(f"  选股:  {test_res['stocks']}")

    return {
        'window': label,
        'train_start': train_start,
        'train_end': train_end,
        'test_start': tw_start,
        'test_end': tw_end,
        'train_ic': round(best['mean_ic'], 4),
        'train_frac': round(best['frac'], 3),
        'selected_strategy': f"{best_strat['name']} top{best_strat['top_n']}",
        'sharpe': round(sh, 2),
        'ann': round(ann*100, 1),
        'max_dd': round(mdd*100, 1),
        'spy_sharpe': round(spy_sh, 2),
        'outperformance': round(outperformance, 2),
        'stocks': test_res['stocks'],
        'factor_dirs': best_dirs,
    }

# ------------------------------------------------------------------
# 6. 运行所有滚动窗口
# ------------------------------------------------------------------
print("\n" + "="*60)
print("滚动 Walk-Forward v3 — Train选策略，Test验证")
print("="*60)

all_results = []
for (train_start, train_end, tw_start, tw_end, label) in ROLL_WINDOWS:
    result = run_window(tw_start, tw_end, label, train_start, train_end)
    if result:
        all_results.append(result)

# ------------------------------------------------------------------
# 7. 实盘推荐
# ------------------------------------------------------------------
print("\n\n" + "="*60)
print("实盘推荐 (Train 2020~2024)")
print("="*60)
live_result = run_window(
    LIVE_WINDOW[2], LIVE_WINDOW[3], LIVE_WINDOW[4],
    LIVE_WINDOW[0], LIVE_WINDOW[1],
    is_live=True
)

# ------------------------------------------------------------------
# 8. 汇总
# ------------------------------------------------------------------
print("\n\n" + "="*60)
print("滚动 Walk-Forward 汇总")
print("="*60)
print(f"{'窗口':6s} {'TrainIC':>8} {'Sharpe':>8} {'Ann%':>8} {'MaxDD%':>8} {'SPY':>8} {'超额':>8} {'策略'}")
print('-'*90)
for r in all_results:
    outp = r['outperformance']
    marker = "★" if outp > 0.3 else ("☆" if outp > 0 else "")
    print(f"{r['window']:6s} {r['train_ic']:>+8.4f} {r['sharpe']:8.2f} {r['ann']:7.1f}% {r['max_dd']:7.1f}% {r['spy_sharpe']:8.2f} {outp:>+7.2f}  {marker}  {r['selected_strategy'][:25]}")
print('-'*90)

sharpes = [r['sharpe'] for r in all_results]
spy_sharpes = [r['spy_sharpe'] for r in all_results]
outperformances = [r['outperformance'] for r in all_results]
train_ics = [r['train_ic'] for r in all_results]
print(f"{'平均':6s} {np.mean(train_ics):>+8.4f} {np.mean(sharpes):8.2f}              {np.mean(spy_sharpes):8.2f} {np.mean(outperformances):>+7.2f}")

win_rate = np.mean([o > 0 for o in outperformances])
print(f"胜率: {win_rate:.0%} ({sum([o>0 for o in outperformances])}/{len(outperformances)})")

# Top-1 selection bias校准：75个策略里挑最优，期望IC要打折
print(f"\n注意: Train期从75个策略里挑最优 → selection bias存在")
print(f"期望真实IC ≈ Train_IC * sqrt(1 - 1/75) ≈ Train_IC * 0.993")

if live_result:
    print(f"\n实盘推荐: {live_result['selected_strategy']}")
    print(f"  Train IC: {live_result['train_ic']:+.4f}, 胜率: {live_result['train_frac']:.2f}")
    print(f"  选股: {live_result['stocks']}")

# 保存
output = {
    'backtest': all_results,
    'live': live_result,
}
with open('/tmp/rolling_wfa_v3_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n已保存: /tmp/rolling_wfa_v3_results.json")