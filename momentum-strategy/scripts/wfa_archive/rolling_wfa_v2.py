#!/usr/bin/env python3
"""
Rolling Walk-Forward v2 —严格时序分离
=====================================
Train/Test严格滚动：
  Train 2009~2013 → Test 2014
  Train 2010~2014 → Test 2015
  Train 2011~2015 → Test 2016
  Train 2012~2016 → Test 2017
  Train 2013~2017 → Test 2018
  Train 2014~2018 → Test 2019
  Train 2015~2019 → Test 2020
  Train 2016~2020 → Test 2021
  Train 2017~2021 → Test 2022
  Train 2018~2022 → Test 2023
  Train 2019~2023 → Test 2024
  实盘: Train 2020~2024 → 推荐2025

每个窗口：
  1. Train期内做IC分析 → 确定因子方向和权重
  2. Test期用LOCKED权重选股，不调整参数
  3. 每个窗口独立，不重算历史
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, warnings
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
# 3. 滚动窗口定义
# ------------------------------------------------------------------
# 5年Train → 1年Test
ROLL_WINDOWS = [
    # (train_start, train_end, test_start, test_end, label)
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

# 实盘窗口（用最新的Train数据）
LIVE_WINDOW = ('2020-01-01', '2025-01-01', '2025-01-01', '2026-01-01', 'LIVE')

def calc_ic(returns, factor_vals):
    """计算当期IC（截面相关性）"""
    mask = ~(returns.isna() | factor_vals.isna())
    if mask.sum() < 5:
        return 0.0
    return float(returns[mask].corr(factor_vals[mask]))

def run_window(tw_start, tw_end, label, train_start, train_end,
               factor_dir=None, factor_weights=None, is_live=False):
    """运行单个滚动窗口"""
    print(f"\n{'='*55}")
    print(f"窗口 {label}  |  Train: {train_start[:4]}~{train_end[:4]}  |  Test: {tw_start[:4]}")
    print(f"{'='*55}")

    # ----- Train期：计算因子IC -----
    train_months = monthly[train_start:train_end].index
    factor_ics = {f: [] for f in ['roc20', 'roc60', 'roc120', 'vol20', 'vol60']}

    for dt in train_months:
        dt_str = str(dt)[:10]
        sub = week[:dt_str]
        if len(sub) < 120:
            continue

        # 计算因子值
        fvals = {}
        for wk in [20, 60, 120]:
            if len(sub) >= wk:
                fvals[f'roc{wk}'] = (sub.iloc[-1] / sub.iloc[-wk] - 1)
        ret = sub.pct_change().dropna()
        for wk in [20, 60]:
            if len(ret) >= wk:
                fvals[f'vol{wk}'] = ret.rolling(wk).std().iloc[-1]

        # 下月收益
        try:
            next_dt = monthly.index[monthly.index.get_loc(dt) + 1]
            mret = monthly.loc[str(next_dt)[:10]]
        except:
            continue

        for fn in fvals:
            ic = calc_ic(mret, fvals[fn])
            factor_ics[fn].append(ic)

    # 汇总IC统计
    if not is_live:
        print("IC统计 (Train期):")
    else:
        print("IC统计 (实盘Train期):")

    discovered_dir = {}
    for fn, ics in factor_ics.items():
        arr = np.array(ics)
        mean_ic = np.nanmean(arr)
        frac = np.nanmean(arr > 0)
        n = len(arr)
        status = "✓" if n >= 30 and frac > 0.55 else "✗"
        print(f"  {fn:8s}: IC={mean_ic:+.4f}, frac={frac:.2f}, n={n:3d}  {status}")
        if n >= 30:
            if frac > 0.55:
                discovered_dir[fn] = +1
            elif frac < 0.45:
                discovered_dir[fn] = -1

    if not discovered_dir:
        print("  ⚠️ 无有效因子，跳过")
        return None

    print(f"  有效因子: {discovered_dir}")

    # 用Train期IC均值作为权重
    weights = {}
    for fn in discovered_dir:
        weights[fn] = abs(np.nanmean(factor_ics[fn]))
    total_w = sum(weights.values())
    for fn in weights:
        weights[fn] /= total_w
    print(f"  因子权重: {', '.join([f'{fn}={w:.2f}' for fn,w in weights.items()])}")

    # ----- Test期：LOCKED权重选股 -----
    # 用Test期起点（tw_start）那一刻的截面因子
    # ---- 截面因子得分：用tw_start前120周的数据 ----
    lookback_end = tw_start
    # 取 tw_start 之前最后120周
    hist_for_lookback = week[week.index < lookback_end]
    if len(hist_for_lookback) < 120:
        print(f"  ⚠️ 历史数据不足（{len(hist_for_lookback)} < 120周），跳过")
        return None
    lookback_start = str(hist_for_lookback.index[-120])[:10]
    lb_sub = week[lookback_start:lookback_end]

    pool = [c for c in lb_sub.columns if c in prices.columns]
    # 价格filter：排除太便宜的
    price_at_start = prices.loc[tw_start:tw_end].iloc[0]
    pool = [t for t in pool if price_at_start.get(t, 0) > 3]

    print(f"  候选池: {len(pool)} 只")
    scores = pd.DataFrame(index=pool)

    for wk in [20, 60, 120]:
        if len(lb_sub) >= wk:
            scores[f'roc{wk}'] = (lb_sub.iloc[-1] / lb_sub.iloc[-wk] - 1)
    ret = lb_sub.pct_change().dropna()
    for wk in [20, 60]:
        if len(ret) >= wk:
            scores[f'vol{wk}'] = ret.rolling(wk).std().iloc[-1]

    # 综合得分
    composite = pd.Series(0.0, index=pool)
    for fn in discovered_dir:
        if fn in scores.columns:
            direction = discovered_dir[fn]
            w = weights[fn]
            composite += direction * w * scores[fn].fillna(0)

    # 选 top5
    top5 = composite.nlargest(5).index.tolist()
    print(f"  选股: {top5}")

    # Test期回测
    test_prices = prices[top5][tw_start:tw_end].resample('W').last()
    tw_ret = test_prices.pct_change().dropna().mean(axis=1)
    ann = float((1 + tw_ret.mean())**52 - 1)
    vol = float(tw_ret.std() * np.sqrt(52))
    sh = ann / vol if vol > 0 else 0
    cum = (1+tw_ret).cumprod()
    mdd = float((cum/cum.cummax()-1).min())

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

    outperformance = sh - spy_sh
    marker = "★" if outperformance > 0.3 else ("☆" if outperformance > 0 else "")
    print(f"  结果: Sharpe={sh:.2f}, Ann={ann*100:.1f}%, MaxDD={mdd*100:.1f}%  |  SPY={spy_sh:.2f}  {marker}")

    return {
        'window': label,
        'train_start': train_start,
        'train_end': train_end,
        'test_start': tw_start,
        'test_end': tw_end,
        'sharpe': round(sh, 2),
        'ann': round(ann*100, 1),
        'max_dd': round(mdd*100, 1),
        'spy_sharpe': round(spy_sh, 2),
        'outperformance': round(outperformance, 2),
        'stocks': top5,
        'factor_dir': discovered_dir,
        'weights': {k: round(v, 4) for k, v in weights.items()},
    }

# ------------------------------------------------------------------
# 4. 运行所有滚动窗口
# ------------------------------------------------------------------
all_results = []
for (train_start, train_end, tw_start, tw_end, label) in ROLL_WINDOWS:
    result = run_window(tw_start, tw_end, label, train_start, train_end)
    if result:
        all_results.append(result)

# ------------------------------------------------------------------
# 5. 实盘推荐（用2020~2024训练，推荐2025）
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
# 6. 汇总
# ------------------------------------------------------------------
print("\n\n" + "="*60)
print("滚动 Walk-Forward 汇总")
print("="*60)
print(f"{'窗口':6s} {'Sharpe':>8} {'Ann%':>8} {'MaxDD%':>8} {'SPY':>8} {'超额':>8} {'选股'}")
print('-'*75)
for r in all_results:
    outp = r['outperformance']
    marker = "★" if outp > 0.3 else ("☆" if outp > 0 else "")
    print(f"{r['window']:6s} {r['sharpe']:8.2f} {r['ann']:7.1f}% {r['max_dd']:7.1f}% {r['spy_sharpe']:8.2f} {outp:+.2f}  {marker}  {','.join(r['stocks'][:3])}")
print('-'*75)

sharpes = [r['sharpe'] for r in all_results]
spy_sharpes = [r['spy_sharpe'] for r in all_results]
outperformances = [r['outperformance'] for r in all_results]
print(f"{'平均':6s} {np.mean(sharpes):8.2f}              {np.mean(spy_sharpes):8.2f} {np.mean(outperformances):+.2f}")
win_rate = np.mean([o > 0 for o in outperformances])
print(f"胜率: {win_rate:.0%} ({sum([o>0 for o in outperformances])}/{len(outperformances)})")

if live_result:
    print(f"\n实盘推荐: {live_result['stocks']}")
    print(f"  因子权重: {live_result['weights']}")

# 保存
output = {
    'backtest': all_results,
    'live': live_result,
}
with open('/tmp/rolling_wfa_v2_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n已保存: /tmp/rolling_wfa_v2_results.json")