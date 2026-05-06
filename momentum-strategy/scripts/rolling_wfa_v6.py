#!/usr/bin/env python3
"""
Rolling Walk-Forward v6 — IPO过滤 + 方向修复 + 多策略选择
=========================================================
修复：
  - IPO过滤：每只股票在Test期起点必须已有足够历史数据
  - 方向判定：用IC符号
  - 用rank代替原始值

运行：python momentum-strategy/scripts/rolling_wfa_v6.py
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
# 3. 获取每只股票的IPO日期
# ------------------------------------------------------------------
print("\n获取IPO日期...")
IPO_DATES = {}
for t in UNIVERSE:
    try:
        info = yf.Ticker(t).info
        ipo_ts = info.get('firstTradeDateEpochGregorian')
        if ipo_ts:
            from datetime import datetime
            ipo_date = datetime.utcfromtimestamp(ipo_ts)
            IPO_DATES[t] = ipo_date
        else:
            # 估算：从价格数据推断
            first_price_date = prices[t].dropna().index[0]
            IPO_DATES[t] = pd.Timestamp(first_price_date).to_pydatetime()
    except:
        first_price_date = prices[t].dropna().index[0]
        IPO_DATES[t] = pd.Timestamp(first_price_date).to_pydatetime()

# IPO早于2010的不需要过滤
print(f"IPO日期获取完成: {sum(1 for d in IPO_DATES.values() if d.year <= 2010)} 只在2010年前IPO")

# ------------------------------------------------------------------
# 4. 预计算月度因子值
# ------------------------------------------------------------------
print("\n预计算月度因子值...")
MONTHS = monthly.index

FACTOR_CONFIGS = [
    ('roc20', 20),
    ('roc60', 60),
    ('roc120', 120),
    ('vol20', 20),
    ('vol60', 60),
]

factor_matrices = {}
for fname, wk in FACTOR_CONFIGS:
    rows = {}
    for dt in MONTHS:
        dt_str = str(dt)[:10]
        sub = week[:dt_str]
        if len(sub) < wk + 5:
            continue
        if fname.startswith('roc'):
            vals = sub.iloc[-1] / sub.iloc[-wk] - 1
        else:  # vol
            ret = sub.pct_change().dropna()
            if len(ret) >= wk:
                vals = ret.rolling(wk).std().iloc[-1]
            else:
                continue
        rows[dt_str] = vals
    if rows:
        factor_matrices[fname] = pd.DataFrame(rows).T

print(f"因子矩阵: {list(factor_matrices.keys())}")

# ------------------------------------------------------------------
# 5. 候选策略
# ------------------------------------------------------------------
STRAT_CONFIGS = []

# 单因子
for fn in ['roc20', 'roc60', 'roc120', 'vol20', 'vol60']:
    for n in [5, 10, 20]:
        STRAT_CONFIGS.append(([fn], n))

# 双因子
for f1 in ['roc20', 'roc60', 'roc120']:
    for f2 in ['vol20', 'vol60']:
        for n in [5, 10, 20]:
            STRAT_CONFIGS.append(([f1, f2], n))

# 三因子
for f1 in ['roc20', 'roc60', 'roc120']:
    for f2 in ['vol20', 'vol60']:
        for f3 in ['roc20', 'roc60', 'roc120']:
            if f3 != f1:
                for n in [5, 10, 20]:
                    STRAT_CONFIGS.append(([f1, f2, f3], n))

print(f"候选策略: {len(STRAT_CONFIGS)} 个")

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

# ------------------------------------------------------------------
# 7. 核心函数
# ------------------------------------------------------------------
def get_valid_tickers(tw_start, min_weeks=120):
    """获取在Test期起点时已有足够历史的股票"""
    from datetime import datetime
    tw_ts = pd.Timestamp(tw_start)
    valid = []
    for t in prices.columns:
        if t not in IPO_DATES:
            continue
        ipo = IPO_DATES[t]
        # 股票IPO日期距离tw_start至少要有min_weeks周的数据
        price_start = prices[t].dropna().index[0]
        # 最保守：要求IPO日期在tw_start之前至少min_weeks周
        if (tw_ts - pd.Timestamp(price_start)).days >= min_weeks * 7:
            valid.append(t)
    return valid

def calc_ic(returns_vec, factor_vec):
    mask = ~(returns_vec.isna() | factor_vec.isna())
    if mask.sum() < 5:
        return 0.0
    return float(returns_vec[mask].corr(factor_vec[mask]))

def get_factor_at_date(fname, dt_str):
    if fname not in factor_matrices:
        return None
    df = factor_matrices[fname]
    if dt_str not in df.index:
        return None
    return df.loc[dt_str]

def eval_strat_on_train(train_start, train_end, strat_factors, valid_pool):
    """评估策略在Train期"""
    train_months = monthly[train_start:train_end].index.tolist()
    monthly_ics = []

    for i in range(len(train_months) - 1):
        dt = train_months[i]
        dt_str = str(dt)[:10]
        next_dt = train_months[i + 1]
        next_str = str(next_dt)[:10]

        if next_str not in monthly.index:
            continue
        mret = monthly.loc[next_str]

        # 获取各因子值
        fvals = {}
        valid = True
        for fn in strat_factors:
            fv = get_factor_at_date(fn, dt_str)
            if fv is None:
                valid = False
                break
            # 只用有效池子里的股票
            fv = fv[valid_pool]
            fvals[fn] = fv

        if not valid:
            continue

        # 计算综合得分（rank平均）
        score = pd.Series(0.0, index=mret.index)
        for fn in strat_factors:
            score += fvals[fn].rank(pct=True)

        ic = calc_ic(mret[valid_pool], score[valid_pool])
        monthly_ics.append(ic)

    if len(monthly_ics) < 24:
        return None

    arr = np.array(monthly_ics)
    mean_ic = np.nanmean(arr)
    frac = np.nanmean(arr > 0)
    direction = +1 if mean_ic > 0 else -1

    return {
        'mean_ic': mean_ic,
        'frac': frac,
        'n': len(arr),
        'direction': direction,
    }

def run_backtest(tw_start, tw_end, strat_factors, direction, top_n, valid_pool, lookback_weeks=120):
    """Test期回测"""
    # 获取截面因子
    hist = week[week.index < tw_start]
    if len(hist) < lookback_weeks:
        return None

    lb_end_idx = -1
    lb_start_idx = -lookback_weeks
    lb_sub = hist[lb_start_idx:]

    pool = [t for t in valid_pool if t in lb_sub.columns]
    price_at_start = prices.loc[tw_start:tw_end].iloc[0]
    pool = [t for t in pool if price_at_start.get(t, 0) > 3]

    if len(pool) < top_n * 2:
        return None

    # 计算截面因子得分
    scores = pd.Series(index=pool, dtype=float)
    for fn in strat_factors:
        if len(lb_sub) < 120:
            continue
        wk = 120 if fn == 'roc120' else (60 if fn == 'roc60' else 20)
        if fn.startswith('roc'):
            fv = (lb_sub.iloc[-1] / lb_sub.iloc[-wk] - 1)[pool]
        else:
            ret = lb_sub[pool].pct_change().dropna()
            vol_wk = 20 if fn == 'vol20' else 60
            fv = ret.rolling(vol_wk).std().iloc[-1]
        # rank
        scores += fv.rank(pct=True)

    # 选股
    if direction == -1:
        top_stocks = scores.nsmallest(top_n).index.tolist()
    else:
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

    return {'sharpe': sh, 'ann': ann, 'max_dd': mdd, 'stocks': top_stocks}

def run_window(tw_start, tw_end, label, train_start, train_end):
    """运行单个窗口"""
    print(f"\n{'='*55}")
    print(f"窗口 {label}  |  Train: {train_start[:4]}~{train_end[:4]}  |  Test: {tw_start[:4]}")
    print(f"{'='*55}")

    # IPO过滤
    valid_pool = get_valid_tickers(tw_start, min_weeks=130)
    print(f"  有效股票(IPO>={tw_start[:4]}年前): {len(valid_pool)} 只")

    # 评估所有策略
    train_results = []
    for si, (factors, top_n) in enumerate(STRAT_CONFIGS):
        res = eval_strat_on_train(train_start, train_end, factors, valid_pool)
        if res is None:
            continue
        if res['n'] < 24:
            continue
        train_results.append({
            'factors': factors,
            'top_n': top_n,
            **res
        })

    if not train_results:
        print("  ⚠️ 无可用策略")
        return None

    # 按|IC|排序，选最优
    train_sorted = sorted(train_results, key=lambda x: abs(x['mean_ic']), reverse=True)
    best = train_sorted[0]

    print(f"  最优策略: {best['factors']} top{best['top_n']}")
    print(f"  Train IC={best['mean_ic']:+.4f} frac={best['frac']:.2f} dir={best['direction']}")

    # 回测
    test_res = run_backtest(tw_start, tw_end, best['factors'], best['direction'],
                            best['top_n'], valid_pool)
    if test_res is None:
        print("  ⚠️ 回测失败")
        return None

    # SPY
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
    outperformance = sh - spy_sh
    marker = "★" if outperformance > 0.3 else ("☆" if outperformance > 0 else "")

    print(f"  结果: Sharpe={sh:.2f}, Ann={test_res['ann']*100:.1f}%, MaxDD={test_res['max_dd']*100:.1f}%")
    print(f"  SPY={spy_sh:.2f}  超额={outperformance:+.2f}  {marker}")
    print(f"  选股: {test_res['stocks'][:3]}...")

    return {
        'window': label,
        'factors': best['factors'],
        'top_n': best['top_n'],
        'direction': best['direction'],
        'train_ic': round(best['mean_ic'], 4),
        'train_frac': round(best['frac'], 3),
        'valid_pool_size': len(valid_pool),
        'sharpe': round(sh, 2),
        'ann': round(test_res['ann']*100, 1),
        'max_dd': round(test_res['max_dd']*100, 1),
        'spy_sharpe': round(spy_sh, 2),
        'outperformance': round(outperformance, 2),
        'stocks': test_res['stocks'],
    }

# ------------------------------------------------------------------
# 8. 运行
# ------------------------------------------------------------------
print("\n" + "="*55)
print("滚动 Walk-Forward v6 — IPO过滤 + 方向修复")
print("="*55)

all_results = []
for (train_start, train_end, tw_start, tw_end, label) in ROLL_WINDOWS:
    result = run_window(tw_start, tw_end, label, train_start, train_end)
    if result:
        all_results.append(result)

# ------------------------------------------------------------------
# 9. 汇总
# ------------------------------------------------------------------
print("\n\n" + "="*55)
print("滚动 Walk-Forward v6 汇总")
print("="*55)
print(f"{'窗口':6s} {'TrainIC':>8} {'Sharpe':>8} {'Ann%':>8} {'MaxDD%':>8} {'SPY':>8} {'超额':>8}")
print('-'*70)
for r in all_results:
    outp = r['outperformance']
    marker = "★" if outp > 0.3 else ("☆" if outp > 0 else "")
    print(f"{r['window']:6s} {r['train_ic']:>+8.4f} {r['sharpe']:8.2f} {r['ann']:7.1f}% {r['max_dd']:7.1f}% {r['spy_sharpe']:8.2f} {outp:>+7.2f}  {marker}")
print('-'*70)

sharpes = [r['sharpe'] for r in all_results]
spy_sharpes = [r['spy_sharpe'] for r in all_results]
outs = [r['outperformance'] for r in all_results]
print(f"{'平均':6s} {np.mean([r['train_ic'] for r in all_results]):>+8.4f} {np.mean(sharpes):8.2f}              {np.mean(spy_sharpes):8.2f} {np.mean(outs):>+7.2f}")

win_rate = np.mean([o > 0 for o in outs])
print(f"胜率: {win_rate:.0%} ({sum([o>0 for o in outs])}/{len(outs)})")

with open('/tmp/rolling_wfa_v6_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n已保存: /tmp/rolling_wfa_v6_results.json")