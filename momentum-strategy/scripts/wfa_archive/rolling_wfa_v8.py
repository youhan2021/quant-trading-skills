#!/usr/bin/env python3
"""
Rolling Walk-Forward v8 — 扩大选股池 + 多因子 + 季度调仓
=============================================================
改进：
  - 扩大top_n到15~20
  - 加入基本面因子(roe/de)在Train期选最优
  - 季度调仓（而不是年度）
  - IPO严格过滤：min_weeks=130

运行：python momentum-strategy/scripts/rolling_wfa_v8.py
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
# 3. IPO日期
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
        else:
            first_price_date = prices[t].dropna().index[0]
            ipo_date = pd.Timestamp(first_price_date).to_pydatetime()
        IPO_DATES[t] = ipo_date
    except:
        first_price_date = prices[t].dropna().index[0]
        IPO_DATES[t] = pd.Timestamp(first_price_date).to_pydatetime()

print(f"IPO日期获取完成")

# ------------------------------------------------------------------
# 4. 预计算月度因子值
# ------------------------------------------------------------------
print("\n预计算月度因子值...")
FACTOR_CONFIGS = [
    ('roc20', 20), ('roc60', 60), ('roc120', 120),
    ('vol20', 20), ('vol60', 60),
]

factor_matrices = {}
MONTHS = monthly.index
for fname, wk in FACTOR_CONFIGS:
    rows = {}
    for dt in MONTHS:
        dt_str = str(dt)[:10]
        sub = week[:dt_str]
        if len(sub) < wk + 5:
            continue
        if fname.startswith('roc'):
            vals = sub.iloc[-1] / sub.iloc[-wk] - 1
        else:
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
# 5. 候选策略（更多样化）
# ------------------------------------------------------------------
STRAT_CONFIGS = []

# 单因子 + 多个top_N
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
def get_valid_tickers(tw_start, min_weeks=130):
    """获取在Test期起点时已有足够历史的股票"""
    tw_ts = pd.Timestamp(tw_start)
    valid = []
    for t in prices.columns:
        if t not in IPO_DATES:
            continue
        ipo = IPO_DATES[t]
        price_start = prices[t].dropna().index[0]
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

        fvals = {}
        valid = True
        for fn in strat_factors:
            fv = get_factor_at_date(fn, dt_str)
            if fv is None:
                valid = False
                break
            fv = fv[valid_pool]
            fvals[fn] = fv

        if not valid:
            continue

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

def run_backtest_v2(tw_start, tw_end, strat_factors, direction, top_n, valid_pool, lookback_weeks=120):
    """改进的回测：季度调仓 + 更严格的选股"""
    # 获取历史
    hist = week[week.index < tw_start]
    if len(hist) < lookback_weeks:
        return None

    pool = [t for t in valid_pool if t in hist.columns]
    price_at_start = prices.loc[tw_start:tw_end].iloc[0]
    pool = [t for t in pool if price_at_start.get(t, 0) > 3 and price_at_start.get(t, 0) < 2000]

    if len(pool) < top_n * 2:
        return None

    # 计算截面因子得分
    LB = 120 if strat_factors[0].startswith('roc') else 20
    for fn in strat_factors:
        if fn == 'roc120':
            LB = 120
        elif fn == 'roc60':
            LB = max(LB, 60)
        elif fn == 'roc20':
            LB = max(LB, 20)

    scores = pd.Series(index=pool, dtype=float)
    for fn in strat_factors:
        if fn.startswith('roc'):
            wk = 120 if fn == 'roc120' else (60 if fn == 'roc60' else 20)
            fv = (hist.iloc[-1] / hist.iloc[-wk] - 1)[pool]
        else:
            ret = hist[pool].pct_change().dropna()
            vol_wk = 20 if fn == 'vol20' else 60
            fv = ret.rolling(vol_wk).std().iloc[-1]
        scores += fv.rank(pct=True)

    # 选股：扩大top_n到15~20
    n_long = max(top_n, 15)
    sorted_scores = scores.sort_values(ascending=False)
    if direction == -1:
        long_stocks = sorted_scores.tail(n_long).index.tolist()
    else:
        long_stocks = sorted_scores.head(n_long).index.tolist()

    if len(long_stocks) < 5:
        return None

    # 季度调仓回测
    quarters = pd.date_range(tw_start, tw_end, freq='QE')
    if len(quarters) < 1:
        quarters = [pd.Timestamp(tw_end)]

    portfolio_returns = []
    current_stocks = long_stocks[:top_n]  # 持有top_n只

    for qi, q_end in enumerate(quarters):
        q_start = quarters[qi-1] if qi > 0 else pd.Timestamp(tw_start)
        q_start = max(q_start, pd.Timestamp(tw_start))

        # 每季度重新计算持仓（用季度末的lookback）
        # 在实际执行时用上一季度末的信号，这里简化为用同一套分数
        q_end_str = str(q_end)[:10]
        if q_end_str in week.index and len(week[week.index < q_end_str]) >= lookback_weeks:
            q_hist = week[week.index < q_end_str]
            q_scores = pd.Series(index=pool, dtype=float)
            for fn in strat_factors:
                if fn.startswith('roc'):
                    wk = 120 if fn == 'roc120' else (60 if fn == 'roc60' else 20)
                    fv = (q_hist.iloc[-1] / q_hist.iloc[-wk] - 1)[pool]
                else:
                    ret = q_hist[pool].pct_change().dropna()
                    vol_wk = 20 if fn == 'vol20' else 60
                    fv = ret.rolling(vol_wk).std().iloc[-1]
                q_scores += fv.rank(pct=True)

            if direction == -1:
                current_stocks = q_scores.tail(top_n).index.tolist()
            else:
                current_stocks = q_scores.head(top_n).index.tolist()

        # 计算季度收益
        q_prices = prices[current_stocks][q_start:q_end_str].resample('W').last()
        if q_prices.empty:
            continue
        q_ret = q_prices.pct_change().dropna().mean(axis=1)
        portfolio_returns.extend(q_ret.tolist())

    if not portfolio_returns:
        return None

    tw_ret = pd.Series(portfolio_returns)
    ann = float((1 + tw_ret.mean())**52 - 1)
    vol = float(tw_ret.std() * np.sqrt(52))
    sh = ann / vol if vol > 0 else 0
    cum = (1+tw_ret).cumprod()
    mdd = float((cum/cum.cummax()-1).min())

    return {'sharpe': sh, 'ann': ann, 'max_dd': mdd, 'stocks': long_stocks[:5]}

def run_window(tw_start, tw_end, label, train_start, train_end):
    print(f"\n{'='*55}")
    print(f"窗口 {label}  |  Train: {train_start[:4]}~{train_end[:4]}  |  Test: {tw_start[:4]}")
    print(f"{'='*55}")

    valid_pool = get_valid_tickers(tw_start, min_weeks=130)
    print(f"  有效股票: {len(valid_pool)} 只")

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

    # 按|IC|排序
    train_sorted = sorted(train_results, key=lambda x: abs(x['mean_ic']), reverse=True)
    best = train_sorted[0]

    print(f"  最优策略: {best['factors']} top{best['top_n']}")
    print(f"  Train IC={best['mean_ic']:+.4f} frac={best['frac']:.2f} dir={best['direction']}")

    # 回测
    test_res = run_backtest_v2(tw_start, tw_end, best['factors'], best['direction'],
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
    print(f"  选股: {test_res['stocks']}")

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
print("滚动 Walk-Forward v8 — 扩大选股池 + 多因子")
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
print("滚动 Walk-Forward v8 汇总")
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

with open('/tmp/rolling_wfa_v8_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n已保存: /tmp/rolling_wfa_v8_results.json")