#!/usr/bin/env python3
"""
Rolling Walk-Forward v11 — 精简版：候选池~150 + 熊市filter
=============================================================================
运行：python momentum-strategy/scripts/rolling_wfa_v11.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 候选池（150只，精选）
# ==============================================================================
UNIVERSE_TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','AVGO','ADBE','CRM',
    'CSCO','ACN','ORCL','IBM','INTC','AMD','QCOM','TXN','HON','AMAT','MU',
    'LRCX','ASML','KLAC','MPWR','ON','PANW','SNPS','CDNS','FTNT','NOW',
    'WDAY','PLTR','SNOW','DDOG','NET','VEEV','SQ','SHOP','PYPL','MU',
    'JPM','BAC','WFC','GS','MS','C','BLK','AXP','V','MA','COF',
    'SCHW','USB','PNC','TFC','UNH','JNJ','LLY','PFE','ABT','ABBV',
    'BMY','MRK','AMGN','GILD','BIIB','REGN','VRTX','MRNA','ISRG','DXCM',
    'TMO','DHR','BDX','SYK','MDT','EW','ZTS','HCA','CAT','DE',
    'BA','HON','UPS','RTX','LMT','GE','MMM','EMR','FDX','CSX',
    'NSC','UNP','WM','RSG','FAST','PCAR','PG','KO','PEP','WMT',
    'COST','HD','MCD','NKE','SBUX','TGT','LOW','EL','CL','GIS',
    'MDLZ','KHC','DG','ROST','BBY','EBAY','XOM','CVX','COP','SLB',
    'EOG','MPC','PSX','VLO','PXD','OXY','HAL','BKR','DVN','APA',
    'AMT','PLD','CCI','EQIX','PSA','AVB','EQR','NEE','DUK','SO',
    'D','AEP','EXC','XEL','SRE','TMUS','T','VZ','CMCSA','LIN',
    'APD','ECL','SHW','NEM','FCX','NUE','DOW','DD','PPG','ALB',
    'GM','F','RIVN','LCID','SNAP','PINS','CRWD','ZS','HUBS','MDB',
    'NET','OKTA','TEAM','WDAY','DT','OKTA','FICO','SPGI','NDAQ','ICE',
    'AXON','TDG','CHRW','JBHT','GPC','POOL','GRMN','EFX','COST','UL',
    'MORN','HUM','CNC','HCA','UHS','STE','IQV','BALL','KEYS','TER',
]
UNIVERSE_TICKERS = sorted(set(t for t in UNIVERSE_TICKERS if isinstance(t, str) and len(t) <= 5))
print(f"候选池: {len(UNIVERSE_TICKERS)} 只")

# ==============================================================================
# 2. 下载价格数据
# ==============================================================================
print("\n下载价格数据...")
all_prices = {}
BATCH = 30
for i in range(0, len(UNIVERSE_TICKERS), BATCH):
    batch = UNIVERSE_TICKERS[i:i+BATCH]
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
    print(f"  {min(i+BATCH, len(UNIVERSE_TICKERS))}/{len(UNIVERSE_TICKERS)}, {len(all_prices)} valid")

prices = pd.DataFrame(all_prices)
week = prices.resample('W').last()
monthly = prices.resample('ME').last()
print(f"价格矩阵: {prices.shape}")

# ==============================================================================
# 3. IPO日期（从价格数据推断）
# ==============================================================================
IPO_DATES = {}
for t in all_prices.keys():
    first_date = prices[t].dropna().index[0]
    IPO_DATES[t] = first_date.to_pydatetime()

# ==============================================================================
# 4. 预计算月度因子值
# ==============================================================================
print("\n预计算月度因子值...")
factor_matrices = {}
for fname, wk in [('roc20',20),('roc60',60),('roc120',120),('vol20',20),('vol60',60)]:
    rows = {}
    for dt in monthly.index:
        dt_str = str(dt)[:10]
        sub = week[:dt_str]
        if len(sub) < wk+5:
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

# ==============================================================================
# 5. 候选策略（精简）
# ==============================================================================
STRAT_CONFIGS = []
for fn in ['roc20','roc60','roc120','vol20','vol60']:
    for n in [10, 20]:
        STRAT_CONFIGS.append(([fn], n, False))
for f1 in ['roc20','roc60','roc120']:
    for f2 in ['vol20','vol60']:
        for n in [10, 20]:
            STRAT_CONFIGS.append(([f1, f2], n, False))
# 反向策略
for fn in ['roc20','roc60','roc120']:
    for n in [10, 20]:
        STRAT_CONFIGS.append(([fn], n, True))
print(f"候选策略: {len(STRAT_CONFIGS)} 个")

# ==============================================================================
# 6. 滚动窗口（2015起，有足够历史）
# ==============================================================================
ROLL_WINDOWS = [
    ('2010-01-01','2015-01-01','2015-01-01','2016-01-01','2015'),
    ('2011-01-01','2016-01-01','2016-01-01','2017-01-01','2016'),
    ('2012-01-01','2017-01-01','2017-01-01','2018-01-01','2017'),
    ('2013-01-01','2018-01-01','2018-01-01','2019-01-01','2018'),
    ('2014-01-01','2019-01-01','2019-01-01','2020-01-01','2019'),
    ('2015-01-01','2020-01-01','2020-01-01','2021-01-01','2020'),
    ('2016-01-01','2021-01-01','2021-01-01','2022-01-01','2021'),
    ('2017-01-01','2022-01-01','2022-01-01','2023-01-01','2022'),
    ('2018-01-01','2023-01-01','2023-01-01','2024-01-01','2023'),
    ('2019-01-01','2024-01-01','2024-01-01','2025-01-01','2024'),
]

# ==============================================================================
# 7. 核心函数
# ==============================================================================
def get_valid_tickers(tw_start, min_weeks=100):
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

def eval_train(train_start, train_end, factors, valid_pool, is_reverse):
    train_months = monthly[train_start:train_end].index.tolist()
    monthly_ics = []
    for i in range(len(train_months)-1):
        dt_str = str(train_months[i])[:10]
        next_str = str(train_months[i+1])[:10]
        if next_str not in monthly.index:
            continue
        mret = monthly.loc[next_str]
        fvs = []
        ok = True
        for fn in factors:
            if fn not in factor_matrices or dt_str not in factor_matrices[fn].index:
                ok = False
                break
            fvs.append(factor_matrices[fn].loc[dt_str][valid_pool])
        if not ok:
            continue
        score = sum(fv.rank(pct=True) for fv in fvs)
        ic = calc_ic(mret[valid_pool], score)
        if is_reverse:
            ic = -ic
        monthly_ics.append(ic)
    if len(monthly_ics) < 24:
        return None
    arr = np.array(monthly_ics)
    return {'mean_ic': np.nanmean(arr), 'frac': np.nanmean(arr>0), 'n': len(arr),
             'direction': +1 if np.nanmean(arr) > 0 else -1}

def get_spy_regime(tw_start, tw_end):
    """SPY是否在200日均线上方"""
    spy = yf.download('SPY','2007-01-01','2026-03-01',auto_adjust=False,progress=False)['Close'].squeeze()
    if spy.index.tz: spy = spy.tz_localize(None)
    spy_w = spy.resample('W').last()
    spy_sma200 = spy.rolling(200).mean()
    tw_spy = spy_w[tw_start:tw_end]
    tw_sma = spy_sma200.reindex(tw_spy.index)
    regimes = tw_spy > tw_sma
    return regimes

def run_backtest_regime(tw_start, tw_end, factors, direction, top_n, valid_pool, is_reverse):
    hist = week[week.index < tw_start]
    if len(hist) < 120:
        return None
    pool = [t for t in valid_pool if t in hist.columns]
    p0 = prices.loc[tw_start:tw_end].iloc[0]
    pool = [t for t in pool if 3 < p0.get(t, 0) < 1000]
    if len(pool) < top_n*2:
        return None

    scores = pd.Series(0.0, index=pool)
    for fn in factors:
        wk = 120 if fn=='roc120' else (60 if fn=='roc60' else 20)
        if fn.startswith('roc'):
            fv = (hist.iloc[-1]/hist.iloc[-wk]-1)[pool]
        else:
            ret = hist[pool].pct_change().dropna()
            vol_wk = 20 if fn=='vol20' else 60
            fv = ret.rolling(vol_wk).std().iloc[-1]
        scores += fv.rank(pct=True)

    if is_reverse or direction == -1:
        selected = scores.nsmallest(top_n).index.tolist()
    else:
        selected = scores.nlargest(top_n).index.tolist()

    if len(selected) < 5:
        return None

    spy_regimes = get_spy_regime(tw_start, tw_end)
    tp = prices[selected][tw_start:tw_end].resample('W').last()
    ret = tp.pct_change().dropna().mean(axis=1)

    ret_filtered = []
    for dt in ret.index:
        in_bull = True
        if dt in spy_regimes.index:
            in_bull = bool(spy_regimes.loc[dt])
        weekly_ret = ret.loc[dt]
        if not in_bull:
            ret_filtered.append(weekly_ret * 0.5)
        else:
            ret_filtered.append(weekly_ret)

    ret_f = pd.Series(ret_filtered, index=ret.index)
    ann = float((1+ret_f.mean())**52-1)
    vol = float(ret_f.std()*np.sqrt(52))
    sh = ann/vol if vol > 0 else 0
    cum = (1+ret_f).cumprod()
    mdd = float((cum/cum.cummax()-1).min())

    return {'sharpe': sh, 'ann': ann, 'max_dd': mdd, 'stocks': selected[:5]}

def run_window(tw_start, tw_end, label, train_start, train_end):
    print(f"\n{'='*55}")
    print(f"窗口 {label}  |  Train: {train_start[:4]}~{train_end[:4]}  |  Test: {tw_start[:4]}")
    print(f"{'='*55}")

    valid_pool = get_valid_tickers(tw_start, min_weeks=100)
    print(f"  有效股票: {len(valid_pool)} 只")

    train_results = []
    for factors, top_n, is_rev in STRAT_CONFIGS:
        res = eval_train(train_start, train_end, factors, valid_pool, is_rev)
        if res is None:
            continue
        train_results.append({'factors': factors, 'top_n': top_n, 'is_reverse': is_rev, **res})

    if not train_results:
        print("  ⚠️ 无可用策略")
        return None

    train_sorted = sorted(train_results, key=lambda x: abs(x['mean_ic']), reverse=True)
    top3 = train_sorted[:3]

    print(f"  Top-3策略:")
    for t in top3:
        rev_str = " [rev]" if t['is_reverse'] else ""
        print(f"    {t['factors']} top{t['top_n']}{rev_str} IC={t['mean_ic']:+.4f} frac={t['frac']:.2f}")

    spy = yf.download('SPY','2007-01-01','2026-03-01',auto_adjust=False,progress=False)['Close'].squeeze()
    if spy.index.tz: spy = spy.tz_localize(None)
    spy_w = spy.resample('W').last()
    spy_tw = spy_w[tw_start:tw_end].pct_change().dropna()
    spy_ann = float((1+spy_tw.mean())**52-1)
    spy_vol = float(spy_tw.std()*np.sqrt(52))
    spy_sh = spy_ann/spy_vol if spy_vol > 0 else 0

    regimes = get_spy_regime(tw_start, tw_end)
    bull_pct = float(regimes.mean())
    print(f"  牛市占比: {bull_pct:.0%}")

    all_rets = []
    for strat in top3:
        tr = run_backtest_regime(tw_start, tw_end, strat['factors'], strat['direction'],
                                 strat['top_n'], valid_pool, strat['is_reverse'])
        if tr:
            all_rets.append(tr)

    if not all_rets:
        print("  ⚠️ 回测失败")
        return None

    avg_sh = np.mean([r['sharpe'] for r in all_rets])
    avg_ann = np.mean([r['ann'] for r in all_rets])
    avg_mdd = np.mean([r['max_dd'] for r in all_rets])
    outp = avg_sh - spy_sh
    marker = "★" if outp > 0.3 else ("☆" if outp > 0 else "")

    print(f"  Top-3平均+Regime: Sharpe={avg_sh:.2f}, Ann={avg_ann*100:.1f}%, MaxDD={avg_mdd*100:.1f}%")
    print(f"  SPY={spy_sh:.2f}  超额={outp:+.2f}  {marker}")

    return {
        'window': label,
        'train_ic_avg': round(np.mean([s['mean_ic'] for s in top3]), 4),
        'sharpe': round(avg_sh, 2),
        'ann': round(avg_ann*100, 1),
        'max_dd': round(avg_mdd*100, 1),
        'spy_sharpe': round(spy_sh, 2),
        'outperformance': round(outp, 2),
        'bull_pct': round(bull_pct, 3),
        'valid_pool_size': len(valid_pool),
        'top3': [{'factors': s['factors'], 'top_n': s['top_n'],
                  'is_reverse': s['is_reverse'], 'train_ic': round(s['mean_ic'], 4)} for s in top3]
    }

# ==============================================================================
# 运行
# ==============================================================================
print("\n" + "="*55)
print("滚动 Walk-Forward v11 — 精简版 150股票+熊市filter")
print("="*55)

all_results = []
for (train_start, train_end, tw_start, tw_end, label) in ROLL_WINDOWS:
    result = run_window(tw_start, tw_end, label, train_start, train_end)
    if result:
        all_results.append(result)

# ==============================================================================
# 汇总
# ==============================================================================
print("\n\n" + "="*55)
print("滚动 Walk-Forward v11 汇总")
print("="*55)
print(f"{'窗口':6s} {'TrainIC':>8} {'Sharpe':>8} {'Ann%':>8} {'MaxDD%':>8} {'SPY':>8} {'超额':>8} {'牛市%':>6s}")
print('-'*85)
for r in all_results:
    outp = r['outperformance']
    marker = "★" if outp > 0.3 else ("☆" if outp > 0 else "")
    print(f"{r['window']:6s} {r['train_ic_avg']:>+8.4f} {r['sharpe']:8.2f} {r['ann']:7.1f}% {r['max_dd']:7.1f}% {r['spy_sharpe']:8.2f} {outp:>+7.2f}  {marker} {r['bull_pct']:.0%}")
print('-'*85)

sharpes = [r['sharpe'] for r in all_results]
spy_sharpes = [r['spy_sharpe'] for r in all_results]
outs = [r['outperformance'] for r in all_results]
print(f"{'平均':6s} {np.mean([r['train_ic_avg'] for r in all_results]):>+8.4f} {np.mean(sharpes):8.2f}              {np.mean(spy_sharpes):8.2f} {np.mean(outs):>+7.2f}")

win_rate = np.mean([o > 0 for o in outs])
print(f"胜率: {win_rate:.0%} ({sum([o>0 for o in outs])}/{len(outs)})")
print(f"平均有效股票池: {np.mean([r['valid_pool_size'] for r in all_results]):.0f} 只")

with open('/tmp/rolling_wfa_v11_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n已保存: /tmp/rolling_wfa_v11_results.json")