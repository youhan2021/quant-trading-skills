#!/usr/bin/env python3
"""
Rolling Walk-Forward v7 — Long-Short + 基本面过滤
=========================================================
改进：
  - Long-Short 10只多头 + 10只空头，对冲市场beta
  - 基本面过滤：PE<30, PE>0, 剔除金融和公用事业
  - IPO严格过滤：Test期期初必须有至少100周历史

运行：python momentum-strategy/scripts/rolling_wfa_v7.py
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
# 3. IPO + 基本面过滤
# ------------------------------------------------------------------
print("\n获取股票元数据...")
METADATA = {}
EXCLUDE_SECTORS = {'Financial', 'Utilities', 'Real Estate'}

for t in UNIVERSE:
    try:
        info = yf.Ticker(t).info
        ipo_ts = info.get('firstTradeDateEpochGregorian')
        if ipo_ts:
            from datetime import datetime
            ipo_date = datetime.utcfromtimestamp(ipo_ts)
        else:
            first_price = prices[t].dropna()
            ipo_date = first_price.index[0].to_pydatetime()
        
        pe = info.get('trailingPE') or info.get('forwardPE') or 0
        sector = info.get('sector') or ''
        industry = info.get('industry') or ''
        
        METADATA[t] = {
            'ipo': ipo_date,
            'pe': pe,
            'sector': sector,
            'industry': industry,
        }
    except:
        METADATA[t] = {'ipo': None, 'pe': 0, 'sector': '', 'industry': ''}

# ------------------------------------------------------------------
# 4. 预计算月度因子值
# ------------------------------------------------------------------
print("\n预计算月度因子值...")
FACTOR_CONFIGS = [
    ('roc20', 20),
    ('roc60', 60),
    ('roc120', 120),
    ('vol20', 20),
    ('vol60', 60),
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
# 5. 候选策略（简化为最重要的）
# ------------------------------------------------------------------
# 单因子策略
STRAT_CONFIGS = []
for fn in ['roc20', 'roc60', 'roc120', 'vol20', 'vol60']:
    for n in [5, 10]:
        STRAT_CONFIGS.append(([fn], n))

# 双因子组合
for f1 in ['roc20', 'roc60', 'roc120']:
    for f2 in ['vol20', 'vol60']:
        for n in [5, 10]:
            STRAT_CONFIGS.append(([f1, f2], n))

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
def get_valid_tickers(tw_start, min_weeks=100):
    """获取在Test期起点时有足够历史的股票"""
    from datetime import datetime
    tw_ts = pd.Timestamp(tw_start)
    valid = []
    for t in prices.columns:
        if t not in METADATA:
            continue
        ipo = METADATA[t].get('ipo')
        if ipo is None:
            continue
        # 检查：IPO日期距离tw_start至少min_weeks
        ipo_ts = pd.Timestamp(ipo)
        weeks_since_ipo = (tw_ts - ipo_ts).days / 7
        if weeks_since_ipo < min_weeks:
            continue
        # 基本面过滤：剔除金融、公用事业、RE
        sector = METADATA[t].get('sector', '')
        if sector in EXCLUDE_SECTORS:
            continue
        # PE过滤：0 < PE < 50
        pe = METADATA[t].get('pe', 0)
        if pe <= 0 or pe > 50:
            continue
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
            fv = fv[valid_pool]
            fvals[fn] = fv

        if not valid:
            continue

        # rank综合得分
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

def run_backtest_ls(tw_start, tw_end, strat_factors, direction, top_n, valid_pool, lookback_weeks=120):
    """Long-Short回测：多头top_n + 空头bottom_n"""
    hist = week[week.index < tw_start]
    if len(hist) < lookback_weeks:
        return None

    pool = [t for t in valid_pool if t in hist.columns]
    price_at_start = prices.loc[tw_start:tw_end].iloc[0]
    pool = [t for t in pool if price_at_start.get(t, 0) > 3 and price_at_start.get(t, 0) < 2000]

    if len(pool) < top_n * 3:
        return None

    # 计算截面因子得分
    scores = pd.Series(index=pool, dtype=float)
    for fn in strat_factors:
        # 正确的lookback
        LB = 120 if fn == 'roc120' else (60 if fn == 'roc60' else 20)
        if fn.startswith('roc'):
            fv = (hist.iloc[-1] / hist.iloc[-LB] - 1)[pool]
        else:
            ret = hist[pool].pct_change().dropna()
            vol_wk = 20 if fn == 'vol20' else 60
            fv = ret.rolling(vol_wk).std().iloc[-1]
        scores += fv.rank(pct=True)

    # Long-Short：用不同的阈值保证不重叠
    # 多头：得分最高的top_n只；空头：得分最低的top_n只（排除已选的多头）
    sorted_scores = scores.sort_values(ascending=False)
    if direction == -1:
        long_stocks = sorted_scores.tail(top_n).index.tolist()  # 最低的
        short_stocks = sorted_scores.head(top_n).index.tolist()  # 最高的
    else:
        long_stocks = sorted_scores.head(top_n).index.tolist()  # 最高的
        short_stocks = sorted_scores.tail(top_n).index.tolist()  # 最低的

    if len(long_stocks) < 3 or len(short_stocks) < 3:
        return None

    # 回测
    test_prices = prices[long_stocks + short_stocks][tw_start:tw_end].resample('W').last()
    
    long_ret = test_prices[long_stocks].pct_change().dropna().mean(axis=1)
    short_ret = test_prices[short_stocks].pct_change().dropna().mean(axis=1)
    ls_ret = long_ret - short_ret  # 市场中性
    
    ls_ann = float((1 + ls_ret.mean())**52 - 1)
    ls_vol = float(ls_ret.std() * np.sqrt(52))
    ls_sh = ls_ann / ls_vol if ls_vol > 0 else 0
    
    # 也计算纯多头作为对比
    long_ann = float((1 + long_ret.mean())**52 - 1)
    long_vol = float(long_ret.std() * np.sqrt(52))
    long_sh = long_ann / long_vol if long_vol > 0 else 0
    
    cum = (1+ls_ret).cumprod()
    mdd = float((cum/cum.cummax()-1).min())

    return {
        'ls_sharpe': ls_sh, 'ls_ann': ls_ann, 'ls_max_dd': mdd,
        'long_sharpe': long_sh, 'long_ann': long_ann,
        'long_stocks': long_stocks, 'short_stocks': short_stocks,
    }

def run_window(tw_start, tw_end, label, train_start, train_end):
    """运行单个窗口"""
    print(f"\n{'='*55}")
    print(f"窗口 {label}  |  Train: {train_start[:4]}~{train_end[:4]}  |  Test: {tw_start[:4]}")
    print(f"{'='*55}")

    valid_pool = get_valid_tickers(tw_start, min_weeks=100)
    print(f"  有效股票(IPO>=100周 + PE过滤): {len(valid_pool)} 只")

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
    test_res = run_backtest_ls(tw_start, tw_end, best['factors'], best['direction'],
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

    ls_sh = test_res['ls_sharpe']
    ls_out = ls_sh - spy_sh
    long_sh = test_res['long_sharpe']
    long_out = long_sh - spy_sh

    print(f"  Long-Short: Sharpe={ls_sh:.2f}, Ann={test_res['ls_ann']*100:.1f}%, MaxDD={test_res['ls_max_dd']*100:.1f}%")
    print(f"    SPY={spy_sh:.2f}  LS超额={ls_out:+.2f}")
    print(f"  Long Only: Sharpe={long_sh:.2f}, Ann={test_res['long_ann']*100:.1f}%  超额={long_out:+.2f}")
    print(f"  多头: {test_res['long_stocks'][:3]}...  空头: {test_res['short_stocks'][:3]}...")

    return {
        'window': label,
        'factors': best['factors'],
        'top_n': best['top_n'],
        'direction': best['direction'],
        'train_ic': round(best['mean_ic'], 4),
        'train_frac': round(best['frac'], 3),
        'valid_pool_size': len(valid_pool),
        'ls_sharpe': round(ls_sh, 2),
        'ls_ann': round(test_res['ls_ann']*100, 1),
        'ls_max_dd': round(test_res['ls_max_dd']*100, 1),
        'long_sharpe': round(long_sh, 2),
        'long_ann': round(test_res['long_ann']*100, 1),
        'spy_sharpe': round(spy_sh, 2),
        'ls_outperformance': round(ls_out, 2),
        'long_outperformance': round(long_out, 2),
        'long_stocks': test_res['long_stocks'],
        'short_stocks': test_res['short_stocks'],
    }

# ------------------------------------------------------------------
# 8. 运行
# ------------------------------------------------------------------
print("\n" + "="*55)
print("滚动 Walk-Forward v7 — Long-Short + 基本面过滤")
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
print("滚动 Walk-Forward v7 汇总")
print("="*55)
print(f"{'窗口':6s} {'TrainIC':>8} {'LS-Sharp':>8} {'Long-Sh':>8} {'SPY':>8} {'LS超额':>8} {'Long超额':>8}")
print('-'*75)
for r in all_results:
    ls_out = r['ls_outperformance']
    long_out = r['long_outperformance']
    ls_mark = "★" if ls_out > 0.3 else ("☆" if ls_out > 0 else "")
    long_mark = "★" if long_out > 0.3 else ("☆" if long_out > 0 else "")
    print(f"{r['window']:6s} {r['train_ic']:>+8.4f} {r['ls_sharpe']:8.2f} {r['long_sharpe']:8.2f} {r['spy_sharpe']:8.2f} {ls_out:>+7.2f}{ls_mark} {long_out:>+7.2f}{long_mark}")
print('-'*75)

ls_sharpes = [r['ls_sharpe'] for r in all_results]
long_sharpes = [r['long_sharpe'] for r in all_results]
spy_sharpes = [r['spy_sharpe'] for r in all_results]
ls_outs = [r['ls_outperformance'] for r in all_results]
long_outs = [r['long_outperformance'] for r in all_results]

print(f"{'平均':6s} {np.mean([r['train_ic'] for r in all_results]):>+8.4f} {np.mean(ls_sharpes):8.2f} {np.mean(long_sharpes):8.2f} {np.mean(spy_sharpes):8.2f} {np.mean(ls_outs):>+7.2f} {np.mean(long_outs):>+7.2f}")

ls_wr = np.mean([o > 0 for o in ls_outs])
long_wr = np.mean([o > 0 for o in long_outs])
print(f"胜率: LS={ls_wr:.0%}({sum([o>0 for o in ls_outs])}/{len(ls_outs)}) Long={long_wr:.0%}({sum([o>0 for o in long_outs])}/{len(long_outs)})")

with open('/tmp/rolling_wfa_v7_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n已保存: /tmp/rolling_wfa_v7_results.json")