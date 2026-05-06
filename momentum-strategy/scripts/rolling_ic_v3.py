"""
Price Factor 回测 v3 — 修复所有 bug

三段式：Train(2011-2016) / Val(2016-2021) / Test(2021-2026)
- 月度调仓，等权组合
- 真实 MaxDD（追踪组合净值）
- 预计算所有因子时间序列
- 滚动 IC 用 month-by-month 计算
"""

import sys
sys.path.insert(0, '/home/ubuntu/.hermes/skills/quant-trading-momentum/scripts')

import json, math, time, warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

UNIVERSE = [
    'AAPL','MSFT','AMZN','GOOGL','GOOG','META','NVDA','AVGO','TSLA','CSCO',
    'ADBE','NFLX','ORCL','CRM','AMD','INTC','QCOM','TXN','MU','AMAT','IBM',
    'JPM','BAC','WFC','GS','MS','C','BLK','AXP','V','MA','SCHW','USB','PNC',
    'LMT','BA','CAT','GE','UPS','HON','RTX','DE','MMM','ITW','ETN','PH',
    'LLY','UNH','JNJ','PFE','ABBV','MRK','TMO','DHR','AMGN','ISRG','MDT',
    'ABT','BMY','GILD',
    'WMT','HD','COST','PG','KO','PEP','MCD','SBUX','NKE','TGT','LOW','DG',
    'XOM','CVX','COP','SLB','EOG','PSX','VLO','OXY',
    'LIN','APD','SHW','FCX','NEM','DHI','LEN','VMC','MLM',
    'PLD','AMT','EQIX','CCI','PSA','SPG','O',
    'SPY','QQQ','GLD','TLT','EFA','EEM','IWM','VTI','VEA','VWO','BND',
    'BKNG','MAR','ABNB','NOW','SNOW','CRWD','PANW','ZS','TEAM','F',
]
UNIVERSE = sorted(set([t for t in UNIVERSE if isinstance(t, str) and t.isupper()]))

FACTOR_CONFIG = {
    'roc20':  {'type': 'roc',  'window': 20},
    'roc60':  {'type': 'roc',  'window': 60},
    'roc120': {'type': 'roc',  'window': 120},
    'vol20':  {'type': 'vol',  'window': 20},
    'vol60':  {'type': 'vol',  'window': 60},
}

# ============================================================
# 下载价格数据
# ============================================================
print("Downloading price data...")
DATA_START = '2010-01-01'
DATA_END   = '2026-02-01'

batch_size = 50
all_closes = []
for i in range(0, len(UNIVERSE), batch_size):
    batch = UNIVERSE[i:i+batch_size]
    n_batch = (len(UNIVERSE)-1)//batch_size + 1
    print(f"  Batch {i//batch_size + 1}/{n_batch}", end='', flush=True)
    try:
        d = yf.download(batch, start=DATA_START, end=DATA_END,
                        progress=False, auto_adjust=True)
        close = d.xs('Close', axis=1, level=0)
        all_closes.append(close)
        print(f" OK({close.shape[1]})")
    except Exception as e:
        print(f" FAIL: {e}")
    time.sleep(0.2)

data_all = pd.concat(all_closes, axis=1, sort=True)
if data_all.columns.duplicated().any():
    data_all = data_all.loc[:, ~data_all.columns.duplicated()]

available = []
for t in UNIVERSE:
    if t not in data_all.columns:
        continue
    col = data_all[t].dropna()
    if len(col) < 252:
        continue
    if col.index[0] <= pd.Timestamp('2010-12-31'):
        available.append(t)

UNIVERSE = sorted(set(available))
print(f"\nAvailable: {len(UNIVERSE)} tickers")
ticker_idx = {t: i for i, t in enumerate(UNIVERSE)}
n_tickers  = len(UNIVERSE)

# ============================================================
# 构建月频价格矩阵 (n_tickers x n_months)
# ============================================================
print("\nBuilding monthly price matrix...")
monthly_ends = pd.date_range('2010-03-01', '2026-01-01', freq='ME')
n_months = len(monthly_ends)

px_matrix = np.full((n_tickers, n_months), np.nan)
for ti, t in enumerate(UNIVERSE):
    px = data_all[t].dropna()
    for mi, m_end in enumerate(monthly_ends):
        valid = px.index[px.index <= m_end]
        if len(valid) > 0:
            px_matrix[ti, mi] = float(px.loc[valid[-1]])

# 月收益率矩阵
ret_matrix = np.full((n_tickers, n_months), np.nan)
for ti in range(n_tickers):
    for mi in range(1, n_months):
        p_prev = px_matrix[ti, mi-1]
        p_cur  = px_matrix[ti, mi]
        if not (np.isnan(p_prev) or np.isnan(p_cur) or p_prev <= 0 or p_cur <= 0):
            ret_matrix[ti, mi] = p_cur / p_prev - 1

# 期间索引
def mi_for(date_str):
    ts = pd.Timestamp(date_str)
    idx = monthly_ends.get_indexer([ts], method='ffill')[0]
    return max(0, idx)

TRAIN_MI = mi_for('2011-01-01')
VAL_MI   = mi_for('2016-01-01')
TEST_MI  = mi_for('2021-01-01')
END_MI   = n_months - 1

# ============================================================
# 预计算因子时间序列
# ============================================================
print("Pre-computing factor time series...")

def compute_factor_matrix(factor_key, window):
    """返回 (n_tickers, n_months) 因子矩阵"""
    ftype = FACTOR_CONFIG[factor_key]['type']
    result = np.full((n_tickers, n_months), np.nan)

    for ti in range(n_tickers):
        for mi in range(window, n_months):
            start_mi = mi - window + 1
            px_win = px_matrix[ti, start_mi:mi+1]
            if np.any(np.isnan(px_win)):
                continue
            if ftype == 'roc':
                result[ti, mi] = px_win[-1] / px_win[0] - 1
            else:  # vol: 年化月收益标准差
                rets = np.diff(px_win) / px_win[:-1]
                result[ti, mi] = float(np.std(rets) * math.sqrt(12))
    return result

factor_ts = {}
for fname in FACTOR_CONFIG:
    print(f"  {fname}...", end='', flush=True)
    factor_ts[fname] = compute_factor_matrix(fname, FACTOR_CONFIG[fname]['window'])
    print(" done")

# ============================================================
# 滚动 IC 计算（修复版）
# ============================================================

def compute_rolling_ic(factor_key, period_start_mi, period_end_mi):
    """
    计算 period 内每个月 IC_t：
    因子值：month mi（用 window 期历史数据）
    收益：month mi+1（次月收益）
    返回 IC 列表
    """
    ic_list = []
    lookback = FACTOR_CONFIG[factor_key]['window']

    for mi in range(period_start_mi + lookback, period_end_mi):
        fvals = factor_ts[factor_key][:, mi]        # 当月因子
        rets  = ret_matrix[:, mi + 1]               # 次月收益

        mask = ~(np.isnan(fvals) | np.isnan(rets))
        if mask.sum() < 15:
            continue

        xs = fvals[mask]
        ys = rets[mask]
        if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
            continue

        ic = float(np.corrcoef(xs, ys)[0, 1])
        ic_list.append(ic)

    return ic_list

def ic_summary(ic_list):
    if len(ic_list) < 3:
        return dict(mean=0, std=0, frac_pos=0, stability=0, n=len(ic_list))
    arr = np.array(ic_list)
    mean_ic = float(np.mean(arr))
    std_ic  = float(np.std(arr))
    frac    = float(np.mean((arr > 0).astype(float)))
    stab    = mean_ic / std_ic if std_ic > 1e-9 else 0.0
    return dict(mean=mean_ic, std=std_ic, frac_pos=frac, stability=stab, n=len(ic_list))

# ============================================================
# STEP 1: Train IC
# ============================================================
print("\n" + "="*70)
print("STEP 1: Train 期滚动月度 IC")
print("="*70)

train_ics = {}
for fname in FACTOR_CONFIG:
    train_ics[fname] = compute_rolling_ic(fname, TRAIN_MI, VAL_MI)

print(f"\n{'Factor':<10} {'Mean IC':>9} {'Std IC':>8} {'IC>0%':>8} {'Stability':>10} {'n':>5}")
print("-"*55)
for fname in FACTOR_CONFIG:
    s = ic_summary(train_ics[fname])
    sig = "✓" if s['frac_pos'] > 0.55 and s['stability'] > 0.3 else " "
    print(f"  {fname:<8} {s['mean']:>+9.3f} {s['std']:>8.3f} {s['frac_pos']:>8.1%} {s['stability']:>+10.3f} {s['n']:>5} {sig}")

# ============================================================
# STEP 2: Val IC
# ============================================================
print("\n" + "="*70)
print("STEP 2: Val 期滚动月度 IC")
print("="*70)

val_ics = {}
for fname in FACTOR_CONFIG:
    val_ics[fname] = compute_rolling_ic(fname, VAL_MI, TEST_MI)

print(f"\n{'Factor':<10} {'Val Mean IC':>12} {'Val Std IC':>10} {'IC>0%':>8} {'Stability':>10} {'n':>5}")
print("-"*60)
for fname in FACTOR_CONFIG:
    s = ic_summary(val_ics[fname])
    sig = "✓" if s['frac_pos'] > 0.55 and s['stability'] > 0.3 else " "
    print(f"  {fname:<8} {s['mean']:>+12.3f} {s['std']:>10.3f} {s['frac_pos']:>8.1%} {s['stability']:>+10.3f} {s['n']:>5} {sig}")

# ============================================================
# STEP 3: 因子方向 + 权重
# ============================================================

factor_direction = {}
factor_weight    = {}

for fname in FACTOR_CONFIG:
    vs = ic_summary(val_ics[fname])
    val_frac = vs['frac_pos']
    if val_frac >= 0.55:
        factor_direction[fname] = +1
    elif val_frac <= 0.45:
        factor_direction[fname] = -1
    else:
        factor_direction[fname] = 0
    factor_weight[fname] = abs(vs['stability'])

active_factors = [f for f in FACTOR_CONFIG
                  if factor_direction[f] != 0 and factor_weight[f] > 0]

print(f"\nActive factors:")
for fname in active_factors:
    vs = ic_summary(val_ics[fname])
    print(f"  {fname:<8} dir={factor_direction[fname]:>+2d}  weight={factor_weight[fname]:>6.3f}  Val_IC={vs['mean']:>+6.3f}  IC>0%={vs['frac_pos']:.0%}")

# ============================================================
# STEP 4: 回测（正确 MaxDD）
# ============================================================

def backtest(top_n, factor_keys, start_mi, end_mi):
    """
    月度调仓回测
    - 用 start_mi 当月初的因子值选股
    - 用 start_mi 到 start_mi+1 的月收益率计算持仓收益
    - 追踪组合净值，计算真实 MaxDD
    """
    if start_mi >= end_mi - 1:
        return None

    portfolio_values = [1.0]
    prev_val = 1.0

    for mi in range(start_mi, end_mi):
        # 选 top_n
        if factor_keys:
            scores = np.zeros(n_tickers)
            for fname in factor_keys:
                ft = factor_ts[fname][:, mi]
                mask = ~np.isnan(ft)
                ft_norm = np.zeros(n_tickers)
                ft_norm[mask] = (ft[mask] - np.mean(ft[mask])) / (np.std(ft[mask]) + 1e-9)
                scores += factor_direction[fname] * factor_weight[fname] * ft_norm

            top_indices = np.argsort(scores)[-top_n:]
            # 去掉当月无收益的
            valid = ~np.isnan(ret_matrix[top_indices, mi + 1])
            top_indices = top_indices[valid]
        else:
            top_indices = np.arange(n_tickers)

        if len(top_indices) == 0:
            portfolio_values.append(prev_val)
            continue

        port_ret = float(np.nanmean(ret_matrix[top_indices, mi + 1]))
        prev_val = prev_val * (1 + port_ret)
        portfolio_values.append(prev_val)

    pv = np.array(portfolio_values)

    # 月度收益率序列
    monthly_rets = np.diff(pv) / pv[:-1]

    ann  = float(np.mean(monthly_rets) * 12)
    std  = float(np.std(monthly_rets) * math.sqrt(12))
    shrp = ann / std if std > 1e-9 else 0.0

    # 真实 MaxDD
    running_max = np.maximum.accumulate(pv)
    drawdowns   = (pv - running_max) / running_max
    max_dd      = float(np.min(drawdowns))

    return dict(ann=ann, std=std, sharpe=shrp, max_dd=max_dd, n_months=len(monthly_rets))

# SPY B&H
spy_idx = ticker_idx.get('SPY')
spy_vals = [1.0]
prev = 1.0
for mi in range(TEST_MI, END_MI):
    r = ret_matrix[spy_idx, mi + 1] if not np.isnan(ret_matrix[spy_idx, mi + 1]) else 0.0
    prev = prev * (1 + r)
    spy_vals.append(prev)
spy_pv = np.array(spy_vals)
spy_rets = np.diff(spy_pv) / spy_pv[:-1]
spy_ann  = float(np.mean(spy_rets) * 12)
spy_std  = float(np.std(spy_rets) * math.sqrt(12))
spy_shrp = spy_ann / spy_std if spy_std > 1e-9 else 0.0
spy_mdd  = float(np.min((spy_pv - np.maximum.accumulate(spy_pv)) / np.maximum.accumulate(spy_pv)))

# ============================================================
# 运行回测
# ============================================================
print(f"\n" + "="*70)
print(f"STEP 4: Test 期回测")
print("="*70)

print(f"\n{'Strategy':<35} {'n':>5} {'Sharpe':>8} {'Ann%':>10} {'MaxDD%':>10}")
print("-"*72)

all_results = []

for fname in active_factors:
    for top_n in [5, 10, 20, 30]:
        r = backtest(top_n, [fname], TEST_MI, END_MI)
        if r:
            name = f"Price-{fname} top{top_n}"
            print(f"  {name:<33} {r['n_months']:>5} {r['sharpe']:>8.2f} {r['ann']*100:>+10.1f}% {r['max_dd']*100:>+10.1f}%")
            all_results.append({**r, 'strategy': name})

for top_n in [5, 10, 20, 30]:
    r = backtest(top_n, active_factors, TEST_MI, END_MI)
    if r:
        name = f"Multi-Factor top{top_n}"
        print(f"  {name:<33} {r['n_months']:>5} {r['sharpe']:>8.2f} {r['ann']*100:>+10.1f}% {r['max_dd']*100:>+10.1f}%")
        all_results.append({**r, 'strategy': name})

print(f"  {'SPY B&H':<33} {'~60':>5} {spy_shrp:>8.2f} {spy_ann*100:>+10.1f}% {spy_mdd*100:>+10.1f}%")

# ============================================================
# 保存
# ============================================================
output = {
    'train_ic': {fname: ic_summary(ics) for fname, ics in train_ics.items()},
    'val_ic':   {fname: ic_summary(ics) for fname, ics in val_ics.items()},
    'factor_direction': {k: int(v) for k, v in factor_direction.items()},
    'factor_weight':    {k: float(v) for k, v in factor_weight.items()},
    'test_results':     all_results,
}

with open('/tmp/rolling_ic_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to /tmp/rolling_ic_results.json")
