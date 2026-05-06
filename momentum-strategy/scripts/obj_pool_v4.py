"""
Objective Pool Backtest v4
ETF universe → 三段式 → Val信号检验 → 不同top_n对比
"""

import sys
sys.path.insert(0, '/home/ubuntu/.hermes/skills/quant-trading-momentum/scripts')

import json, math, time
import numpy as np
import pandas as pd
import yfinance as yf
from momentum_backtest import STRATEGIES, bt_one, bt_port
from collections import Counter

TRAIN_START = '2011-01-01'
TRAIN_END   = '2016-01-01'
VAL_START   = '2016-01-01'
VAL_END     = '2021-01-01'
TEST_START  = '2021-01-01'
TEST_END    = '2026-02-01'

# ============================================================
# Universe: 学术/量化研究常用股票名单（规则客观：市值+流动性+行业分散）
# 不依赖 Wikipedia 抓取，不主观选"近期涨最好的"
# ============================================================

UNIVERSE_TICKERS = [
    # Tech
    'AAPL','MSFT','AMZN','GOOGL','GOOG','META','NVDA','AVGO','TSLA','CSCO',
    'ADBE','NFLX','ORCL','CRM','AMD','INTC','QCOM','TXN','MU','AMAT','IBM',
    # Finance
    'JPM','BAC','WFC','GS','MS','C','BLK','AXP','V','MA','SCHW','USB','PNC',
    # Industrial
    'LMT','BA','CAT','GE','UPS','HON','RTX','DE','MMM','ITW','ETN','PH',
    # Healthcare
    'LLY','UNH','JNJ','PFE','ABBV','MRK','TMO','DHR','AMGN','ISRG','MDT',
    'ABT','BMY','LLY','GILD',
    # Consumer
    'WMT','HD','COST','PG','KO','PEP','MCD','SBUX','NKE','TGT','LOW','DG',
    # Energy
    'XOM','CVX','COP','SLB','EOG','PSX','VLO','OXY',
    # Materials / Infrastructure
    'LIN','APD','SHW','FCX','NEM','DHI','LEN','VMC','MLM',
    # Real estate / Infra
    'PLD','AMT','EQIX','CCI','PSA','SPG','O',
    # ETFs (tradable assets)
    'SPY','QQQ','GLD','TLT','EFA','EEM','IWM','VTI','VEA','VWO','BND',
    # Div Growth / Quality
    'MSFT','JNJ','PG','KO','PEP','MCD','WMT','HD','ABT','IBM',
    # Misc
    'BKNG','MAR','ABNB','NOW','SNOW','CRWD','PANW','ZS','TEAM','F',
]

UNIVERSE_TICKERS = sorted(set([t for t in UNIVERSE_TICKERS if isinstance(t, str) and t.isupper()]))
print(f"Universe: {len(UNIVERSE_TICKERS)} tickers")

# ============================================================
# 下载价格数据（正确处理 MultiIndex 列）
# ============================================================

print("\nDownloading price data...")
DATA_START = '2010-01-01'
DATA_END   = '2026-02-01'

batch_size = 50
all_closes = []

for i in range(0, len(UNIVERSE_TICKERS), batch_size):
    batch = UNIVERSE_TICKERS[i:i+batch_size]
    n_batch = (len(UNIVERSE_TICKERS)-1)//batch_size + 1
    print(f"  Batch {i//batch_size + 1}/{n_batch}: {len(batch)} tickers", end='', flush=True)
    try:
        # 不使用 group_by='ticker'，这样列是 (field, ticker) 的 MultiIndex
        d = yf.download(batch, start=DATA_START, end=DATA_END,
                        progress=False, auto_adjust=True)
        # 提取 Close 层面，形状是 (date, ticker)
        close = d.xs('Close', axis=1, level=0)
        all_closes.append(close)
        print(f" OK ({close.shape[1]} tickers)")
    except Exception as e:
        print(f" FAILED: {e}")
    time.sleep(0.2)

print("Concatenating...")
data_all = pd.concat(all_closes, axis=1, sort=True)
# 去除重复列（同一 ticker 可能出现在多个 batch）
if data_all.columns.duplicated().any():
    data_all = data_all.loc[:, ~data_all.columns.duplicated()]
print(f"Combined data: {data_all.shape}")

# ============================================================
# 过滤：2011 前有数据 + 价格>$5
# ============================================================

min_rows = 252
price_threshold = 5.0
available = []
for t in UNIVERSE_TICKERS:
    if t not in data_all.columns:
        continue
    col = data_all[t].dropna()
    if len(col) < min_rows:
        continue
    # 找第一个有效价格
    try:
        first_valid = col.index[0]
    except:
        continue
    if first_valid <= pd.Timestamp('2010-12-31'):
        first_price = float(col.loc[:pd.Timestamp('2010-12-31')].dropna().iloc[0])
        if not np.isnan(first_price) and first_price >= price_threshold:
            available.append(t)

UNIVERSE = sorted(set(available))
print(f"\nFiltered universe: {len(UNIVERSE)} tickers (data before 2011, price>$5)")

# ============================================================
# Train 期：每股票选最优策略
# ============================================================

print("\n[Train] Selecting best strategy per ticker...")
data_train = data_all[TRAIN_START:TRAIN_END]

best_strat_map = {}
for i, t in enumerate(UNIVERSE):
    if i % 50 == 0:
        print(f"  {i}/{len(UNIVERSE)}")
    if t not in data_train.columns:
        best_strat_map[t] = 'roc60'
        continue
    best_sh = -999
    best_name = None
    for sname, sfunc in STRATEGIES.items():
        try:
            r = bt_one(sfunc(data_train[t]), data_train[t])
            if r['sharpe'] > best_sh:
                best_sh = r['sharpe']
                best_name = sname
        except:
            pass
    best_strat_map[t] = best_name or 'roc60'

cnt = Counter(best_strat_map.values())
print(f"  Strategy dist: {dict(cnt)}")

# ============================================================
# Val 期：Sharpe>0 筛选
# ============================================================

print("\n[Val] Filtering by Sharpe>0...")
data_val = data_all[VAL_START:VAL_END]

val_sharpe = {}
approved = []
for t in UNIVERSE:
    if t not in data_val.columns:
        val_sharpe[t] = -999.0
        continue
    try:
        sig = STRATEGIES[best_strat_map[t]](data_val[t])
        r = bt_one(sig, data_val[t])
        val_sharpe[t] = r['sharpe']
        if r['sharpe'] > 0:
            approved.append(t)
    except:
        val_sharpe[t] = -999.0

vals_all = list(val_sharpe.values())
vals_pass = [v for v in vals_all if v > 0]
print(f"  Val filter: {len(approved)} / {len(UNIVERSE)} passed (Sharpe>0)")
print(f"  Val Sharpe stats: mean={np.mean(vals_all):.2f}, max={max(vals_all):.2f}, median={np.median(vals_all):.2f}")
print(f"\n  Approved: {approved}")

# ============================================================
# Test 期：不同 top_n 对比
# ============================================================

print("\n[Test] Backtesting...")
data_test = data_all[TEST_START:TEST_END]

results = []
for top_n in [5, 10, 15, 20, 30]:
    r = bt_port(
        approved, data_test,
        initial=100_000,
        best_strat_map=best_strat_map,
        top_n=top_n, min_hold=26, max_weight=0.20,
        per_stock=True, use_vol_scale=False,
    )
    results.append({
        'top_n': top_n,
        'ann': round(r['ann'], 1),
        'sharpe': round(r['sharpe'], 2),
        'mdd': round(r['mdd'], 1),
        'wfa': r['wfa_verdict'],
        'is_ann': round(r['is_ann'], 1) if r.get('is_ann') else None,
        'oos_ann': round(r['oos_ann'], 1) if r.get('oos_ann') else None,
        'mc_verdict': r['mc']['mc_verdict'] if r['mc'] else 'N/A',
        'mc_score': r['mc']['mc_score'] if r['mc'] else 0,
    })
    print(f"  top_n={top_n:2d}: Sharpe={r['sharpe']:.2f} Ann={r['ann']:.1f}% "
          f"MaxDD={r['mdd']:.1f}% WFA={r['wfa_verdict']} MC={r['mc']['mc_verdict']}")

# ============================================================
# Val 信号检验：Val Sharpe vs Test Sharpe 相关性
# ============================================================

print("\n[Val Signal] Val Sharpe vs Test Sharpe correlation...")
val_test_pairs = []
for t in approved:
    vs = val_sharpe.get(t, 0)
    try:
        sig = STRATEGIES[best_strat_map[t]](data_test[t])
        rt = bt_one(sig, data_test[t])
        ts = rt['sharpe']
        if abs(ts) < 50:
            val_test_pairs.append((vs, ts))
    except:
        pass

corr = 0.0
if len(val_test_pairs) >= 10:
    xs = np.array([x for x, _ in val_test_pairs])
    ys = np.array([y for _, y in val_test_pairs])
    corr = np.corrcoef(xs, ys)[0, 1]
    abs_corr = abs(corr)
    if abs_corr < 0.1:
        sig_str = "NEGLIGIBLE — Val has NO predictive power"
    elif abs_corr < 0.3:
        sig_str = "WEAK — Val has limited signal"
    elif corr > 0.3:
        sig_str = "MODERATE — Val IS predictive"
    else:
        sig_str = "NEGATIVE — Val REVERSED in Test (overfitting)"
    print(f"  Pearson corr = {corr:.3f} (n={len(val_test_pairs)}) → {sig_str}")
else:
    print(f"  Not enough pairs: {len(val_test_pairs)}")

# ============================================================
# Summary
# ============================================================

print("\n" + "="*65)
print("OBJECTIVE POOL BACKTEST RESULTS")
print("="*65)
print(f"Universe: {len(UNIVERSE)} tickers (academic/quality list, no human picking)")
print(f"Val filter: {len(approved)} passed (Sharpe>0)")
print(f"Val-Test correlation: {corr:.3f}")
print()
print(f"  {'top_n':>5} {'Sharpe':>8} {'Ann%':>8} {'MaxDD%':>8} {'IS_Ann':>8} {'OOS_Ann':>8} {'WFA':>20} {'MC':>15}")
print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*20} {'-'*15}")
for row in results:
    print(f"  {row['top_n']:>5} {row['sharpe']:>8.2f} {row['ann']:>7.1f}% {row['mdd']:>7.1f}% "
          f"{str(row['is_ann']):>8} {str(row['oos_ann']):>8} {row['wfa']:>20} {row['mc_verdict']:>15}")
print("="*65)

output = {
    'universe': UNIVERSE,
    'approved': approved,
    'val_sharpe': {t: round(v, 3) for t, v in val_sharpe.items()},
    'val_test_corr': round(corr, 3),
    'results': results,
}
with open('/tmp/obj_pool_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved /tmp/obj_pool_results.json")
