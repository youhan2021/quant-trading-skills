#!/usr/bin/env python3
"""
Rolling Walk-Forward — 5步走 Multi-Factor
==========================================
1. 大候选池 (117只 yfinance股票)
2. 基本面初筛 (yfinance: ROE>0, D/E<80%)
3. Val期因子IC (2016-2021) → 因子权重
4. 滚动Test (2021/2022/2023/2024 各Q1) — 用历史截面因子, LOCKED权重
5. 汇总: 平均Sharpe vs SPY
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
    df = yf.download(batch, start='2009-01-01', end='2026-03-01',
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
# 3. 基本面 (yfinance, 当前时点)
# ------------------------------------------------------------------
print("下载基本面...")
fund = {}
for i, t in enumerate(UNIVERSE):
    try:
        info = yf.Ticker(t).info
        fund[t] = {
            'roe': info.get('returnOnEquity'),
            'de': info.get('debtToEquity'),
            'bv': info.get('bookValue'),
        }
    except:
        fund[t] = {}
    if (i+1) % 30 == 0:
        print(f"  {i+1}/{len(UNIVERSE)}")
print(f"  共 {len(fund)} 只")

# ------------------------------------------------------------------
# 4. Val期因子IC (滚动月度)
# ------------------------------------------------------------------
print("\n=== Val IC (2016-2021) ===")

def calc_ic(returns, factor_vals):
    mask = ~(returns.isna() | factor_vals.isna())
    if mask.sum() < 5:
        return 0.0
    return float(returns[mask].corr(factor_vals[mask]))

factor_ics = {f: [] for f in ['roc20','roc60','roc120','vol20','vol60']}

val_months = monthly['2016':'2021'].index
for dt in val_months:
    dt_str = str(dt)[:10]
    sub = week[:dt_str]
    if len(sub) < 60:
        continue

    # 计算因子
    fvals = {}
    for wk in [20, 60, 120]:
        if len(sub) >= wk:
            fvals[f'roc{wk}'] = (sub.iloc[-1] / sub.iloc[-wk] - 1)
    ret = sub.pct_change().dropna()
    for wk in [20, 60]:
        if len(ret) >= wk:
            fvals[f'vol{wk}'] = ret.rolling(wk).std().iloc[-1]

    try:
        mret = monthly.loc[dt_str]
    except:
        continue

    for fn in fvals:
        ic = calc_ic(mret, fvals[fn])
        factor_ics[fn].append(ic)

print("IC统计:")
factor_dir = {}  # 因子方向
for fn, ics in factor_ics.items():
    arr = np.array(ics)
    mean_ic = np.nanmean(arr)
    frac = np.nanmean(arr > 0)
    n = len(arr)
    print(f"  {fn:8s}: IC={mean_ic:+.4f}, frac={frac:.2f}, n={n}")
    if n >= 30:
        if frac > 0.55:
            factor_dir[fn] = +1  # 高因子值 → 高收益
        elif frac < 0.45:
            factor_dir[fn] = -1  # 高因子值 → 低收益 (低波动 = 好)

print(f"\n因子方向: {factor_dir}")

# ------------------------------------------------------------------
# 5. 滚动Test
# ------------------------------------------------------------------
print("\n=== 滚动 Walk-Forward Test ===")

test_windows = [
    ('2021-01-01', '2022-01-01'),
    ('2022-01-01', '2023-01-01'),
    ('2023-01-01', '2024-01-01'),
    ('2024-01-01', '2025-01-01'),
]

# SPY benchmark
spy_df = yf.download('SPY', start='2021-01-01', end='2026-03-01',
                     auto_adjust=False, progress=False)['Close'].squeeze()
if spy_df.index.tz:
    spy_df = spy_df.tz_localize(None)
spy_week = spy_df.resample('W').last()

all_results = []
for tw_start, tw_end in test_windows:
    print(f"\n窗口 {tw_start[:4]}")

    # 基本面筛选
    pool = [t for t in prices.columns if t in fund and
            (fund[t].get('roe') or 0) > 0 and
            (fund[t].get('de') or 999) < 80]
    print(f"  池子: {len(pool)} 只")
    if len(pool) < 5:
        continue

    # 计算截面因子得分
    sub_w = week[pool]
    scores = pd.DataFrame(index=pool)

    for wk in [20, 60, 120]:
        if len(sub_w) >= wk:
            scores[f'roc{wk}'] = (sub_w.iloc[-1] / sub_w.iloc[-wk] - 1)
    ret = sub_w.pct_change().dropna()
    for wk in [20, 60]:
        if len(ret) >= wk:
            scores[f'vol{wk}'] = ret.rolling(wk).std().iloc[-1]

    # 综合得分 = Σ IC方向 * IC强度 * 因子值
    composite = pd.Series(0.0, index=pool)
    ic_sum = 0
    for fn, direction in factor_dir.items():
        if fn in scores.columns:
            mean_ic = np.nanmean(factor_ics[fn])
            composite += direction * abs(mean_ic) * scores[fn].fillna(0)
            ic_sum += abs(mean_ic)

    # 选 top5
    top5 = composite.nlargest(5).index.tolist()
    print(f"  选股: {top5}")

    # 回测
    tw = prices[top5][tw_start:tw_end].resample('W').last().pct_change().dropna().mean(axis=1)
    ann = float((1 + tw.mean())**52 - 1)
    vol = float(tw.std() * np.sqrt(52))
    sh = ann / vol if vol > 0 else 0
    cum = (1+tw).cumprod()
    mdd = float((cum/cum.cummax()-1).min())

    # SPY
    spy_tw = spy_week[tw_start:tw_end].pct_change().dropna()
    spy_ann = float((1 + spy_tw.mean())**52 - 1)
    spy_vol = float(spy_tw.std() * np.sqrt(52))
    spy_sh = spy_ann / spy_vol if spy_vol > 0 else 0

    print(f"  Sharpe: {sh:.2f}, Ann: {ann*100:.1f}%, MaxDD: {mdd*100:.1f}%  |  SPY: {spy_sh:.2f}")
    all_results.append({
        'window': tw_start[:4],
        'sharpe': round(sh,2),
        'ann': round(ann*100,1),
        'max_dd': round(mdd*100,1),
        'spy_sharpe': round(spy_sh,2),
        'stocks': top5,
    })

# ------------------------------------------------------------------
# 6. 汇总
# ------------------------------------------------------------------
print("\n" + "="*60)
print("滚动 Walk-Forward 汇总")
print("="*60)
print(f"{'年份':6s} {'Sharpe':>8} {'Ann%':>8} {'MaxDD%':>8} {'SPY':>8}")
print('-'*60)
spy_sharpes = []
for r in all_results:
    print(f"{r['window']:6s} {r['sharpe']:8.2f} {r['ann']:7.1f}% {r['max_dd']:7.1f}% {r['spy_sharpe']:8.2f}")
    spy_sharpes.append(r['spy_sharpe'])
print('-'*60)
shs = [r['sharpe'] for r in all_results]
print(f"{'平均':6s} {np.mean(shs):8.2f}              {np.mean(spy_sharpes):8.2f}")

with open('/tmp/rolling_wfa_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n已保存: /tmp/rolling_wfa_results.json")
