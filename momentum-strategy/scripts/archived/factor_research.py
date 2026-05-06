"""
因子信号研究 — 找真正有 Val→Test 预测力的因子

测试因子：
1. 动量（ROC20/60）— baseline，被证伪
2. 质量（ROE, GrossMargin, FCF Yield）
3. 价值（P/E, P/B, EV/EBITDA）
4. 低波动（Realized Vol）
5. 规模（Log Market Cap）
6. 营收增速（Revenue Growth）
7. 股息率（Dividend Yield）

三段式：
- Val(2016-2021): 算因子值 → 因子五分位分组 → Test(2021-2026) 各组表现
- 检验：因子 IC (Information Coefficient) = corr(Val因子值, Test收益)
"""

import sys
sys.path.insert(0, '/home/ubuntu/.hermes/skills/quant-trading-momentum/scripts')

import json, math, time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from momentum_backtest import STRATEGIES, bt_one
from collections import defaultdict

warnings.filterwarnings('ignore')

TRAIN_START = '2011-01-01'
TRAIN_END   = '2016-01-01'
VAL_START   = '2016-01-01'
VAL_END     = '2021-01-01'
TEST_START  = '2021-01-01'
TEST_END    = '2026-02-01'

# ============================================================
# 候选 Universe（103只学术名单）
# ============================================================

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

# Filter: 2011前有数据
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

# ============================================================
# 下载基本面数据（yfinance .info）
# ============================================================

print("\nDownloading fundamental data...")
# 用最近5年的数据代表 Val 期基本面的中位数
FUNDAMENTALS = {}
failed = []

for i, t in enumerate(UNIVERSE):
    if i % 20 == 0:
        print(f"  {i}/{len(UNIVERSE)}", end='', flush=True)
    try:
        ticker = yf.Ticker(t)
        info = ticker.info
        # 提取多个维度的基本面数据
        FUNDAMENTALS[t] = {
            # 估值
            'pe':           info.get('trailingPE') or info.get('forwardPE') or np.nan,
            'pb':           info.get('priceToBook') or np.nan,
            'ps':           info.get('priceToSalesTrailing12Months') or np.nan,
            'ev_ebitda':    info.get('enterpriseToEbitda') or np.nan,
            # 盈利质量
            'roe':          info.get('returnOnEquity') or np.nan,
            'roa':          info.get('returnOnAssets') or np.nan,
            'gross_margin': info.get('grossMargins') or np.nan,
            'op_margin':    info.get('operatingMargins') or np.nan,
            'net_margin':   info.get('netProfitMargin') or np.nan,
            # 成长
            'revenue_growth':  info.get('revenueGrowth') or info.get('earningsGrowth') or np.nan,
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth') or np.nan,
            # 财务健康
            'de_ratio':     info.get('debtToEquity') or np.nan,
            'current_ratio':info.get('currentRatio') or np.nan,
            'quick_ratio':  info.get('quickRatio') or np.nan,
            # 股息/回购
            'div_yield':    info.get('dividendYield') or 0.0,
            'buyback_yield': info.get('buybackYield') or 0.0,
            # 规模/流动性
            'mkt_cap':      info.get('marketCap') or np.nan,
            'avg_volume':   info.get('averageVolume') or np.nan,
            # 波动率
            'beta':         info.get('beta') or np.nan,
            # FCF
            'fcf_yield':    info.get('freeCashflow') or np.nan,
        }
        print('.', end='', flush=True)
    except Exception as e:
        FUNDAMENTALS[t] = {k: np.nan for k in [
            'pe','pb','ps','ev_ebitda','roe','roa','gross_margin','op_margin',
            'net_margin','revenue_growth','earnings_quarterly_growth',
            'de_ratio','current_ratio','quick_ratio','div_yield','buyback_yield',
            'mkt_cap','avg_volume','beta','fcf_yield'
        ]}
        failed.append(t)
        print('x', end='', flush=True)

print(f"\n  Failed: {len(failed)} tickers")

# ============================================================
# 计算 Val 期因子值（使用 Val 期数据计算）
# ============================================================

print("\nComputing factors...")

data_val   = data_all[VAL_START:VAL_END]
data_test  = data_all[TEST_START:TEST_END]

# Val 期因子（使用 Val 期价格数据计算）
val_returns = {}
for t in UNIVERSE:
    if t not in data_val.columns:
        val_returns[t] = np.nan
        continue
    try:
        px = data_val[t].dropna()
        if len(px) < 50:
            val_returns[t] = np.nan
            continue
        ret = (px.iloc[-1] / px.iloc[0]) ** (252.0 / len(px)) - 1
        val_returns[t] = ret
    except:
        val_returns[t] = np.nan

# Val 期波动率
val_vol = {}
for t in UNIVERSE:
    if t not in data_val.columns:
        val_vol[t] = np.nan
        continue
    try:
        ret = data_val[t].dropna().pct_change().dropna()
        if len(ret) < 50:
            val_vol[t] = np.nan
            continue
        vol = float(ret.std()) * math.sqrt(252)
        val_vol[t] = vol
    except:
        val_vol[t] = np.nan

# 规模因子
val_mktcap = {}
for t in UNIVERSE:
    mc = FUNDAMENTALS.get(t, {}).get('mkt_cap', np.nan)
    val_mktcap[t] = math.log(mc) if not np.isnan(mc) and mc > 0 else np.nan

# 价值/质量因子直接用 info 字段（来自最新季报，滞后但稳定）
factor_data = {}
FACTOR_LIST = [
    'pe','pb','ps','ev_ebitda',
    'roe','roa','gross_margin','op_margin','net_margin',
    'revenue_growth','earnings_quarterly_growth',
    'de_ratio','current_ratio','quick_ratio',
    'div_yield','buyback_yield',
    'beta',
]

for t in UNIVERSE:
    factor_data[t] = FUNDAMENTALS.get(t, {})

# ============================================================
# Test 期收益（用于 IC 计算）
# ============================================================

test_returns = {}
for t in UNIVERSE:
    if t not in data_test.columns:
        test_returns[t] = np.nan
        continue
    try:
        px = data_test[t].dropna()
        if len(px) < 50:
            test_returns[t] = np.nan
            continue
        ret = (px.iloc[-1] / px.iloc[0]) ** (252.0 / len(px)) - 1
        test_returns[t] = ret
    except:
        test_returns[t] = np.nan

# ============================================================
# IC 计算：因子与 Test 收益的相关性
# ============================================================

print("\n" + "="*70)
print("FACTOR IC ANALYSIS (Val Factor Value vs Test Return)")
print("="*70)
print(f"{'Factor':<25} {'IC':>8} {'IC>0.05?':>10} {'TopQ Ann%':>12} {'BotQ Ann%':>12} {'Diff':>8}")
print("-"*70)

results = {}

# 首先测试简单价格动量因子（我们的baseline）
val_roc20 = {}
val_roc60 = {}
for t in UNIVERSE:
    if t not in data_val.columns:
        val_roc20[t] = np.nan
        val_roc60[t] = np.nan
        continue
    px = data_val[t].dropna()
    if len(px) < 60:
        val_roc20[t] = np.nan
        val_roc60[t] = np.nan
        continue
    try:
        val_roc20[t] = float(px.pct_change(20).iloc[-1])
        val_roc60[t] = float(px.pct_change(60).iloc[-1])
    except:
        val_roc20[t] = np.nan
        val_roc60[t] = np.nan

all_factors = {
    'roc20': val_roc20,
    'roc60': val_roc60,
    'vol20': val_vol,
    'mktcap': val_mktcap,
}
for fkey in FACTOR_LIST:
    all_factors[fkey] = {t: factor_data[t].get(fkey, np.nan) for t in UNIVERSE}

for fname, fvals in all_factors.items():
    # 构建 (因子值, Test收益) 对
    pairs = []
    for t in UNIVERSE:
        fv = fvals.get(t, np.nan)
        tr = test_returns.get(t, np.nan)
        if not (np.isnan(fv) or np.isnan(tr) or abs(tr) > 10):
            pairs.append((fv, tr))

    if len(pairs) < 15:
        results[fname] = {'ic': np.nan, 'n': len(pairs)}
        continue

    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])

    # Pearson IC
    if xs.std() > 1e-9 and ys.std() > 1e-9:
        ic = np.corrcoef(xs, ys)[0, 1]
    else:
        ic = 0.0

    # 五分位分组回测
    valid = [(xs[i], ys[i]) for i in range(len(pairs))]
    valid_sorted = sorted(valid, key=lambda x: x[0])
    n = len(valid_sorted)
    q = n // 5
    top_q = [y for _, y in valid_sorted[-q:]]
    bot_q = [y for _, y in valid_sorted[:q]]
    top_ann = np.mean(top_q) * 100
    bot_ann = np.mean(bot_q) * 100
    diff = top_ann - bot_ann

    results[fname] = {
        'ic': ic,
        'n': len(pairs),
        'top_q_ann': top_ann,
        'bot_q_ann': bot_ann,
        'diff': diff,
    }

    sig = "✓" if abs(ic) > 0.05 else " "
    print(f"  {fname:<23} {ic:>+8.3f} {sig:>10} {top_ann:>+11.1f}% {bot_ann:>+11.1f}% {diff:>+7.1f}%")

print("="*70)

# ============================================================
# 多因子组合：动量 + 价值 + 质量
# ============================================================

print("\n" + "="*70)
print("MULTI-FACTOR PORTFOLIO TEST")
print("="*70)

# 选择 IC > 0.05 的因子组合
# 标准化各因子（z-score），组合打分

def normalize_factor(fvals_dict):
    """Z-score normalization"""
    vals = [(t, v) for t, v in fvals_dict.items() if not np.isnan(v)]
    if len(vals) < 5:
        return {t: 0.0 for t in fvals_dict}
    xs = np.array([v for _, v in vals])
    mu, sigma = np.mean(xs), np.std(xs)
    if sigma < 1e-9:
        return {t: 0.0 for t in fvals_dict}
    result = {}
    for t, v in fvals_dict.items():
        if np.isnan(v):
            result[t] = 0.0
        else:
            result[t] = (v - mu) / sigma
    return result

# 各因子方向（因子高了是好还是坏？）
# 高roc → 动量因子，高了好
# 高roe/gross_margin → 质量好，高了好
# 高pe/pb → 估值贵，低了可能好（价值因子）
# 高vol → 风险高，低了可能好（低波因子）
# 高div_yield → 高股息，高了好
# 高mktcap → 大盘，小了可能好（规模因子）

factor_directions = {
    'roc20': +1,     # 动量：高 ROC → 高预期收益
    'roc60': +1,     # 动量：高 ROC → 高预期收益
    'vol20': -1,     # 低波：高 vol → 低预期收益（低波效应）
    'roe':   +1,     # 质量：高 ROE → 高预期收益
    'roa':   +1,     # 质量：高 ROA → 高预期收益
    'gross_margin': +1, # 质量：高毛利率 → 高预期收益
    'net_margin':   +1, # 质量：高净利率 → 高预期收益
    'revenue_growth': +1, # 成长：高增长 → 高预期收益
    'pe':    -1,     # 价值：低 PE → 高预期收益
    'pb':    -1,     # 价值：低 PB → 高预期收益
    'ps':    -1,     # 价值：低 PS → 高预期收益
    'div_yield': +1,  # 价值：高股息 → 高预期收益
    'beta':  -1,     # 低波：低 beta → 高预期收益
}

# 只用有 IC 数据的因子
multi_factor_scores = {t: 0.0 for t in UNIVERSE}
factor_weights = {}

for fname, fvals in all_factors.items():
    if fname not in factor_directions:
        continue
    norm = normalize_factor(fvals)
    direction = factor_directions[fname]
    ic = results.get(fname, {}).get('ic', 0)
    if abs(ic) < 0.02:
        continue  # 跳过 IC 太低的因子
    weight = ic  # 用 IC 作为权重
    for t in UNIVERSE:
        multi_factor_scores[t] += direction * norm[t] * weight
    factor_weights[fname] = (direction, weight)

print(f"Active factors (|IC|>0.02) with IC weights:")
for fname, (d, w) in sorted(factor_weights.items(), key=lambda x: abs(x[1][1]), reverse=True):
    ic = results.get(fname, {}).get('ic', 0)
    print(f"  {fname:<25} dir={d:>+2d}  weight={w:>+6.3f}  IC={ic:>+6.3f}")

# Top/Bottom 30 多因子组合的 Test 收益
mf_pairs = [(t, mf) for t, mf in multi_factor_scores.items()]
mf_pairs.sort(key=lambda x: x[1], reverse=True)

# Top 30
top_n_vals = mf_pairs[:30]
bot_n_vals = mf_pairs[-30:]

# 计算 top/bottom 在 Test 期的平均收益
def portfolio_ann(tickers, data):
    rets = []
    for t in tickers:
        if t not in data.columns:
            continue
        try:
            px = data[t].dropna()
            if len(px) < 50:
                continue
            r = (px.iloc[-1] / px.iloc[0]) ** (252.0 / len(px)) - 1
            rets.append(r)
        except:
            pass
    if not rets:
        return np.nan, 0
    return np.mean(rets), len(rets)

top_ann, top_n = portfolio_ann([t for t, _ in top_n_vals], data_test)
bot_ann, bot_n = portfolio_ann([t for t, _ in bot_n_vals], data_test)

print(f"\nMulti-Factor Top30 Test Ann: {top_ann*100:+.1f}% (n={top_n})")
print(f"Multi-Factor Bot30 Test Ann: {bot_ann*100:+.1f}% (n={bot_n})")
print(f"Difference: {(top_ann-bot_ann)*100:+.1f}%")

# 对比纯动量 top30
mom_pairs = [(t, val_roc60.get(t, 0)) for t in UNIVERSE]
mom_pairs.sort(key=lambda x: x[1], reverse=True)
mom_top = [t for t, _ in mom_pairs[:30]]
mom_top_ann, _ = portfolio_ann(mom_top, data_test)
print(f"\nPure Momentum Top30 Test Ann: {mom_top_ann*100:+.1f}%")

# ============================================================
# 保存
# ============================================================

output = {
    'factor_ic': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                      for kk, vv in v.items()}
                  for k, v in results.items()},
    'factor_weights': {k: (int(v[0]), float(v[1])) for k, v in factor_weights.items()},
    'multi_factor_top30_ann': float(top_ann * 100),
    'multi_factor_bot30_ann': float(bot_ann * 100),
    'pure_momentum_top30_ann': float(mom_top_ann * 100),
}

with open('/tmp/factor_ic_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to /tmp/factor_ic_results.json")
