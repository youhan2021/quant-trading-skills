"""
Live Weekly Momentum — 实盘模拟
================================
每次只用截至该周收盘时点的数据，模拟真实交易流程：
  每周最后一个交易日收盘 → 算因子排名 → 生成下周一买入清单

Universe: 13 只干净股票（无 info leak）
因子: roc120 + vol20 rank-zscore 等权
持仓: top5 等权
无 Regime Switch

数据: yfinance 日线 → 周线
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ─── Config ───────────────────────────────────────────────────────────────────
UNIVERSE = [
    'SPY', 'QQQ', 'GLD', 'TLT', 'EFA',
    'AMZN', 'LMT', 'V', 'NVDA', 'WMT',
    'GOOGL', 'UUP', 'JPM'
]
DATA_START = "2011-01-01"
DATA_END   = "2026-02-01"
TOP_N      = 5
ROC_WINDOW = 120   # 日
VOL_WINDOW = 20    # 日
TC         = 0.002  # 20bps 单程

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_weekly_prices(tickers, start, end):
    """下载日线 → 重采样为每周最后一个交易日（周线收盘价）"""
    df = yf.download(tickers, start=start, end=end, auto_adjust=False)
    closes = df['Close'].dropna(how='all')
    weekly = closes.resample('W').last().dropna(how='all')
    return weekly


def compute_factors(weekly_prices, roc_win=120, vol_win=20):
    """
    计算每个截面的因子分数（只用截至该截面之前的数据）。
    roc120  → 过去120日收益率
    vol20   → 过去20日波动率（低波=高分）
    rank-zscore 等权组合
    """
    n_periods = weekly_prices.shape[0]
    scores = {}

    for i in range(vol_win, n_periods):
        date = weekly_prices.index[i]
        window_end = date

        # vol20: 取最近20个周收盘（含本周）
        vol_start_idx = max(0, i - vol_win + 1)
        vol_window_prices = weekly_prices.iloc[vol_start_idx:i+1]

        # roc120: 取最近120个日收盘（~24周）
        # 用 dailyprices 做精确120日
        if i < 2:
            continue

        # 获取该周之前的数据做 roc 计算
        daily = yf.download(
            weekly_prices.columns.tolist(),
            start=window_end - timedelta(days=150),
            end=window_end,
            auto_adjust=False,
            progress=False
        )['Close']

        if daily.shape[0] < roc_win:
            continue

        roc = (daily.iloc[-1] / daily.iloc[-roc_win]) - 1
        vol = daily.iloc[-vol_win:].std() * np.sqrt(252)

        # rank-zscore
        def rzscore(series):
            rank = series.rank()
            return (rank - rank.mean()) / rank.std()

        roc_z   = rzscore(roc)
        vol_z   = rzscore(-vol)   # 低波高分
        combined = roc_z + vol_z

        top = combined.nlargest(TOP_N)
        scores[date] = {
            'top5':   top.index.tolist(),
            'scores': top.to_dict()
        }

    return scores


def backtest(signals, weekly_prices, tc=0.0):
    """
    信号驱动回测：
      每周 signal['top5'] → 下周持仓
      收益 = (C[j+1] - C[j]) / C[j]
    """
    rets_list = []
    dates = sorted(signals.keys())
    holdings_list = [None]  # 持仓历史

    for i, date in enumerate(dates):
        prev_date = dates[i-1] if i > 0 else None

        # 新持仓
        current_holdings = set(signals[date]['top5'])

        if i == 0:
            holdings_list[0] = current_holdings
            continue

        prev_holdings = holdings_list[-1]

        # 换手计算
        sold = prev_holdings - current_holdings
        bought = current_holdings - prev_holdings
        n_sold   = len(sold)
        n_bought = len(bought)
        turnover = (n_sold + n_bought) / max(len(prev_holdings), 1)
        did_trade = (n_sold + n_bought) > 0

        # 周收益（所有周都有，即使不换手）
        w1 = weekly_prices.loc[prev_date]
        w2 = weekly_prices.loc[date]
        if pd.isna(w1).any() or pd.isna(w2).any():
            holdings_list.append(current_holdings)
            continue

        ret = (w2 / w1) - 1
        hold = list(current_holdings)
        port_ret = ret[hold].mean()

        # 交易成本（只在换手时扣除）
        tc_cost = tc * turnover if did_trade else 0
        port_ret -= tc_cost

        rets_list.append({
            'date':       date,
            'ret':        port_ret,
            'holdings':   hold,
            'turnover':   turnover if did_trade else 0,
            'did_trade':  did_trade,
        })

        holdings_list.append(current_holdings)

    if not rets_list:
        return pd.DataFrame(), pd.Series([1.0])

    df = pd.DataFrame(rets_list)
    df.set_index('date', inplace=True)

    # equity curve — 所有周都记录
    equity = pd.Series(index=df.index, dtype=float)
    equity.iloc[0] = 1.0 * (1 + df['ret'].iloc[0])
    for j in range(1, len(df)):
        equity.iloc[j] = equity.iloc[j-1] * (1 + df['ret'].iloc[j])

    return df, equity
def calc_metrics(df_rets, equity_series):
    """计算年化指标（df_rets 有 ret, turnover 列）"""
    if len(df_rets) < 10:
        return {}

    rets = df_rets['ret']
    ann_ret  = rets.mean() * 52
    ann_std  = rets.std()  * np.sqrt(52)
    sharpe   = ann_ret / ann_std if ann_std > 0 else 0

    # MaxDD
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min()

    # 年换手率
    total_turnover = df_rets['turnover'].sum()
    avg_annual_turnover = total_turnover / (len(rets) / 52)

    return {
        'Sharpe':       round(sharpe, 2),
        'AnnRet':       f"{ann_ret*100:.1f}%",
        'MaxDD':        f"{max_dd*100:.1f}%",
        'AnnVol':       f"{ann_std*100:.1f}%",
        'AvgTurnover':  round(avg_annual_turnover, 2),
        'TotalTrades':  len(df_rets),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Live Weekly Momentum — 实盘模拟")
    print("=" * 60)

    # 1. 下载数据
    print(f"\n📥 下载 {len(UNIVERSE)} 只股票日线...")
    daily = yf.download(
        UNIVERSE, start=DATA_START, end=DATA_END,
        auto_adjust=False, progress=False
    )['Close']

    weekly = daily.resample('W').last()

    print(f"  日线: {daily.shape}, 周线: {weekly.shape}")

    # 2. 计算因子信号（每周滚动）
    print(f"\n🧮 计算因子信号 (roc={ROC_WINDOW}, vol={VOL_WINDOW})...")
    n_weeks = weekly.shape[0]
    signals = {}

    for i in range(VOL_WINDOW + 5, n_weeks - 1):
        date = weekly.index[i]

        # 下载截至 date 的数据（只用过去150天，避免太大）
        window_start = date - timedelta(days=160)
        d = daily.loc[:date].iloc[-150:]

        if d.shape[0] < ROC_WINDOW:
            continue

        roc = (d.iloc[-1] / d.iloc[-ROC_WINDOW]) - 1
        vol = d.iloc[-VOL_WINDOW:].std() * np.sqrt(252)

        def rzscore(s):
            r = s.rank()
            return (r - r.mean()) / r.std()

        combined = rzscore(roc) + rzscore(-vol)
        top = combined.nlargest(TOP_N)

        signals[date] = {
            'top5':   top.index.tolist(),
            'scores': top.to_dict()
        }

    print(f"  生成信号: {len(signals)} 周")

    # 3. 回测（0 TC）
    print(f"\n📊 回测 (TC=0)...")
    df0, eq0 = backtest(signals, weekly, tc=0.0)

    # 4. 回测（20bps TC）
    print(f"📊 回测 (TC=20bps)...")
    df2, eq2 = backtest(signals, weekly, tc=0.002)

    # 5. 年度分解（20bps）
    print(f"\n📅 年度明细 (20bps):")
    df2['year'] = df2.index.year
    annual = df2.groupby('year').agg(
        年度收益=('ret', 'sum'),
        交易次数=('turnover', lambda x: (x > 0).sum())
    )

    # SPY B&H
    spy = yf.download('SPY', start=DATA_START, end=DATA_END,
                      auto_adjust=False, progress=False)['Close'].squeeze()
    spy_weekly = spy.resample('W').last().dropna()
    spy_rets = spy_weekly.pct_change().dropna()
    spy_ann   = spy_rets.mean() * 52
    spy_std   = spy_rets.std()  * np.sqrt(52)
    spy_sharpe = float(spy_ann / spy_std)
    spy_maxdd  = float((spy_weekly / spy_weekly.cummax()).min() - 1)

    print(f"\n{'年份':<6} {'收益':>8} {'换手次数':>8}")
    print("-" * 28)
    for yr, row in annual.iterrows():
        sign = '+' if row['年度收益'] > 0 else ''
        print(f"{yr:<6} {sign}{row['年度收益']*100:>6.1f}%  {int(row['交易次数']):>6}")

    # 6. 总指标
    m0 = calc_metrics(df0, eq0)
    m2 = calc_metrics(df2, eq2)

    print(f"\n{'指标':<20} {'无TC':>10} {'20bps':>10}")
    print("-" * 42)
    for k in ['Sharpe', 'AnnRet', 'MaxDD', 'AnnVol']:
        print(f"{k:<20} {m0.get(k,'-'):>10} {m2.get(k,'-'):>10}")

    print(f"\n{'SPY B&H':<20} {round(float(spy_sharpe),2):>10}")
    print(f"  (SPY 年化 {float(spy_ann)*100:.1f}%, MaxDD {float(spy_maxdd)*100:.1f}%)")

    # 7. 存入缓存供后续分析
    result = {
        'config': {
            'universe':    UNIVERSE,
            'top_n':       TOP_N,
            'roc_window':  ROC_WINDOW,
            'vol_window': VOL_WINDOW,
            'frequency':   'weekly',
            'regime_switch': False
        },
        'metrics_0tc':  m0,
        'metrics_20bps': m2,
        'spy_sharpe':   round(spy_sharpe, 2),
        'annual_detail': annual.to_dict(),
        'n_signals':    len(signals),
        'n_trades':     len(df2),
    }

    out = '/tmp/live_weekly_result.json'
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✅ 结果已存 → {out}")