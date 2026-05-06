"""
Rolling Retrain Regime-Switching Backtest
==========================================
完全自包含版本，不依赖 comprehensive_backtest

逻辑：
- 固定策略（训一次跑到底）  vs  Regime-Switching（Sharpe<0连续4月→切换防御）
- 防御状态：50% Bond(IEF) + 50% LowVol(SPLV)
- Regime检测：滚动24个月 Sharpe
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
import os

# ─── Config ────────────────────────────────────────────────────────────────

START         = "2011-01-01"
END           = "2026-01-31"
TOP_N         = 5
REGIME_WINDOW = 24   # Regime 检测窗口（月）
SWITCH_THR    = 4    # 连续N月 Sharpe<0 才切换
TC            = 0.002

BOND_TICKER   = "IEF"
LOWVOL_TICKER = "SPLV"

# ─── 数据获取 ──────────────────────────────────────────────────────────────

def get_prices():
    cache = "/tmp/rolling_retrain_prices.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"  Loaded from cache: {df.shape}")
        return df

    tickers = sorted(set([
        "SPY","QQQ","GLD","TLT","EFA","AMZN","LMT","V","NVDA",
        "WMT","GOOGL","UUP","JPM","XOM","PG","JNJ","KO","PEP",
        "MCD","COST","HD","NKE","UNH","BMY","PFE","ABBV","MRK",
        "AMGN","GILD","BIIB","ISRG","MDT","SYK","ZTS",
        "SPXL","SPXS","TQQQ","SQQQ","SOXL","LABU","FAS","FAZ",
        "IWM","VTI","IWB","IWD","IWF","EEM","VWO","DBA",
        "UNG","USO","SLV","IAU","CPER","PDBC",
        "VIXY","SVIX","UVXY","TVIX","ZIV","VXZ",
        "IEF","TLT","LQD","HYG","EMB","BND","AGG","SCHZ",
        "SPLV","USMV","SPHD","IDV","PRF","SIZE","QUAL","VLUE","USSG",
        BOND_TICKER, LOWVOL_TICKER
    ]))

    print(f"Fetching {len(tickers)} tickers from yfinance...")
    data = yf.download(tickers, start=START, end=END, auto_adjust=False, progress=False)
    closes = data['Close'].dropna(how='all', axis=1)
    closes.index = closes.index.tz_localize(None) if closes.index.tz else closes.index
    resampled = closes.resample('ME').last()
    resampled.to_csv(cache.replace('.parquet', '.csv'))
    print(f"  Saved: {resampled.shape}")
    return resampled


# ─── 因子计算 ─────────────────────────────────────────────────────────────

def rolling_roc(series: pd.Series, window: int) -> pd.Series:
    """滚动收益率：最新收盘 / window天前收盘 - 1"""
    return series / series.shift(window) - 1


def rolling_vol(series: pd.Series, window: int) -> pd.Series:
    """滚动波动率（年化）"""
    return series.pct_change().rolling(window).std() * np.sqrt(252)


def rank_zscore(s: pd.Series) -> pd.Series:
    """Series rank-percentile → zscore"""
    rank = s.rank(pct=True, na_option='bottom')
    return (rank - rank.mean()) / rank.std()


def compute_factor_scores(prices: pd.DataFrame, cutoff: str) -> pd.Series:
    """用 cutoff 之前的数据算 roc120+vol20 组合因子打分"""
    hist = prices.loc[:cutoff].dropna(how='all', axis=1)
    if hist.shape[0] < 130:
        return pd.Series(dtype=float)

    scores = {}
    for ticker in hist.columns:
        s = hist[ticker].dropna()
        if len(s) < 130:
            continue
        try:
            roc = rolling_roc(s, 120)
            vol = rolling_vol(s, 20)
            if roc.iloc[-1] > 0 and vol.iloc[-1] > 0:  # 只保留有效数据点
                scores[ticker] = float(roc.iloc[-1]) * 0.5 + float(vol.iloc[-1]) * 0.5
        except:
            continue

    if not scores:
        return pd.Series(dtype=float)

    s = pd.Series(scores)
    return rank_zscore(s)


# ─── 月度收益矩阵 ──────────────────────────────────────────────────────────

def build_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """月度收益率矩阵：(C[t] - C[t-1]) / C[t-1]"""
    C = prices.dropna(how='all', axis=1)
    rets = C.pct_change().iloc[1:]  # 第一行NaN不要
    rets.index = rets.index.tz_localize(None) if rets.index.tz else rets.index
    return rets


def select_top5(scores: pd.Series, prices: pd.DataFrame, date: pd.Timestamp, n: int = TOP_N) -> list:
    """按因子打分选 top5，只选有价格数据的"""
    valid = scores.dropna().nlargest(n)
    result = []
    for ticker in valid.index:
        if ticker in prices.columns:
            try:
                p = prices.loc[:date, ticker].dropna()
                if len(p) > 0 and p.iloc[-1] > 0:
                    result.append(ticker)
            except:
                continue
    return result


# ─── 滚动 Sharpe ────────────────────────────────────────────────────────────

def rolling_sharpe_12m(rets: pd.Series, window: int = REGIME_WINDOW) -> pd.Series:
    """滚动 12 个月 Sharpe（年化）"""
    roll_mean = rets.rolling(window, min_periods=max(6, window//2)).mean() * 12
    roll_std  = rets.rolling(window, min_periods=max(6, window//2)).std() * np.sqrt(12)
    return roll_mean / roll_std


# ─── 绩效统计 ──────────────────────────────────────────────────────────────

def compute_stats(rets: pd.Series):
    if len(rets) == 0 or rets.std() == 0:
        return {'sharpe': 0, 'ann': 0, 'maxdd': 0, 'n': len(rets)}
    ann   = rets.mean() * 12
    std   = rets.std() * np.sqrt(12)
    sharpe = ann / std if std > 0 else 0
    cummax = (1 + rets).cumprod().cummax()
    drawdown = (1 + rets).cumprod() / cummax - 1
    maxdd = drawdown.min()
    return {'sharpe': round(sharpe, 3), 'ann': round(ann*100, 1), 'maxdd': round(maxdd*100, 1), 'n': len(rets)}


def calc_turnover(holdings_history: list) -> float:
    if len(holdings_history) < 2:
        return 0.0
    total = 0
    for i in range(1, len(holdings_history)):
        old = set(holdings_history[i-1]) or set()
        new = set(holdings_history[i]) or set()
        union = old | new
        if len(union) > 0:
            total += len(old ^ new) / len(union)
    return total / max(len(holdings_history) - 1, 1)


# ─── 主回测 ─────────────────────────────────────────────────────────────────

def run_backtest(prices: pd.DataFrame, ret_mat: pd.DataFrame, regime_switch: bool = False):
    """
    模拟 Monthly rebalance：
    - 每月末用当前可用的所有历史数据计算因子
    - 下月第一个交易日开盘持仓，等权
    - 收益 = (C[t+1] - C[t]) / C[t]（次月开盘相对当月收盘）
    """

    monthly_ends = ret_mat.index.tolist()

    # 跳过前130天，保证 roc120 有足够历史
    first_valid = min(i for i, d in enumerate(monthly_ends) if d >= pd.Timestamp("2013-01-01"))
    monthly_ends = monthly_ends[first_valid:]

    print(f"  Backtest period: {monthly_ends[0].strftime('%Y-%m')} → {monthly_ends[-1].strftime('%Y-%m')}, N={len(monthly_ends)}")

    # ── 预计算每个月末的 top5 ──────────────────────────────────────────
    print("  Precomputing factor scores...")
    top5_cache = {}
    regime_sharpe_series = pd.Series(dtype=float, index=monthly_ends)

    for date in monthly_ends:
        scores = compute_factor_scores(prices, date.strftime('%Y-%m-%d'))
        if len(scores) >= TOP_N:
            top5_cache[date] = select_top5(scores, prices, date, TOP_N)

    # ── 主循环 ──────────────────────────────────────────────────────────
    rets_fixed   = []
    rets_switch  = []
    holdings_fixed  = []
    holdings_switch = []

    strategy_fixed_rets = []   # 用于 rolling Sharpe 计算
    in_defensive = False
    switch_dates = []
    regime_log = []

    current_holdings_fixed  = []
    current_holdings_switch = []

    # ── 主循环 ──────────────────────────────────────────────────────────
    dates_to_process = monthly_ends[first_valid:]  # list of timestamps

    rets_fixed   = []
    rets_switch  = []
    holdings_fixed  = []
    holdings_switch = []

    strategy_fixed_rets = []   # 用于 rolling Sharpe 计算
    in_defensive = False
    switch_dates = []
    regime_log   = []

    current_holdings_fixed  = []
    current_holdings_switch = []

    for i, date in enumerate(dates_to_process):
        holdings = top5_cache.get(date, [])
        if len(holdings) < TOP_N:
            rets_fixed.append(0.0)
            rets_switch.append(0.0)
            holdings_fixed.append(list(current_holdings_fixed))
            holdings_switch.append(list(current_holdings_switch))
            regime_log.append({'date': date, 'regime_sharpe': np.nan,
                              'in_defensive': in_defensive, 'ret_fixed': 0.0, 'ret_switch': 0.0,
                              'holdings': list(current_holdings_fixed)})
            continue

        # 月度收益（次月开盘相对当月收盘）
        next_idx = i + 1
        if next_idx >= len(dates_to_process):
            break
        next_date = dates_to_process[next_idx]
        monthly_ret = ret_mat.loc[next_date, holdings].dropna()
        ret_fixed = float(monthly_ret.mean()) if len(monthly_ret) > 0 else 0.0

        # ── Regime 检测 ────────────────────────────────────────────────
        strategy_fixed_rets.append(ret_fixed)
        strat_s = pd.Series(strategy_fixed_rets)

        # 滚动 Sharpe（用已实现收益序列）
        reg_sharpe_val = np.nan
        if len(strat_s) >= 6:
            roll_mean = strat_s.rolling(REGIME_WINDOW, min_periods=max(6, REGIME_WINDOW//2)).mean().iloc[-1] * 12
            roll_std  = strat_s.rolling(REGIME_WINDOW, min_periods=max(6, REGIME_WINDOW//2)).std().iloc[-1] * np.sqrt(12)
            if not np.isnan(roll_std) and roll_std > 0:
                reg_sharpe_val = float(roll_mean / roll_std)

        # ── Regime Switch 逻辑 ─────────────────────────────────────────
        ret_switch = ret_fixed

        if regime_switch:
            if in_defensive:
                # 检查是否恢复：最近 REGIME_WINDOW 个月 Sharpe > 0
                if not np.isnan(reg_sharpe_val) and reg_sharpe_val > 0:
                    in_defensive = False
                    switch_dates.append(date)
                    current_holdings_switch = list(holdings)
            else:
                # 检查是否需要切换（连续 SWITCH_THR 个月 Sharpe < 0）
                if len(strategy_fixed_rets) >= SWITCH_THR:
                    recent_rets = pd.Series(strategy_fixed_rets[-SWITCH_THR:])
                    r_mean = recent_rets.mean() * 12
                    r_std  = recent_rets.std() * np.sqrt(12)
                    if r_std > 0:
                        recent_sharpe = r_mean / r_std
                        if recent_sharpe < 0:
                            in_defensive = True
                            switch_dates.append(date)

        # ── 防御状态收益 ────────────────────────────────────────────────
        if regime_switch and in_defensive:
            def_rets = ret_mat.loc[next_date, [BOND_TICKER, LOWVOL_TICKER]].dropna()
            ret_switch = float(def_rets.mean()) if len(def_rets) > 0 else 0.0

        rets_fixed.append(ret_fixed)
        rets_switch.append(ret_switch)

        current_holdings_fixed  = list(holdings)
        if not in_defensive:
            current_holdings_switch = list(holdings)

        holdings_fixed.append(list(current_holdings_fixed))
        holdings_switch.append(list(current_holdings_switch))

        regime_log.append({
            'date': date,
            'regime_sharpe': reg_sharpe_val,
            'in_defensive': in_defensive,
            'ret_fixed': ret_fixed,
            'ret_switch': ret_switch,
            'holdings': holdings
        })

    # ── 索引对齐 ────────────────────────────────────────────────────────
    valid_dates = dates_to_process[:len(rets_fixed)]
    rets_fixed_s  = pd.Series(rets_fixed,  index=valid_dates)
    rets_switch_s = pd.Series(rets_switch, index=valid_dates)

    rA = compute_stats(rets_fixed_s)
    rB = compute_stats(rets_switch_s)
    rA['turnover'] = round(calc_turnover(holdings_fixed), 3)
    rB['turnover'] = round(calc_turnover(holdings_switch), 3)
    rB['n_switches'] = len(switch_dates)

    # Regime 分析
    rlog = pd.DataFrame(regime_log)
    n_def = int((rlog['in_defensive'] == True).sum()) if len(rlog) > 0 else 0

    return {
        '固定': rA,
        '切换': rB,
        'switch_dates': [d.strftime('%Y-%m-%d') for d in switch_dates],
        'n_defensive_months': n_def,
        'regime_sharpe_series': regime_sharpe_series.dropna(),
        'holdings_fixed': holdings_fixed,
        'holdings_switch': holdings_switch,
        'rets_fixed': rets_fixed_s,
        'rets_switch': rets_switch_s,
    }


# ─── 主程序 ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 62)
    print("Rolling Retrain Regime-Switching Backtest")
    print("=" * 62)

    prices = get_prices()
    ret_mat = build_return_matrix(prices)

    print(f"\nPrice matrix: {prices.shape}")
    print(f"Return matrix: {ret_mat.shape}")
    print(f"Period: {prices.index[0].strftime('%Y-%m')} → {prices.index[-1].strftime('%Y-%m')}")

    print("\n[Mode A] Fixed strategy (roc120+vol20, monthly rebal)...")
    result = run_backtest(prices, ret_mat, regime_switch=False)
    rA = result['固定']

    print("\n[Mode B] Regime-Switching (Sharpe<0 × 4mo → defensive)...")
    resultB = run_backtest(prices, ret_mat, regime_switch=True)
    rB = resultB['切换']

    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)

    print(f"\n{'指标':<18} {'固定策略(A)':>14} {'RegimeSwitch(B)':>14} {'B-A':>10}")
    print("-" * 58)
    print(f"{'Sharpe':.<18} {rA['sharpe']:>14.2f} {rB['sharpe']:>14.2f} {rB['sharpe']-rA['sharpe']:>+10.2f}")
    print(f"{'年化收益':.<18} {rA['ann']:>13.1f}% {rB['ann']:>13.1f}% {rB['ann']-rA['ann']:>+9.1f}%")
    print(f"{'MaxDD':.<18} {rA['maxdd']:>14.1f}% {rB['maxdd']:>14.1f}% {rB['maxdd']-rA['maxdd']:>+10.1f}%")
    print(f"{'月度样本':.<18} {rA['n']:>14} {rB['n']:>14}")
    print(f"{'年换手率':.<18} {rA['turnover']:>14.2f}x {rB['turnover']:>14.2f}x")
    print(f"{'切换次数':.<18} {'N/A':>14} {rB['n_switches']:>14}")

    n_def = resultB['n_defensive_months']
    total = resultB['固定']['n'] + resultB['切换']['n'] if 'n' in resultB['切换'] else 0
    if n_def > 0:
        print(f"\n防御状态月数: {n_def}/{len(resultB['holdings_switch'])} ({100*n_def/len(resultB['holdings_switch']):.0f}%)")

    switch_dates = resultB['switch_dates']
    if switch_dates:
        print(f"切换时间点:")
        for d in switch_dates:
            print(f"  {d}")

    # Regime Sharpe 什么时候 < 0
    rss = resultB['regime_sharpe_series']
    if len(rss) > 0:
        bad = rss < 0
        print(f"\nRegime Sharpe < 0 的月数: {bad.sum()}/{len(rss)} ({100*bad.sum()/len(rss):.0f}%)")

    # 存结果
    output = {
        '固定': rA,
        '切换': rB,
        'switch_dates': switch_dates,
        'n_defensive_months': n_def,
        'regime_sharpe_negative_pct': float(100*bad.sum()/len(rss)) if len(rss) > 0 else 0
    }
    out_path = '/tmp/rolling_retrain_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")
