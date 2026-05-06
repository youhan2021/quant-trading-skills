"""
Weekly Regime-Switching Backtest
=================================
对比两种策略：
  A. 固定策略（roc120+vol20 rank-zscore top5，每周rebal）
  B. Regime-Switching（连续4周 Sharpe<0 → 切换IEF/SPLV，每周rebal）

数据：yfinance 日线 → 周线（每周最后一个交易日）
因子：roc120 + vol20（用日线算，rank-zscore，等权组合）
Regime检测：滚动12周 Sharpe
切换目标：50% IEF + 50% SPLV（国债+低波）
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
import os

# ─── Config ────────────────────────────────────────────────────────────────
START   = "2011-01-01"
END     = "2026-01-31"
TOP_N   = 5
REGIME_N    = 12   # 滚动窗口（周）—— Regime 检测用
SWITCH_THR  = 8    # 连续N周 Sharpe<0 → 切换（更稳定）
RESTORE_THR = 8    # 连续N周 Sharpe>0 → 恢复（更稳定）
TC          = 0.002  # 20bps 单程交易成本

BOND_TICKER = "IEF"
LOWVOL_TICKER = "SPLV"

# 13只干净股票 universe（无杠杆ETF/VIX/债券，纯真实股票）
TICKERS = sorted([
    "SPY","QQQ","GLD","TLT","EFA","AMZN","LMT","V","NVDA",
    "WMT","GOOGL","UUP","JPM","IEF","SPLV"   # IEF/SPLV 用于防御态仓位
])

# ─── 数据获取 ──────────────────────────────────────────────────────────────

def get_prices():
    cache = "/tmp/rolling_retrain_weekly_prices.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"  Loaded from cache: {df.shape}")
        return df

    print(f"Fetching {len(TICKERS)} tickers from yfinance (weekly backtest)...")
    data = yf.download(TICKERS, start=START, end=END, auto_adjust=False, progress=False)
    closes = data['Close'].dropna(how='all', axis=1)
    closes.index = closes.index.tz_localize(None) if closes.index.tz else closes.index

    # 周线：每周最后一个交易日
    weekly = closes.resample('W').last()
    weekly.to_csv(cache)
    print(f"  Saved: {weekly.shape}")
    return weekly


# ─── 因子计算（日线窗口）─────────────────────────────────────────────────

def rolling_roc(series: pd.Series, window: int) -> pd.Series:
    return series / series.shift(window) - 1


def rolling_vol(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window).std() * np.sqrt(252)


def rank_zscore(s: pd.Series) -> pd.Series:
    """rank percentile → zscore"""
    rank = s.rank(pct=True, na_option='bottom')
    return (rank - rank.mean()) / rank.std()


def compute_factor_scores_weekly(price_weekly: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """
    用 cutoff 之前的日线数据计算因子。
    price_weekly 索引是周，cutoff 是对应周最后一个交易日。
    在 cutoff 之前的全部日线数据上算 roc120 + vol20。
    """
    # 找 cutoff 对应的日线数据
    if cutoff.tz:
        cutoff = cutoff.tz_localize(None)
    price_daily = get_daily_prices(cutoff)
    if price_daily.shape[0] < 130:
        return pd.Series(dtype=float)

    scores = {}
    for ticker in price_daily.columns:
        s = price_daily[ticker].dropna()
        if len(s) < 130:
            continue
        try:
            roc = rolling_roc(s, 120)
            vol = rolling_vol(s, 20)
            if not (np.isnan(roc.iloc[-1]) or np.isnan(vol.iloc[-1])):
                # 低波动 → 排名高（ascending=False 说明 vol 低分高排）
                scores[ticker] = float(roc.iloc[-1])
        except:
            continue

    if not scores:
        return pd.Series(dtype=float)

    s = pd.Series(scores)
    roc_rank = rank_zscore(s)

    # vol 用日线全部历史
    vol_scores = {}
    for ticker in price_daily.columns:
        s = price_daily[ticker].dropna()
        if len(s) < 20:
            continue
        try:
            vol = rolling_vol(s, 20)
            if not np.isnan(vol.iloc[-1]):
                vol_scores[ticker] = float(vol.iloc[-1])
        except:
            continue

    if not vol_scores:
        return pd.Series(dtype=float)

    vol_s = pd.Series(vol_scores)
    # 低波动 → 高排名（低波动=安全资产）
    vol_rank = rank_zscore(vol_s)

    # 组合：roc高 + vol低
    combined = roc_rank + vol_rank
    return rank_zscore(combined)


_daily_data = None  # 全局日线数据，只加载一次

def get_daily_prices(up_to: pd.Timestamp) -> pd.DataFrame:
    """获取 up_to 之前的日线数据（全部缓存在内存，一次下载）"""
    global _daily_data
    if _daily_data is None:
        print(f"  Downloading daily prices from yfinance...")
        data = yf.download(TICKERS, start=START, end=END, auto_adjust=False, progress=False)
        closes = data['Close'].dropna(how='all', axis=1)
        closes.index = closes.index.tz_localize(None) if closes.index.tz else closes.index
        _daily_data = closes
        print(f"  Daily data loaded: {_daily_data.shape}")
    return _daily_data.loc[:up_to]


# ─── 预计算每周因子打分 ───────────────────────────────────────────────────

def precompute_weekly_factors(price_weekly: pd.DataFrame) -> dict:
    """
    对每个周末计算 roc120+vol20 组合因子打分。
    返回 dict: {week_end_date: [ticker_list]}
    """
    print(f"  Precomputing weekly factor scores for {len(price_weekly)} weeks...")
    cache = "/tmp/weekly_top5_cache.json"
    if os.path.exists(cache):
        with open(cache) as f:
            raw = json.load(f)
        return {pd.Timestamp(k): v for k, v in raw.items()}

    # 获取日线数据（只下载一次）
    daily = get_daily_prices(price_weekly.index[-1])

    top5_cache = {}
    for i, week_date in enumerate(price_weekly.index):
        if i < 26:  # 跳过前26周（~6个月），保证 roc120 有足够数据
            continue

        scores = compute_factor_scores_weekly(price_weekly, week_date)
        if len(scores) >= TOP_N:
            top = scores.nlargest(TOP_N)
            # 过滤：必须有下周收益
            next_i = i + 1
            if next_i >= len(price_weekly):
                continue
            next_week = price_weekly.index[next_i]
            valid = []
            for t in top.index:
                if t in price_weekly.columns:
                    try:
                        p = price_weekly.loc[:next_week, t].dropna()
                        if len(p) > 0 and p.iloc[-1] > 0:
                            valid.append(t)
                    except:
                        continue
            if len(valid) >= TOP_N:
                top5_cache[week_date] = valid

        if (i + 1) % 52 == 0:
            print(f"    Week {i+1}/{len(price_weekly)} done...")

    # 保存
    raw = {k.strftime('%Y-%m-%d'): v for k, v in top5_cache.items()}
    with open(cache, 'w') as f:
        json.dump(raw, f)
    print(f"  Cached {len(top5_cache)} weeks of factor scores → {cache}")

    return top5_cache


# ─── 主回测 ───────────────────────────────────────────────────────────────

def build_weekly_rets(price_weekly: pd.DataFrame) -> pd.DataFrame:
    """周收益率矩阵：(C[t+1] - C[t]) / C[t]"""
    rets = price_weekly.pct_change().iloc[1:]  # 第一行NaN
    return rets


def rolling_sharpe_12w(rets: pd.Series, window: int = REGIME_N) -> float:
    if len(rets) < 6:
        return np.nan
    roll_mean = rets.rolling(window, min_periods=4).mean().iloc[-1] * np.sqrt(52)
    roll_std  = rets.rolling(window, min_periods=4).std().iloc[-1] * np.sqrt(52)
    if roll_std == 0 or np.isnan(roll_std):
        return np.nan
    return float(roll_mean / roll_std)


def compute_stats(rets: pd.Series):
    if len(rets) == 0:
        return {'sharpe': 0, 'ann': 0, 'maxdd': 0, 'n': 0}
    ann    = rets.mean() * 52
    std    = rets.std() * np.sqrt(52)
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
        old = set(holdings_history[i-1] or [])
        new = set(holdings_history[i] or [])
        union = old | new
        if len(union) > 0:
            total += len(old ^ new) / len(union)
    return total / (len(holdings_history) - 1)


def run_backtest(price_weekly: pd.DataFrame, ret_mat: pd.DataFrame,
                 top5_cache: dict, regime_switch: bool = False):
    """
    Weekly rebalance backtest。
    每个周末算因子 → 下周持有 → 下周末算收益。
    """

    # 有效起始点：跳过前26周（~6个月，保证roc120有数据）
    weekly_idx = price_weekly.index.tolist()
    first_valid_i = 26
    dates_to_process = weekly_idx[first_valid_i:]

    print(f"  Backtest: {dates_to_process[0].strftime('%Y-%m-%d')} → {dates_to_process[-1].strftime('%Y-%m-%d')}, N={len(dates_to_process)}")

    # 预计算每周 IEF/SPLV 收益率
    bench_rets = ret_mat[[BOND_TICKER, LOWVOL_TICKER]].mean(axis=1)

    # ── 主循环 ────────────────────────────────────────────────────────
    rets_fixed   = []
    rets_switch  = []
    holdings_fixed  = []
    holdings_switch = []

    strat_rets_fixed  = []  # 用于 rolling Sharpe
    in_defensive = False
    switch_dates  = []
    regime_log    = []

    cur_holdings_fixed  = []
    cur_holdings_switch = []

    for i, date in enumerate(dates_to_process[:-1]):   # 最后一周没有下周收益
        holdings = top5_cache.get(date, [])
        if len(holdings) < TOP_N:
            rets_fixed.append(0.0)
            rets_switch.append(0.0)
            holdings_fixed.append(list(cur_holdings_fixed))
            holdings_switch.append(list(cur_holdings_switch))
            regime_log.append({'date': date, 'regime_sharpe': np.nan,
                              'in_defensive': in_defensive,
                              'ret_fixed': 0.0, 'ret_switch': 0.0,
                              'holdings': list(cur_holdings_fixed)})
            continue

        # 下周收益率（次周开盘 / 当周收盘）
        next_date = dates_to_process[i + 1]
        wret = ret_mat.loc[next_date, holdings].dropna()
        ret_fixed = float(wret.mean()) if len(wret) > 0 else 0.0

        # ── Regime 检测 ────────────────────────────────────────────────
        strat_rets_fixed.append(ret_fixed)
        strat_s = pd.Series(strat_rets_fixed)
        reg_sharpe = rolling_sharpe_12w(strat_s, REGIME_N)

        # ── Regime Switch 逻辑 ───────────────────────────────────────
        if regime_switch:
            if in_defensive:
                # 恢复条件：连续 RESTORE_THR 周 Sharpe > 0
                if len(strat_rets_fixed) >= RESTORE_THR:
                    recent = pd.Series(strat_rets_fixed[-RESTORE_THR:])
                    r_m = recent.mean() * 52
                    r_s = recent.std() * np.sqrt(52)
                    if r_s > 0:
                        s = r_m / r_s
                        if s > 0:
                            in_defensive = False
                            switch_dates.append(('restore', date))
                            cur_holdings_switch = list(holdings)
            else:
                # 触发条件：连续 SWITCH_THR 周 Sharpe < 0
                if len(strat_rets_fixed) >= SWITCH_THR:
                    recent = pd.Series(strat_rets_fixed[-SWITCH_THR:])
                    r_m = recent.mean() * 52
                    r_s = recent.std() * np.sqrt(52)
                    if r_s > 0:
                        s = r_m / r_s
                        if s < 0:
                            in_defensive = True
                            switch_dates.append(('switch', date))

        ret_switch = ret_fixed

        # ── 防御状态收益 ────────────────────────────────────────────
        if regime_switch and in_defensive:
            def_ret = float(bench_rets.loc[next_date]) if next_date in bench_rets.index else 0.0
            ret_switch = def_ret
        else:
            ret_switch = ret_fixed

        rets_fixed.append(ret_fixed)
        rets_switch.append(ret_switch)

        cur_holdings_fixed  = list(holdings)
        if not in_defensive:
            cur_holdings_switch = list(holdings)

        holdings_fixed.append(list(cur_holdings_fixed))
        holdings_switch.append(list(cur_holdings_switch))

        regime_log.append({
            'date': date,
            'regime_sharpe': reg_sharpe if not np.isnan(reg_sharpe) else 0.0,
            'in_defensive': in_defensive,
            'ret_fixed': ret_fixed,
            'ret_switch': ret_switch,
            'holdings': holdings
        })

    # ── 统计 ──────────────────────────────────────────────────────────
    rets_fixed_s  = pd.Series(rets_fixed,  index=dates_to_process[:len(rets_fixed)])
    rets_switch_s = pd.Series(rets_switch, index=dates_to_process[:len(rets_switch)])

    rA = compute_stats(rets_fixed_s)
    rB = compute_stats(rets_switch_s)
    rA['turnover']  = round(calc_turnover(holdings_fixed), 3)
    rB['turnover']  = round(calc_turnover(holdings_switch), 3)
    rB['n_switches'] = len(switch_dates)

    rlog = pd.DataFrame(regime_log)
    n_def = int((rlog['in_defensive'] == True).sum()) if len(rlog) > 0 else 0

    return {
        '固定': rA,
        '切换': rB,
        'switch_dates': [(t, d.strftime('%Y-%m-%d')) for t, d in switch_dates],
        'n_defensive_weeks': n_def,
        'total_weeks': len(rets_fixed),
        'regime_log': rlog,
        'rets_fixed': rets_fixed_s,
        'rets_switch': rets_switch_s,
    }


# ─── 主程序 ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 62)
    print("Weekly Regime-Switching Backtest")
    print("=" * 62)

    prices = get_prices()
    ret_mat = build_weekly_rets(prices)

    print(f"\nWeekly price matrix: {prices.shape}")
    print(f"Period: {prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')}")

    # 预计算因子
    top5_cache = precompute_weekly_factors(prices)

    print(f"\n[Mode A] Fixed strategy (weekly rebal, roc120+vol20 top5)...")
    resultA = run_backtest(prices, ret_mat, top5_cache, regime_switch=False)
    rA = resultA['固定']

    print(f"\n[Mode B] Regime-Switching (4w Sharpe<0 → IEF/SPLV)...")
    resultB = run_backtest(prices, ret_mat, top5_cache, regime_switch=True)
    rB = resultB['切换']

    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)

    print(f"\n{'指标':<16} {'固定(A)':>12} {'RegimeSw(B)':>12} {'B-A':>10}")
    print("-" * 52)
    print(f"{'Sharpe':.<16} {rA['sharpe']:>12.2f} {rB['sharpe']:>12.2f} {rB['sharpe']-rA['sharpe']:>+10.2f}")
    print(f"{'年化收益':.<16} {rA['ann']:>11.1f}% {rB['ann']:>11.1f}% {rB['ann']-rA['ann']:>+9.1f}%")
    print(f"{'MaxDD':.<16} {rA['maxdd']:>12.1f}% {rB['maxdd']:>12.1f}% {rB['maxdd']-rA['maxdd']:>+10.1f}%")
    print(f"{'周样本':.<16} {rA['n']:>12} {rB['n']:>12}")
    print(f"{'年换手率':.<16} {rA['turnover']:>12.2f}x {rB['turnover']:>12.2f}x")
    print(f"{'切换次数':.<16} {'N/A':>12} {rB['n_switches']:>12}")

    n_def = resultB['n_defensive_weeks']
    total = resultB['total_weeks']
    pct = 100 * n_def / total if total > 0 else 0
    print(f"\n防御状态: {n_def}/{total} 周 ({pct:.0f}%)")

    switch_dates = resultB['switch_dates']
    # 存结果（switch_dates 可能是tuple或string）
    switch_dates_str = []
    for item in switch_dates:
        if isinstance(item, tuple):
            switch_dates_str.append((item[0], item[1].strftime('%Y-%m-%d') if hasattr(item[1], 'strftime') else str(item[1])))
        else:
            switch_dates_str.append(str(item))

    # SPY Buy&Hold baseline
    spy_rets = ret_mat['SPY'].dropna()
    spy_stats = compute_stats(spy_rets)
    print(f"\nSPY B&H baseline: Sharpe={spy_stats['sharpe']:.2f}, Ann={spy_stats['ann']:.1f}%, MaxDD={spy_stats['maxdd']:.1f}%")

    # 存结果
    output = {
        '固定': rA,
        '切换': rB,
        'switch_dates': switch_dates_str,
        'n_defensive_weeks': n_def,
        'total_weeks': total,
        'spy': spy_stats
    }
    out_path = '/tmp/rolling_retrain_weekly_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")
