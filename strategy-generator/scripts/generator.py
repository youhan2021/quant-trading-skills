#!/usr/bin/env python3
"""
strategy-generator/scripts/generator.py
Multi-strategy signal generator for US stocks/ETFs.

Supported strategies:
  ma_cross, ema_cross, macd, rsi, bollinger,
  momentum, breakout, kdj, mean_reversion

Usage:
    python3 generator.py SPY                # load from cache, generate all signals
    python3 generator.py SPY --strategies ma_cross,rsi
    python3 generator.py SPY --strategies all --fast 10 --slow 30
"""

import argparse
import json
import os
import sys
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLLECTOR   = os.path.join(os.path.dirname(SKILL_DIR), "market-info-collector", "scripts", "collector.py")
sys.path.insert(0, os.path.dirname(COLLECTOR))

DATA_DIR    = os.path.join(SKILL_DIR, "data")
SIGNALS_DIR = os.path.join(DATA_DIR, "signals")
os.makedirs(SIGNALS_DIR, exist_ok=True)

# ── strategies registry ──────────────────────────────────────────────────────
ALL_STRATEGIES = [
    "ma_cross", "ema_cross", "macd", "rsi",
    "bollinger", "momentum", "breakout", "kdj", "mean_reversion",
]


def sma(series, period):
    return series.rolling(window=period).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi_calc(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs  = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd_calc(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def kdj_calc(high, low, close, period=9):
    lowest_low  = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    K = rsv.ewm(alpha=1/3, adjust=False).mean()
    D = K.ewm(alpha=1/3, adjust=False).mean()
    J = 3*K - 2*D
    return K, D, J

def bollinger_bands(series, period=20, std_dev=2):
    mid   = sma(series, period)
    std   = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low  - close.shift(1))
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


# ── signal generation ───────────────────────────────────────────────────────

def generate_ma_cross(df, fast=20, slow=50):
    """Moving average crossover."""
    close = df["close"]
    fast_sma = sma(close, fast)
    slow_sma = sma(close, slow)
    signal   = (fast_sma > slow_sma).astype(int)
    signal  = signal.where(fast_sma.notna() & slow_sma.notna(), 0)
    # 1=long, -1=short
    return signal.replace(0, -1).fillna(-1).astype(int)

def generate_ema_cross(df, fast=12, slow=26):
    f = ema(df["close"], fast)
    s = ema(df["close"], slow)
    sig = (f > s).astype(int)
    return sig.where(f.notna() & s.notna(), 0).replace(0, -1).fillna(-1).astype(int)

def generate_macd(df, fast=12, slow=26, signal=9):
    _, sig_line, hist = macd_calc(df["close"], fast, slow, signal)
    # MACD > signal → bullish
    bullish = hist > 0
    sig = bullish.astype(int)
    return sig.where(hist.notna(), 0).replace(0, -1).fillna(-1).astype(int)

def generate_rsi(df, period=14, oversold=30, overbought=70):
    r = rsi_calc(df["close"], period)
    sig = 1  # default neutral → treat as hold (-1 maps to no position)
    sig = (r < oversold).astype(int)   # oversold → buy
    sell = (r > overbought).astype(int) * -1
    sig  = sig + sell
    return sig.fillna(-1).astype(int)

def generate_bollinger(df, period=20, std_dev=2):
    upper, mid, lower = bollinger_bands(df["close"], period, std_dev)
    close = df["close"]
    buy  = (close < lower).astype(int)
    sell = (close > upper).astype(int)
    sig  = buy + sell
    return sig.where(lower.notna(), -1).fillna(-1).astype(int)

def generate_momentum(df, lookback=20, threshold=0.02):
    ret = df["close"].pct_change(lookback)
    sig = (ret > threshold).astype(int)
    return sig.where(ret.notna(), -1).fillna(-1).astype(int)

def generate_breakout(df, lookback=20):
    high = df["high"].rolling(lookback).max()
    sig  = (df["close"] > high.shift(1)).astype(int)
    return sig.where(high.notna(), -1).fillna(-1).astype(int)

def generate_kdj(df, period=9, overbought=80, oversold=20):
    K, D, J = kdj_calc(df["high"], df["low"], df["close"], period)
    buy  = (K > D) & (K < overbought)
    sell = (K < D) & (K > oversold)
    sig  = buy.astype(int) + sell.astype(int) * -1
    return sig.fillna(-1).astype(int)

def generate_mean_reversion(df, period=20, z_thresh=2.0):
    close  = df["close"]
    ma     = sma(close, period)
    std    = close.rolling(period).std()
    z      = (close - ma) / std
    sig    = (-z).clip(-1, 1)   # strong negative z → price too low → buy
    sig    = sig.where(z.abs() > z_thresh, -1)  # otherwise neutral
    return sig.fillna(-1).astype(int)


# ── master dispatch ──────────────────────────────────────────────────────────

STRATEGY_FUNCS = {
    "ma_cross":       generate_ma_cross,
    "ema_cross":      generate_ema_cross,
    "macd":           generate_macd,
    "rsi":            generate_rsi,
    "bollinger":      generate_bollinger,
    "momentum":       generate_momentum,
    "breakout":       generate_breakout,
    "kdj":            generate_kdj,
    "mean_reversion": generate_mean_reversion,
}

# Default params per strategy
STRATEGY_DEFAULTS = {
    "ma_cross":       {"fast": 20,  "slow": 50},
    "ema_cross":      {"fast": 12,  "slow": 26},
    "macd":           {"fast": 12,  "slow": 26, "signal": 9},
    "rsi":            {"period": 14, "oversold": 30, "overbought": 70},
    "bollinger":      {"period": 20, "std_dev": 2},
    "momentum":       {"lookback": 20, "threshold": 0.02},
    "breakout":       {"lookback": 20},
    "kdj":            {"period": 9, "overbought": 80, "oversold": 20},
    "mean_reversion": {"period": 20, "z_thresh": 2.0},
}

# ── parameter grids for scanning ────────────────────────────────────────────
PARAM_GRIDS = {
    "ma_cross": [
        {"fast": 5,  "slow": 20},
        {"fast": 10, "slow": 40},
        {"fast": 20, "slow": 50},
        {"fast": 20, "slow": 200},
        {"fast": 50, "slow": 200},
    ],
    "ema_cross": [
        {"fast": 12, "slow": 26},
        {"fast": 8,  "slow": 21},
        {"fast": 5,  "slow": 13},
    ],
    "macd": [
        {"fast": 12, "slow": 26, "signal": 9},
        {"fast": 8,  "slow": 21, "signal": 9},
        {"fast": 19, "slow": 39, "signal": 9},
    ],
    "rsi": [
        {"period": 7,  "oversold": 30, "overbought": 70},
        {"period": 14, "oversold": 30, "overbought": 70},
        {"period": 21, "oversold": 25, "overbought": 75},
    ],
    "bollinger": [
        {"period": 10, "std_dev": 1.5},
        {"period": 20, "std_dev": 2.0},
        {"period": 30, "std_dev": 2.5},
    ],
    "momentum": [
        {"lookback": 10, "threshold": 0.01},
        {"lookback": 20, "threshold": 0.02},
        {"lookback": 60, "threshold": 0.05},
    ],
    "breakout": [
        {"lookback": 20},
        {"lookback": 50},
        {"lookback": 100},
    ],
    "kdj": [
        {"period": 9,  "overbought": 80, "oversold": 20},
        {"period": 14, "overbought": 80, "oversold": 20},
        {"period": 9,  "overbought": 70, "oversold": 30},
    ],
    "mean_reversion": [
        {"period": 10, "z_thresh": 1.5},
        {"period": 20, "z_thresh": 2.0},
        {"period": 30, "z_thresh": 2.5},
    ],
}


def generate_signals(df: dict, strategy: str, params: dict = None) -> dict:
    """
    Generate signals from price data dict (with 'data' key from collector).
    Returns signal records list.
    """
    import pandas as pd

    params = params or {}
    defaults = STRATEGY_DEFAULTS.get(strategy, {})
    defaults.update(params)
    p = defaults

    # Build DataFrame — strip timezone to avoid mixed-timezone issues
    rows = df["data"]
    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_localize(None)
    data.set_index("date", inplace=True)
    data.sort_index(inplace=True)

    # Generate
    func = STRATEGY_FUNCS.get(strategy)
    if not func:
        return {"error": f"Unknown strategy: {strategy}"}

    try:
        sig_series = func(data, **p)
    except Exception as e:
        return {"error": str(e)}

    # Build records
    records = []
    for date, sig in sig_series.items():
        if sig == 0:
            continue  # skip hold
        row = data.loc[date]
        records.append({
            "date":    str(date.date()),
            "signal":  int(sig),   # 1=long, -1=short
            "price":   round(float(row["close"]), 2),
            "reason":  strategy,
        })

    return {
        "ticker":    df["ticker"],
        "strategy":  strategy,
        "params":    p,
        "signals":   records,
        "count":     len(records),
    }


def generate_all_strategies(df: dict) -> dict:
    """Generate all strategies for a ticker."""
    results = {}
    for strat in ALL_STRATEGIES:
        pgrid = PARAM_GRIDS.get(strat, [{}])
        strat_results = []
        for params in pgrid:
            r = generate_signals(df, strat, params)
            if "error" not in r and r["count"] > 0:
                strat_results.append(r)
        results[strat] = strat_results
    return results


def load_ticker_data(ticker: str, period: str = "2y", interval: str = "1d") -> dict:
    """Load ticker data from collector cache."""
    from collector import load_from_cache, fetch_ticker
    data = load_from_cache(ticker, period, interval)
    if not data:
        print(f"  Cache miss for {ticker}, fetching ...")
        data = fetch_ticker(ticker, period, interval)
    return data


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description="Strategy signal generator")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols")
    parser.add_argument("--strategies", default="all",
                        help="Comma-separated strategies or 'all'")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--params", help="JSON params dict")
    args = parser.parse_args()

    if args.tickers == []:
        print("Usage: generator.py <ticker> [--strategies ma_cross,rsi]")
        sys.exit(0)

    if args.strategies == "all":
        strategies = ALL_STRATEGIES
    else:
        strategies = [s.strip() for s in args.strategies.split(",")]

    for ticker in args.tickers:
        print(f"\n=== {ticker} ===")
        df = load_ticker_data(ticker, args.period, args.interval)
        if "error" in df:
            print(f"  ERROR: {df['error']}")
            continue

        for strat in strategies:
            pgrid = PARAM_GRIDS.get(strat, [{}])
            for params in pgrid:
                r = generate_signals(df, strat, params)
                if "error" in r:
                    print(f"  {strat}: ERROR {r['error']}")
                    continue
                buys  = sum(1 for s in r["signals"] if s["signal"] == 1)
                sells = sum(1 for s in r["signals"] if s["signal"] == -1)
                p_str = " ".join(f"{k}={v}" for k, v in r["params"].items())
                print(f"  {strat:20s} {p_str:40s}  buy={buys:3d}  sell={sells:3d}  total={r['count']:3d}")

                # Save
                fname = f"{ticker}_{strat}_{'_'.join(f'{k}{v}' for k,v in r['params'].items())}.json"
                fname = fname.replace(" ", "_")
                out_path = os.path.join(SIGNALS_DIR, fname)
                with open(out_path, "w") as f:
                    json.dump(r, f, indent=2)

        print(f"  → Signals saved to {SIGNALS_DIR}/")

# ── NEW STRATEGIES ──────────────────────────────────────────────────────────

def roc_calc(series, period=20):
    """Rate of Change: percentage change over period."""
    return series.pct_change(period) * 100

def generate_roc_momentum(df, period=20, threshold=0.0):
    """
    ROC Momentum Strategy (validated: #2 best SPY strategy, 19yr backtest).
    - Go LONG when ROC > threshold (price accelerating upward)
    - Exit to cash when ROC < threshold
    threshold=0: ROC > 0 = positive momentum = go long
    """
    close = df["close"]
    roc = roc_calc(close, period)
    sig = (roc > threshold).astype(int)
    return sig.where(roc.notna(), -1).fillna(-1).astype(int)

def generate_rsi_momentum(df, period=14, threshold=50):
    """
    RSI Momentum Strategy (validated: #1 best SPY strategy, 19yr backtest).
    Unlike the oversold/overbought RSI, this is a TREND-FOLLOWING strategy:
    - Go LONG when RSI crosses ABOVE threshold (default 50 = neutral line)
    - Stay in cash when RSI < threshold
    This works on SPY because RSI > 50 = sustained uptrend
    """
    r = rsi_calc(df["close"], period)
    sig = (r > threshold).astype(int)
    return sig.where(r.notna(), -1).fillna(-1).astype(int)

# ── REGIME FILTER (applied on top of any base strategy) ─────────────────────

def generate_regime_filter(df, fast=50, slow=200):
    """
    Regime filter: 50/200 MA crossover.
    Returns 1 (BULL) or -1 (BEAR) to multiply with base signal.
    Use as: final_signal = base_signal * regime
    """
    close = df["close"]
    fast_ma = sma(close, fast)
    slow_ma = sma(close, slow)
    regime = (fast_ma > slow_ma).astype(int).replace(0, -1)
    return regime.where(fast_ma.notna() & slow_ma.notna(), 1).fillna(1)

# ── Register new strategies ─────────────────────────────────────────────────

STRATEGY_FUNCS.update({
    "roc_momentum":   generate_roc_momentum,
    "rsi_momentum":   generate_rsi_momentum,
})

STRATEGY_DEFAULTS.update({
    "roc_momentum":   {"period": 20, "threshold": 0.0},
    "rsi_momentum":   {"period": 14, "threshold": 50},
})

PARAM_GRIDS.update({
    "roc_momentum": [
        {"period": 10,  "threshold": 0.0},
        {"period": 20,  "threshold": 0.0},
        {"period": 20,  "threshold": 2.0},   # require 2% ROC to enter
        {"period": 60,  "threshold": 0.0},
        {"period": 5,   "threshold": 0.0},
        {"period": 126, "threshold": 0.0},   # ~6-month ROC (like kengo benchmark)
    ],
    "rsi_momentum": [
        {"period": 14,  "threshold": 50},
        {"period": 7,   "threshold": 50},
        {"period": 21,  "threshold": 50},
        {"period": 10,  "threshold": 50},
    ],
})

ALL_STRATEGIES += ["roc_momentum", "rsi_momentum"]

# ── COMPOSITE STRATEGIES (base + regime filter) ─────────────────────────────

def generate_ma_cross_regime(df, fast=20, slow=50, rf_fast=50, rf_slow=200):
    """
    MA Cross with 50/200 Regime Filter.
    Only go long when MA golden cross AND market is in uptrend.
    """
    base = generate_ma_cross(df, fast, slow)
    regime = generate_regime_filter(df, rf_fast, rf_slow)
    sig = base * regime
    # Neutral: when regime bearish, stay out even if base says long
    return sig.fillna(-1).astype(int)

def generate_roc_momentum_regime(df, period=20, threshold=0.0, rf_fast=50, rf_slow=200):
    """
    ROC Momentum with Regime Filter.
    Only go long when ROC momentum signal AND 50/200 MA confirms uptrend.
    """
    base = generate_roc_momentum(df, period, threshold)
    regime = generate_regime_filter(df, rf_fast, rf_slow)
    sig = base * regime
    return sig.fillna(-1).astype(int)

def generate_rsi_momentum_regime(df, period=14, threshold=50, rf_fast=50, rf_slow=200):
    """
    RSI Momentum with Regime Filter.
    Only go long when RSI momentum AND 50/200 MA confirms uptrend.
    """
    base = generate_rsi_momentum(df, period, threshold)
    regime = generate_regime_filter(df, rf_fast, rf_slow)
    sig = base * regime
    return sig.fillna(-1).astype(int)

# ── Register composite strategies ────────────────────────────────────────────

STRATEGY_FUNCS.update({
    "ma_cross_regime":       generate_ma_cross_regime,
    "roc_momentum_regime":   generate_roc_momentum_regime,
    "rsi_momentum_regime":   generate_rsi_momentum_regime,
})

STRATEGY_DEFAULTS.update({
    "ma_cross_regime":       {"fast": 20,  "slow": 50},
    "roc_momentum_regime":   {"period": 20, "threshold": 0.0},
    "rsi_momentum_regime":   {"period": 14, "threshold": 50},
})

PARAM_GRIDS.update({
    "ma_cross_regime": [
        {"fast": 20,  "slow": 50},
        {"fast": 10,  "slow": 40},
        {"fast": 5,   "slow": 20},
    ],
    "roc_momentum_regime": [
        {"period": 20,  "threshold": 0.0},
        {"period": 20,  "threshold": 2.0},
        {"period": 60,  "threshold": 0.0},
        {"period": 126, "threshold": 0.0},
    ],
    "rsi_momentum_regime": [
        {"period": 14,  "threshold": 50},
        {"period": 7,   "threshold": 50},
        {"period": 21,  "threshold": 50},
    ],
})

ALL_STRATEGIES += ["ma_cross_regime", "roc_momentum_regime", "rsi_momentum_regime"]
