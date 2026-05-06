#!/usr/bin/env python3
"""
backtest-engine/scripts/backtester.py
Full backtest engine with parameter scanning.

Usage:
    python3 backtester.py SPY ma_cross --fast 20 --slow 50
    python3 backtester.py SPY rsi --period 14 --oversold 30 --overbought 70
    python3 backtester.py --scan SPY        # scan all strategies+params
    python3 backtester.py --scan-all         # scan all strategies on all tickers
"""

import argparse
import json
import math
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from pathlib import Path

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLLECTOR   = os.path.join(os.path.dirname(SKILL_DIR), "market-info-collector", "scripts", "collector.py")
sys.path.insert(0, os.path.dirname(COLLECTOR))

GENERATOR   = os.path.join(os.path.dirname(SKILL_DIR), "strategy-generator", "scripts", "generator.py")
sys.path.insert(0, os.path.dirname(GENERATOR))

RESULTS_DIR = os.path.join(SKILL_DIR, "data", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import pandas as pd
import numpy as np


# ── performance metrics ───────────────────────────────────────────────────────

def calc_max_drawdown(equity_curve: list[dict]) -> float:
    """Returns max drawdown as a negative fraction."""
    equities = [e["equity"] for e in equity_curve]
    peak = equities[0]
    worst = 0.0
    for e in equities:
        if e > peak:
            peak = e
        dd = (e - peak) / peak
        if dd < worst:
            worst = dd
    return round(worst, 6)


def calc_sharpe(returns: list[float], risk_free: float = 0.04) -> float:
    if len(returns) < 2:
        return 0.0
    rets = np.array(returns, dtype=float)
    std = np.std(rets, ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    excess = rets - risk_free / 252
    return round(float(np.mean(excess) / std * math.sqrt(252)), 4)


def calc_sortino(returns: list[float], risk_free: float = 0.04) -> float:
    if len(returns) < 2:
        return 0.0
    rets = np.array(returns, dtype=float)
    std = np.std(rets, ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    downside = rets[rets < 0]
    if len(downside) <= 1:
        return 0.0
    excess = rets - risk_free / 252
    dstd = np.std(downside, ddof=1)
    if dstd == 0 or np.isnan(dstd):
        return 0.0
    return round(float(np.mean(excess) / dstd * math.sqrt(252)), 4)


def calc_calmar(ann_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return round(ann_return / abs(max_dd), 4)


def win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return round(wins / len(trades), 4)


def profit_factor(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss   = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    if gross_loss == 0:
        return 99.99 if gross_profit > 0 else 0.0
    return round(gross_profit / gross_loss, 4)


# ── core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    ticker: str,
    strategy: str,
    price_data: dict,
    signal_records: list[dict],
    params: dict,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    risk_free: float = 0.04,
) -> dict:
    """
    Run a backtest given price data dict and signal records.
    Returns a full backtest result dict.
    """
    rows = price_data["data"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # Build signal lookup
    sig_map = {}
    for s in signal_records:
        sig_map[s["date"]] = s["signal"]  # 1=long, 0/-1=cash (no short)

    # Simulate
    trades = []
    equity_curve = [{"date": str(df.index[0].date()), "equity": initial_capital}]
    position = 0
    entry_price = 0.0
    entry_date = None
    capital = initial_capital

    trade_id = 0
    for i, (date, row) in enumerate(df.iterrows()):
        date_str = str(date.date())
        price = row["close"]

        # Track equity every day even when flat
        equity_curve.append({"date": date_str, "equity": capital})

        if date_str not in sig_map:
            continue

        signal = sig_map[date_str]

        # FLAT → go LONG when signal == 1
        if position == 0 and signal == 1:
            shares = int(capital * 0.95 / price)
            if shares < 1:
                continue
            cost = shares * price * (1 + commission)
            position   = shares
            entry_price = price
            entry_date  = date
            equity_curve[-1] = {"date": date_str, "equity": cost}

        # LONG → exit when signal != 1 (signal == -1 or signal == 0)
        elif position > 0 and signal != 1:
            proceeds = position * price * (1 - commission)
            pnl = (price - entry_price) / entry_price
            trade_id += 1
            trades.append({
                "trade_id":    trade_id,
                "entry_date":  str(entry_date.date()),
                "exit_date":   date_str,
                "side":        "long",
                "entry_price": round(entry_price, 4),
                "exit_price":  round(price, 4),
                "pnl":         round(pnl, 6),
                "holding_days": (date - entry_date).days,
            })
            capital = proceeds
            position = 0
            equity_curve[-1] = {"date": date_str, "equity": proceeds}

    # Force close at end
    if position > 0:
        last_price = df.iloc[-1]["close"]
        proceeds = position * last_price * (1 - commission)
        pnl = (last_price - entry_price) / entry_price
        trade_id += 1
        trades.append({
            "trade_id":    trade_id,
            "entry_date":  str(entry_date.date()),
            "exit_date":   str(df.index[-1].date()),
            "side":        "long",
            "entry_price": round(entry_price, 4),
            "exit_price":  round(last_price, 4),
            "pnl":         round(pnl, 6),
            "holding_days": (df.index[-1] - entry_date).days,
        })
        capital = proceeds

    # ── metrics ──────────────────────────────────────────────────────────────
    total_return = (capital - initial_capital) / initial_capital

    if not trades:
        # buy & hold fallback
        first_price = df.iloc[0]["close"]
        last_price  = df.iloc[-1]["close"]
        total_return = (last_price - first_price) / first_price
        n_days = (df.index[-1] - df.index[0]).days
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1 if total_return > -1 else -1.0
        sharpe  = 0.0
        sortino = 0.0
        calmar  = 0.0
        max_dd  = total_return
    else:
        pnls   = [t["pnl"] for t in trades]
        ann_return = (1 + total_return) ** (252 / max(len(df), 1)) - 1 if total_return > -1 else -1.0
        sharpe  = calc_sharpe(pnls, risk_free)
        sortino = calc_sortino(pnls, risk_free)
        max_dd  = calc_max_drawdown(equity_curve)
        calmar  = calc_calmar(ann_return, max_dd)

    win_rate_v  = win_rate(trades)
    pf          = profit_factor(trades)

    # Per-trade returns for volatility
    rets_series = pd.Series([t["pnl"] for t in trades])
    volatility  = float(rets_series.std() * math.sqrt(252)) if len(rets_series) > 1 else 0.0

    # Holding stats
    avg_holding = float(np.mean([t["holding_days"] for t in trades])) if trades else 0.0

    result = {
        "backtest_id": f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "ticker":      ticker,
        "strategy":    strategy,
        "params":      params,
        "period":      f"{price_data.get('start','')[:10]} to {price_data.get('end','')[:10]}",
        "initial_capital": initial_capital,
        "final_capital":   round(capital, 2),
        "performance": {
            "total_return":       round(total_return, 6),
            "annualized_return":  round(ann_return, 6),
            "sharpe_ratio":       sharpe,
            "sortino_ratio":      sortino,
            "max_drawdown":       round(max_dd, 6),
            "calmar_ratio":       calmar,
            "win_rate":           win_rate_v,
            "profit_factor":      pf,
            "volatility":         round(volatility, 6),
        },
        "trade_stats": {
            "total_trades":        len(trades),
            "avg_holding_days":    round(avg_holding, 1),
            "avg_profit_per_trade": round(float(np.mean([t["pnl"] for t in trades])), 6) if trades else 0.0,
        },
        "trades": trades,
        "equity_curve": equity_curve,
        "generated_at": datetime.now().isoformat(),
    }
    return result


def save_result(result: dict) -> str:
    tag = f"{result['ticker']}_{result['strategy']}"
    p_str = "_".join(f"{k}{v}" for k, v in result["params"].items())
    fname = f"{tag}_{p_str}.json"
    fname = fname.replace(" ", "_").replace(".", "p")
    out = os.path.join(RESULTS_DIR, fname)
    with open(out, "w") as f:
        json.dump(result, f)
    return out


# ── scanner ──────────────────────────────────────────────────────────────────

def scan_ticker(ticker: str, period: str = "2y", interval: str = "1d") -> list[dict]:
    """Run all strategies × all param combos for one ticker."""
    from collector import load_from_cache, fetch_ticker
    from generator import (
        load_ticker_data, generate_signals,
        ALL_STRATEGIES, PARAM_GRIDS,
    )

    print(f"\n{'='*60}")
    print(f"SCANNING: {ticker}")
    print(f"{'='*60}")

    price_data = load_from_cache(ticker, period, interval)
    if price_data is None:
        print(f"  [cache miss — fetching {ticker}]")
        price_data = fetch_ticker(ticker, period, interval)
    if price_data is None or "error" in price_data or price_data.get("count", 0) == 0:
        msg = price_data.get("error", "no data") if isinstance(price_data, dict) else "no data"
        print(f"  Cannot load {ticker}: {msg}")
        return []

    all_results = []
    strategy_count = 0

    for strat in ALL_STRATEGIES:
        pgrid = PARAM_GRIDS.get(strat, [{}])
        for params in pgrid:
            signals_r = generate_signals(price_data, strat, params)
            if "error" in signals_r:
                continue
            if signals_r["count"] == 0:
                continue

            bt = run_backtest(
                ticker, strat, price_data,
                signals_r["signals"], params,
            )
            bt["param_tag"] = "_".join(f"{k}{v}" for k, v in params.items())
            out_path = save_result(bt)

            perf = bt["performance"]
            # Filter: Sharpe > 0.5, max_drawdown > -30%
            sharpe  = perf["sharpe_ratio"]
            max_dd  = perf["max_drawdown"]
            ann_ret = perf["annualized_return"]

            flag = ""
            if sharpe >= 1.5 and max_dd >= -0.15 and ann_ret > 0:
                flag = "⭐ TOP"
            elif sharpe >= 1.0 and max_dd >= -0.20 and ann_ret > 0:
                flag = "✓ OK"
            elif sharpe < 0 or max_dd < -0.30:
                flag = "✗ POOR"

            print(f"  [{flag}] {strat:20s} sharpe={sharpe:+.2f} "
                  f"ann_ret={ann_ret:+.1%} max_dd={max_dd:+.1%} "
                  f"trades={bt['trade_stats']['total_trades']:3d}  "
                  f"→ {out_path.split('/')[-1]}")

            all_results.append(bt)
            strategy_count += 1

    print(f"  → {strategy_count} strategy variants tested")
    return all_results


def top_results(results: list[dict], min_sharpe: float = 0.5,
                min_ann_ret: float = 0.0, max_dd: float = -1.0,
                sort_by: str = "sharpe") -> list[dict]:
    """Filter and sort results.
    Default: min_sharpe=0.5, min_ann_ret=0.0 (must be positive),
    max_dd=-1.0 (at least -100% drawdown, i.e. everything).
    Use -0.20 for max_dd to require drawdown > -20%."""
    filtered = [
        r for r in results
        if r["performance"]["sharpe_ratio"] >= min_sharpe
        and r["performance"]["annualized_return"] >= min_ann_ret
        and r["performance"]["max_drawdown"] >= max_dd
    ]
    return sorted(filtered, key=lambda r: r["performance"].get(sort_by, 0), reverse=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest engine")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol")
    parser.add_argument("strategy", nargs="?", help="Strategy name")
    parser.add_argument("--scan", action="store_true", help="Scan all strategies for ticker")
    parser.add_argument("--scan-all", action="store_true",
                        help="Scan all strategies on all cached tickers")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--min-sharpe", type=float, default=1.0)
    parser.add_argument("--min-ann-ret", type=float, default=0.0,
                        help="Minimum annualized return (default 0.0 = must be positive)")
    parser.add_argument("--max-dd", type=float, default=-0.25,
                        help="Maximum drawdown (most negative, default -0.25)")
    parser.add_argument("--sort", default="sharpe",
                        choices=["sharpe","ann_ret","max_dd","win_rate","profit_factor"])
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    if args.scan_all:
        from collector import list_cached_tickers
        cached = list_cached_tickers()
        print(f"\nFound {len(cached)} cached tickers: {', '.join(cached)}")
        all_results = []
        for t in cached:
            all_results.extend(scan_ticker(t, args.period, args.interval))

    elif args.scan and args.ticker:
        all_results = scan_ticker(args.ticker, args.period, args.interval)

    elif args.ticker and args.strategy:
        from generator import load_ticker_data, generate_signals, PARAM_GRIDS
        from collector import load_from_cache, fetch_ticker

        df = load_ticker_data(args.ticker, args.period, args.interval)
        if "error" in df:
            print(f"ERROR: {df['error']}")
            sys.exit(1)

        # Parse extra params
        params = {}
        # Allow --fast, --slow, etc. to be passed
        for kw in ["fast","slow","period","signal","oversold","overbought",
                   "std_dev","lookback","threshold","z_thresh","overbought_kdj","oversold_kdj"]:
            if hasattr(args, kw) and getattr(args, kw) is not None:
                try:
                    params[kw] = float(getattr(args, kw)) if getattr(args, kw) != "True" else True
                except:
                    pass

        sig_r = generate_signals(df, args.strategy, params)
        if "error" in sig_r:
            print(f"ERROR: {sig_r['error']}")
            sys.exit(1)

        bt = run_backtest(args.ticker, args.strategy, df, sig_r["signals"], params)
        out = save_result(bt)
        perf = bt["performance"]
        print(f"\nBacktest result for {args.ticker} / {args.strategy}")
        print(f"  Total return:      {perf['total_return']:+.2%}")
        print(f"  Annualized return: {perf['annualized_return']:+.2%}")
        print(f"  Sharpe ratio:      {perf['sharpe_ratio']:+.2f}")
        print(f"  Sortino ratio:      {perf['sortino_ratio']:+.2f}")
        print(f"  Max drawdown:       {perf['max_drawdown']:+.2%}")
        print(f"  Calmar ratio:      {perf['calmar_ratio']:+.2f}")
        print(f"  Win rate:          {perf['win_rate']:.2%}")
        print(f"  Profit factor:     {perf['profit_factor']:.2f}")
        print(f"  Total trades:      {bt['trade_stats']['total_trades']}")
        print(f"\nSaved to: {out}")
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(0)

    # Summarise top results
    top = top_results(
        all_results,
        min_sharpe=args.min_sharpe,
        min_ann_ret=args.min_ann_ret,
        max_dd=args.max_dd,
        sort_by=args.sort,
    )

    print(f"\n{'='*70}")
    print(f"TOP {min(args.top, len(top))} RESULTS (min_sharpe={args.min_sharpe}, sort={args.sort})")
    print(f"{'='*70}")
    print(f"{'Ticker':<8} {'Strategy':<18} {'Sharpe':>7} {'Ann Ret':>9} {'MaxDD':>8} "
          f"{'WinRate':>8} {'PF':>7} {'Trades':>6}")
    print("-" * 70)
    for r in top[:args.top]:
        p = r["performance"]
        ts = r["trade_stats"]
        params_str = " ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"{r['ticker']:<8} {r['strategy']:<18} {p['sharpe_ratio']:>+7.2f} "
              f"{p['annualized_return']:>+9.1%} {p['max_drawdown']:>+8.1%} "
              f"{p['win_rate']:>8.1%} {p['profit_factor']:>7.2f} {ts['total_trades']:>6}")

    # Save summary
    summary = {
        "scanned_at": datetime.now().isoformat(),
        "total_results": len(all_results),
        "passed_filter": len(top),
        "top": top[:args.top],
    }
    sum_file = os.path.join(RESULTS_DIR, f"scan_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(sum_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {sum_file}")
