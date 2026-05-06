#!/usr/bin/env python3
"""
Weekly Rebalancer — simulates fixed-schedule (weekly) portfolio rebalancing.

Rules:
  1. Every Friday close: read closing prices, compute target share counts
     for each portfolio position based on equal weight allocation.
     Target shares = floor(portfolio_value_at_close / N_positions / close_price)
  2. Next Monday open: execute at open price.
     Actual cost = open_price × target_shares
  3. Track: cash used, position value, portfolio NAV, drawdown.
  4. Rebalance only if deviation from target weight exceeds rebalance_threshold
     (default 20% — i.e., only rebalance when a position drifts >20% from target weight).

Output: trade log + performance summary printed to stdout.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(SKILL_DIR, "market-info-collector/data/cache")
OUTPUT_DIR = os.path.join(SKILL_DIR, "output/portfolio")

# ── price cache helpers ────────────────────────────────────────────────────────

def _cache_key(ticker: str, period: str, interval: str) -> str:
    return f"{ticker}_{period}_{interval}.json"

def load_price_cache(ticker: str, period: str = "2y", interval: str = "1d") -> list[dict]:
    """Load cached yfinance OHLCV data for a ticker."""
    fpath = os.path.join(CACHE_DIR, _cache_key(ticker, period, interval))
    if not os.path.exists(fpath):
        return []
    with open(fpath) as f:
        raw = json.load(f)
    # yfinance saves as {ticker: {price_data: [...]}}
    if isinstance(raw, dict) and "price_data" in raw:
        rows = raw["price_data"]
    elif isinstance(raw, dict):
        # Fallback: try to find the list inside
        for v in raw.values():
            if isinstance(v, list):
                rows = v
                break
        else:
            rows = []
    else:
        rows = []
    # Normalise to list of dicts with date/open/close
    out = []
    for row in rows:
        if isinstance(row, dict):
            out.append({
                "date":   str(row.get("date", row.get("Datetime", "")))[:10],
                "open":   float(row.get("open", 0)),
                "close":  float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
            })
    return out

def is_friday(date_str: str) -> bool:
    from datetime import datetime
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return d.weekday() == 4
    except:
        return False

def is_monday(date_str: str) -> bool:
    from datetime import datetime
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return d.weekday() == 0
    except:
        return False

# ── core simulation ───────────────────────────────────────────────────────────

def simulate_weekly_rebalance(portfolio: list,
                                tickers: list,
                                initial_cash: float = 100_000,
                                rebalance_threshold: float = 0.20,
                                period: str = "2y",
                                interval: str = "1d") -> dict:
    """
    Run weekly rebalance simulation.

    portfolio: list of {ticker, strategy, weight} from allocator
    tickers:   all tickers to load
    initial_cash: starting capital
    rebalance_threshold: rebalance only when weight drifts > this fraction

    Returns dict with trade_log, equity_curve, summary stats.
    """
    # Load all price data
    price_data = {}
    for t in tickers:
        rows = load_price_cache(t, period, interval)
        if not rows:
            print(f"  ⚠️  No cache for {t} — skipping", file=sys.stderr)
            continue
        # Sort by date
        rows.sort(key=lambda x: x["date"])
        price_data[t] = rows

    # Find the overlapping date range
    if not price_data:
        raise ValueError("No price data loaded for any ticker")

    all_dates = set()
    for rows in price_data.values():
        for r in rows:
            all_dates.add(r["date"])
    sorted_dates = sorted(all_dates)

    # Find first Friday and first Monday
    first_friday = None
    first_monday_after = None
    for i, d in enumerate(sorted_dates):
        if is_friday(d):
            first_friday = d
            idx = sorted_dates.index(d)
            if idx + 1 < len(sorted_dates):
                first_monday_after = sorted_dates[idx + 1]
            break

    if not first_friday:
        raise ValueError(f"No Friday found in data range {sorted_dates[0]} – {sorted_dates[-1]}")

    # ── simulation state ──────────────────────────────────────────────────────
    cash      = initial_cash
    positions = {t: {"shares": 0, "avg_cost": 0.0} for t in tickers}
    trade_log = []
    equity_curve = []

    n_positions = len([p for p in portfolio if p.get("weight", 0) > 0])
    target_weight = 1.0 / n_positions if n_positions > 0 else 0

    def portfolio_value(prices: dict) -> float:
        """Current total value: cash + sum(positions × price)."""
        val = cash
        for t, pos in positions.items():
            if pos["shares"] > 0 and t in prices:
                val += pos["shares"] * prices[t]
        return val

    # Scan through dates, find Friday→Monday pairs
    dates_with_data = sorted_dates
    n = len(dates_with_data)

    pending_orders = {}   # ticker -> target_shares (set on Friday)
    pending_prices = {}   # ticker -> execution_price (Monday open)

    i = 0
    while i < n:
        d = dates_with_data[i]

        if is_friday(d):
            # Get Friday close prices
            friday_prices = {}
            for t in tickers:
                rows = price_data.get(t, [])
                row_map = {r["date"]: r for r in rows}
                if d in row_map:
                    friday_prices[t] = row_map[d]["close"]

            total_val = portfolio_value(friday_prices)
            if total_val <= 0:
                i += 1; continue

            # Determine pending orders for next Monday
            new_orders = {}
            for pos in portfolio:
                t = pos["ticker"]
                if t not in friday_prices or friday_prices[t] <= 0:
                    continue
                target_val = total_val * pos["weight"]
                target_shares = math.floor(target_val / friday_prices[t])
                new_orders[t] = target_shares

            pending_orders = new_orders
            pending_prices = {}   # will be filled Monday

            # If first Friday, place initial orders immediately (no Monday delay)
            if d == first_friday:
                # Execute immediately at Friday close for initial capital
                for t, target_shares in pending_orders.items():
                    if target_shares <= 0:
                        continue
                    price = friday_prices[t]
                    cost = target_shares * price
                    if cost > cash:
                        target_shares = math.floor(cash / price)
                        cost = target_shares * price
                    if target_shares <= 0:
                        continue
                    positions[t]["shares"]  = target_shares
                    positions[t]["avg_cost"] = price
                    cash -= cost
                    trade_log.append({
                        "date":   d,
                        "type":   "INIT",
                        "ticker": t,
                        "shares": target_shares,
                        "price":  price,
                        "value":  cost,
                        "cash":   cash,
                    })

        elif is_monday(d) and pending_orders:
            # Execute pending orders at Monday open
            monday_prices = {}
            for t in tickers:
                rows = price_data.get(t, [])
                row_map = {r["date"]: r for r in rows}
                if d in row_map:
                    monday_prices[t] = row_map[d]["open"]

            # Check rebalance trigger
            current_prices = {}
            for t in tickers:
                rows = price_data.get(t, [])
                row_map = {r["date"]: r for r in rows}
                prev_day = dates_with_data[i - 1] if i > 0 else d
                if prev_day in row_map:
                    current_prices[t] = row_map[prev_day]["close"]

            total_val = portfolio_value(current_prices)

            for t, target_shares in list(pending_orders.items()):
                price = monday_prices.get(t, 0)
                if price <= 0:
                    continue

                current_shares = positions[t]["shares"]
                deviation = abs(target_shares - current_shares) / max(current_shares, 1)

                if deviation < rebalance_threshold and current_shares > 0:
                    # Skip rebalance — within threshold; clear pending orders for this ticker
                    trade_log.append({
                        "date":   d,
                        "type":   "SKIP",
                        "ticker": t,
                        "reason": f"deviation {deviation:.1%} < {rebalance_threshold:.0%}",
                        "price":  price,
                    })
                    # Remove from pending so it's not re-evaluated next week
                    del pending_orders[t]
                    continue

                # Execute: sell old, buy new
                if current_shares > 0:
                    proceeds = current_shares * price
                    cash += proceeds
                    trade_log.append({
                        "date":   d,
                        "type":   "SELL",
                        "ticker": t,
                        "shares": current_shares,
                        "price":  price,
                        "value":  proceeds,
                        "cash":   cash,
                    })

                if target_shares > 0:
                    cost = target_shares * price
                    if cost > cash:
                        target_shares = math.floor(cash / price)
                        cost = target_shares * price
                    if target_shares > 0:
                        positions[t]["shares"]  = target_shares
                        positions[t]["avg_cost"] = price
                        cash -= cost
                        trade_log.append({
                            "date":   d,
                            "type":   "BUY",
                            "ticker": t,
                            "shares": target_shares,
                            "price":  price,
                            "value":  cost,
                            "cash":   cash,
                        })

            pending_orders = {}

        # Record equity at end of day
        day_prices = {}
        for t in tickers:
            rows = price_data.get(t, [])
            row_map = {r["date"]: r for r in rows}
            if d in row_map:
                day_prices[t] = row_map[d]["close"]
        total_val = portfolio_value(day_prices)
        equity_curve.append({
            "date": d,
            "nav":  total_val,
            "cash": cash,
        })

        i += 1

    # ── compute stats ────────────────────────────────────────────────────────
    navs = [e["nav"] for e in equity_curve]
    peak = navs[0]
    max_dd = 0.0
    for nav in navs:
        if nav > peak:
            peak = nav
        dd = (nav - peak) / peak
        if dd < max_dd:
            max_dd = dd

    total_return = (navs[-1] - initial_cash) / initial_cash
    # Annualised
    if len(navs) > 1:
        days = (len(navs) - 1)
        years = days / 252
        ann_ret = (navs[-1] / navs[0]) ** (1 / years) - 1 if years > 0 and navs[0] > 0 else 0
    else:
        ann_ret = 0

    # Sharpe approximation: monthly returns
    monthly_rets = []
    m_start = 0
    for i in range(1, len(equity_curve)):
        if i - m_start >= 21:  # roughly monthly
            r = (equity_curve[i]["nav"] - equity_curve[m_start]["nav"]) / equity_curve[m_start]["nav"]
            monthly_rets.append(r)
            m_start = i
    if len(monthly_rets) > 1:
        mean_r = sum(monthly_rets) / len(monthly_rets)
        std_r  = (sum((x - mean_r)**2 for x in monthly_rets) / max(len(monthly_rets)-1, 1)) ** 0.5
        sharpe = (mean_r / std_r) * (12 ** 0.5) if std_r > 0 else 0
    else:
        sharpe = 0

    # Win rate of rebalances
    buys  = [t for t in trade_log if t["type"] == "BUY"]
    sells = [t for t in trade_log if t["type"] == "SELL"]
    total_rebalances = len([t for t in trade_log if t["type"] in ("BUY","SELL")])

    return {
        "trade_log":      trade_log,
        "equity_curve":   equity_curve,
        "summary": {
            "initial_cash":      initial_cash,
            "final_nav":          navs[-1],
            "total_return":       total_return,
            "annualised_return":  ann_ret,
            "sharpe_ratio":       sharpe,
            "max_drawdown":       max_dd,
            "total_rebalances":   total_rebalances,
            "total_trades":       len(trade_log),
        },
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def load_latest_portfolio() -> list:
    """Load the most recent portfolio JSON."""
    files = sorted([
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("portfolio_") and f.endswith(".json")
    ])
    if not files:
        raise FileNotFoundError("No portfolio file found — run allocator first")
    with open(os.path.join(OUTPUT_DIR, files[-1])) as f:
        data = json.load(f)
    return data.get("portfolio", [])


def render_report(result: dict, portfolio: list) -> str:
    s  = result["summary"]
    tl = result["trade_log"]

    header = (
        f"\n{'='*80}\n"
        f"📊 WEEKLY REBALANCE SIMULATION\n"
        f"{'='*80}\n"
    )

    perf = (
        f"Initial capital:   ${s['initial_cash']:>12,.2f}\n"
        f"Final NAV:         ${s['final_nav']:>12,.2f}\n"
        f"Total return:       {s['total_return']:>+11.1%}\n"
        f"Annualised return:  {s['annualised_return']:>+11.1%}\n"
        f"Sharpe ratio:       {s['sharpe_ratio']:>12.2f}\n"
        f"Max drawdown:       {s['max_drawdown']:>+11.1%}\n"
        f"Total rebalances:   {s['total_rebalances']:>12}\n"
        f"Trade log entries:  {len(tl):>12}\n"
    )

    # Portfolio positions summary
    n = len(portfolio)
    pos_header = f"\n{'='*80}\nPositions: {n} equal-weight targets\n"
    pos_header += f"{'#':<3} {'Ticker':<8} {'Strategy':<20} {'Weight':>8}\n{'-'*80}\n"
    pos_lines = []
    for i, p in enumerate(portfolio):
        pos_lines.append(
            f"{i+1:<3} {p['ticker']:<8} {p['strategy']:<20} {p['weight']:>8.1%}"
        )

    # Trade log snippet
    rebal_log = [t for t in tl if t["type"] in ("BUY","SELL","INIT")]
    trade_header = f"\n{'='*80}\nTrade Log (last 20 entries)\n"
    trade_header += f"{'Date':<12} {'Type':<6} {'Ticker':<8} {'Shares':>8} {'Price':>10} {'Value':>12}\n{'-'*80}\n"
    trade_lines = []
    for t in rebal_log[-20:]:
        trade_lines.append(
            f"{t['date']:<12} {t['type']:<6} {t['ticker']:<8} "
            f"{t.get('shares',0):>8} {t.get('price',0):>10.4f} {t.get('value',0):>12.2f}"
        )

    footer = f"\n{'='*80}\n"
    return (
        header + perf + pos_header + "\n".join(pos_lines)
        + trade_header + "\n".join(trade_lines) + footer
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly portfolio rebalancer — simulate fixed-schedule trading")
    parser.add_argument("--cash",        type=float, default=100_000,
                        help="Initial capital (default $100,000)")
    parser.add_argument("--threshold",   type=float, default=0.20,
                        help="Rebalance threshold (drift %, default 0.20 = 20%%)")
    parser.add_argument("--period",      default="2y",
                        choices=["3mo","6mo","1y","2y","3y","5y"])
    parser.add_argument("--interval",    default="1d",
                        choices=["1d","5d","1wk"])
    parser.add_argument("--portfolio-json",
                        help="Path to portfolio JSON (default: latest in output/portfolio/)")
    args = parser.parse_args()

    # Load portfolio
    if args.portfolio_json:
        with open(args.portfolio_json) as f:
            port_data = json.load(f)
        portfolio = port_data.get("portfolio", [])
    else:
        portfolio = load_latest_portfolio()

    if not portfolio:
        print("⚠️  No portfolio positions found.", file=sys.stderr)
        sys.exit(1)

    tickers = [p["ticker"] for p in portfolio if p.get("ticker")]
    print(f"\n[Weekly Rebalancer]")
    print(f"  Portfolio: {', '.join(tickers)}")
    print(f"  Initial cash: ${args.cash:,.0f}")
    print(f"  Rebalance threshold: {args.threshold:.0%}")
    print(f"  Period: {args.period} | Interval: {args.interval}")

    result = simulate_weekly_rebalance(
        portfolio            = portfolio,
        tickers              = tickers,
        initial_cash         = args.cash,
        rebalance_threshold  = args.threshold,
        period               = args.period,
        interval             = args.interval,
    )

    print(render_report(result, portfolio))

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"rebalance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"💾 Saved → {out_path}")
