#!/usr/bin/env python3
"""
market-info-collector/scripts/collector.py
US stock/ETF market data fetching via yfinance.

Usage:
    python3 collector.py SPY            # single ticker, default 2y
    python3 collector.py SPY QQQ TLT    # multiple tickers
    python3 collector.py --tickers SPY,QQQ --period 1y --interval 1d
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(SKILL_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── yfinance import ─────────────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: uv tool install yfinance")
    sys.exit(1)

# ── default universe ─────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    # Major ETFs
    "SPY", "QQQ", "IWM", "VTI", "DIA",          # Broad market
    "TLT", "IEF", "LQD", "HYG", "MBB",            # Bonds
    "GLD", "SLV", "BCI",                          # Commodities
    "VNQ", "IYR",                                  # Real estate
    "UUP", "FXE", "FXY",                          # Currency
    "XLE", "XLF", "XLK", "XLV", "XLY", "XLI",    # Sectors
    # Mega-cap tech / quality
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Defensive / dividend quality
    "JNJ", "PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS",
    # Financials
    "JPM", "BAC", "GS", "MS", "BLK",
    # Growth at reasonable price
    "BRK.B", "V", "MA", "ADBE", "CRM", "ORCL",
]


def fetch_ticker(ticker: str, period: str = "2y", interval: str = "1d") -> dict:
    """
    Fetch OHLCV data for a single ticker.
    Returns dict with metadata + DataFrame rows.
    """
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{period}_{interval}.json")

    # Return cached if exists and < 1 hour old
    if os.path.exists(cache_file):
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        if age.total_seconds() < 3600:
            with open(cache_file) as f:
                return json.load(f)

    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

    if df.empty:
        return {"ticker": ticker, "error": "No data returned"}

    records = []
    for date, row in df.iterrows():
        records.append({
            "date": date.isoformat(),
            "open":  round(float(row["Open"]),  4),
            "high":  round(float(row["High"]),  4),
            "low":   round(float(row["Low"]),   4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]),
        })

    result = {
        "ticker":    ticker,
        "period":    period,
        "interval":  interval,
        "start":     records[0]["date"] if records else None,
        "end":       records[-1]["date"] if records else None,
        "count":     len(records),
        "data":      records,
    }

    # Cache
    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result


def fetch_multiple(tickers: list[str], period: str = "2y", interval: str = "1d") -> dict:
    """Fetch multiple tickers."""
    results = {}
    for t in tickers:
        print(f"  Fetching {t} ...", end=" ", flush=True)
        r = fetch_ticker(t, period, interval)
        if "error" in r:
            print(f"ERROR: {r['error']}")
        else:
            print(f"OK ({r['count']} bars)")
        results[t] = r
    return results


def load_from_cache(ticker: str, period: str = "2y", interval: str = "1d") -> dict | None:
    """Load ticker from cache if exists."""
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{period}_{interval}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    return None


def list_cached_tickers() -> list[str]:
    """Return list of all cached tickers."""
    if not os.path.exists(CACHE_DIR):
        return []
    return sorted(set(
        f.rsplit("_", 2)[0]
        for f in os.listdir(CACHE_DIR)
        if f.endswith(".json")
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market data collector")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols")
    parser.add_argument("--tickers", help="Comma-separated tickers (alt)")
    parser.add_argument("--period", default="2y",
                        choices=["1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"])
    parser.add_argument("--interval", default="1d",
                        choices=["1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo"])
    parser.add_argument("--list-cached", action="store_true")
    parser.add_argument("--force", action="store_true", help="Bypass cache")
    args = parser.parse_args()

    if args.list_cached:
        cached = list_cached_tickers()
        print(f"Cached tickers ({len(cached)}): {', '.join(cached)}")
        sys.exit(0)

    # Resolve ticker list
    tickers = []
    if args.tickers:
        tickers = args.tickers   # positional args
    elif args.tickers and "," in str(args.tickers):
        tickers = [t.strip().upper() for t in str(args.tickers).split(",") if t.strip()]
    elif DEFAULT_TICKERS:
        tickers = DEFAULT_TICKERS
    else:
        print("No tickers specified and no default list.")
        sys.exit(1)

    print(f"\nFetching {len(tickers)} tickers | period={args.period} | interval={args.interval}\n")
    results = fetch_multiple(tickers, args.period, args.interval)

    errors = {t: r["error"] for t, r in results.items() if "error" in r}
    if errors:
        print(f"\n⚠️  Failed tickers: {list(errors.keys())}")
    else:
        print(f"\n✅  All {len(tickers)} tickers fetched successfully")
