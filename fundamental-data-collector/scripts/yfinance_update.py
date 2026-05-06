#!/usr/bin/env python3
"""
yfinance Realtime Update Fetcher
Updates fundamental data using yfinance for current values.

Usage:
    python yfinance_update.py --tickers AAPL,MSFT --output ~/.hermes/fundamental_data/yfinance/
    python yfinance_update.py --all --tickers-file ~/.hermes/fundamental_data/tickers.json --output ~/.hermes/fundamental_data/yfinance/
"""

import json
import time
import argparse
import os
import sys

try:
    import yfinance as yf
    from yfinance import fast_info
except ImportError:
    print("yfinance not installed: pip install yfinance")
    sys.exit(1)


YF_FIELDS = [
    "marketCap", "sharesOutstanding", "trailingPE", "forwardPE",
    "trailingEps", "forwardEps", "bookValue", "totalRevenue",
    "netIncomeToCommon", "returnOnEquity", "debtToEquity",
    "currentRatio", "quickRatio", "profitMargins", "operatingMargins",
    "ebitda", "totalCash", "freeCashflow", "operatingCashflow",
    "ebitdaMargins", "grossMargins", "revenueGrowth", "earningsGrowth",
]


def fetch_yfinance(ticker: str) -> dict:
    """Fetch current fundamental data for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # Try fast_info for market data
        try:
            fi = fast_info.get(ticker)
            market_cap = getattr(fi, "market_cap", None) or info.get("marketCap")
            shares = getattr(fi, "shares", None) or info.get("sharesOutstanding")
            price = getattr(fi, "last_price", None) or info.get("currentPrice")
        except Exception:
            market_cap = info.get("marketCap")
            shares = info.get("sharesOutstanding")
            price = info.get("currentPrice") or info.get("regularMarketPrice")

        return {
            "ticker": ticker,
            "fetch_date": time.strftime("%Y-%m-%d"),
            "data_source": "yfinance",
            "market_cap": market_cap,
            "shares_outstanding": shares,
            "price": price,
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "eps_trailing": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
            "book_value_per_share": info.get("bookValue"),
            "revenue": info.get("totalRevenue"),
            "net_income": info.get("netIncomeToCommon"),
            "roe": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
            "ebitda": info.get("ebitda"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def save_result(result: dict, output_dir: str) -> str:
    """Save ticker data to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    ticker = result["ticker"]
    safe_ticker = ticker.replace("/", "_").replace("\\", "_")
    path = os.path.join(output_dir, f"{safe_ticker}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def load_tickers(tickers_file: str) -> list:
    """Load ticker list from JSON file."""
    if not os.path.exists(tickers_file):
        print(f"Tickers file not found: {tickers_file}")
        return []
    with open(tickers_file) as f:
        data = json.load(f)
    if isinstance(data, dict) and "tickers" in data:
        return [t["ticker"] for t in data["tickers"]]
    if isinstance(data, list):
        return list(data)
    return []


def main():
    parser = argparse.ArgumentParser(description="Fetch yfinance fundamental data")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers")
    parser.add_argument("--tickers-file", type=str, help="Path to tickers.json")
    parser.add_argument("--output", type=str, default="~/.hermes/fundamental_data/yfinance/",
                        help="Output directory")
    parser.add_argument("--all", action="store_true", help="Fetch all tickers from file")
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output)

    # Resolve ticker list
    if args.all or args.tickers_file:
        tickers_file = args.tickers_file or "~/.hermes/fundamental_data/tickers.json"
        ticker_list = load_tickers(tickers_file)
    elif args.tickers:
        ticker_list = [t.strip() for t in args.tickers.split(",")]
    else:
        print("Error: specify --tickers or --tickers-file or --all")
        sys.exit(1)

    print(f"Fetching {len(ticker_list)} tickers -> {output_dir}")

    success = 0
    for ticker in ticker_list:
        result = fetch_yfinance(ticker)
        if result:
            save_result(result, output_dir)
            success += 1
            print(f"  {ticker}: OK (market_cap={result.get('market_cap', 'N/A')})")
        else:
            print(f"  {ticker}: FAILED")

    print(f"\nDone: {success}/{len(ticker_list)} succeeded")


if __name__ == "__main__":
    main()
