#!/usr/bin/env python3
"""
Merge SEC XBRL + yfinance data into unified format.

Usage:
    python merge_fundamental.py --ticker AAPL --sec-dir ... --yf-dir ... --output merged/
    python merge_fundamental.py --all --sec-dir ... --yf-dir ... --output ...
"""

import json
import argparse
import os
import sys
from pathlib import Path


FILING_LAG_DAYS = 60  # Quarterly earnings typically public 60 days after fiscal quarter end


def load_sec_xbrl(ticker: str, sec_dir: str) -> dict:
    """Load SEC XBRL data for a ticker."""
    safe_ticker = ticker.replace("/", "_").replace("\\", "_")
    path = os.path.join(sec_dir, f"{safe_ticker}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_yfinance(ticker: str, yf_dir: str) -> dict:
    """Load yfinance data for a ticker."""
    safe_ticker = ticker.replace("/", "_").replace("\\", "_")
    path = os.path.join(yf_dir, f"{safe_ticker}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def merge_data(ticker: str, sec_data: dict, yf_data: dict) -> dict:
    """Merge SEC XBRL historical + yfinance current into unified structure."""
    result = {
        "ticker": ticker,
        "cik": sec_data.get("cik", yf_data.get("cik", "")) if sec_data else (yf_data.get("cik", "") if yf_data else ""),
        "last_updated": yf_data.get("fetch_date", "") if yf_data else "",
        "data_source": "sec_xbrl+yfinance",
        "income_statement": {},
        "balance_sheet": {},
        "shares_outstanding": {"current": None, "source": "yfinance"},
        "market_cap": {"current": None, "source": "yfinance"},
        "ratios": {},
    }

    # yfinance current values
    if yf_data:
        result["shares_outstanding"]["current"] = yf_data.get("shares_outstanding")
        result["market_cap"]["current"] = yf_data.get("market_cap")
        result["market_cap"]["price"] = yf_data.get("price")
        result["ratios"]["roe"] = [{"date": yf_data.get("fetch_date", ""), "val": yf_data.get("roe"), "source": "yfinance"}] if yf_data.get("roe") else []
        result["ratios"]["debt_to_equity"] = [{"date": yf_data.get("fetch_date", ""), "val": yf_data.get("debt_to_equity"), "source": "yfinance"}] if yf_data.get("debt_to_equity") else []
        result["ratios"]["pe_trailing"] = [{"date": yf_data.get("fetch_date", ""), "val": yf_data.get("pe_trailing"), "source": "yfinance"}] if yf_data.get("pe_trailing") else []

    # SEC XBRL historical
    if sec_data:
        concepts = sec_data.get("concepts", {})

        # Income statement
        for concept, meta in concepts.items():
            if concept in ("Revenues", "NetIncomeLoss", "GrossProfit", "OperatingIncome"):
                key = concept.lower().replace("netincomeloss", "net_income_loss").replace("revenues", "revenues")
                result["income_statement"][key] = meta.get("data", [])

        # Balance sheet
        for concept, meta in concepts.items():
            if concept in ("StockholdersEquity", "Assets", "Liabilities"):
                key = concept.lower().replace("stockholdersequity", "stockholders_equity")
                result["balance_sheet"][key] = meta.get("data", [])

        # Shares outstanding
        if "WeightedAverageSharesOutstandingDiluted" in concepts:
            result["shares_outstanding"]["historical"] = concepts["WeightedAverageSharesOutstandingDiluted"].get("data", [])

    return result


def save_merged(result: dict, output_dir: str) -> str:
    """Save merged data to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    ticker = result["ticker"]
    safe_ticker = ticker.replace("/", "_").replace("\\", "_")
    path = os.path.join(output_dir, f"{safe_ticker}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Merge SEC XBRL + yfinance data")
    parser.add_argument("--ticker", type=str, help="Single ticker")
    parser.add_argument("--sec-dir", type=str, default="~/.hermes/fundamental_data/sec_xbrl/")
    parser.add_argument("--yf-dir", type=str, default="~/.hermes/fundamental_data/yfinance/")
    parser.add_argument("--output", type=str, default="~/.hermes/fundamental_data/merged/")
    parser.add_argument("--all", action="store_true", help="Merge all tickers")
    args = parser.parse_args()

    sec_dir = os.path.expanduser(args.sec_dir)
    yf_dir = os.path.expanduser(args.yf_dir)
    output_dir = os.path.expanduser(args.output)

    if args.all:
        # Find all tickers from sec_xbrl directory
        tickers = [f.replace(".json", "") for f in os.listdir(sec_dir) if f.endswith(".json")]
    elif args.ticker:
        tickers = [args.ticker]
    else:
        print("Error: specify --ticker or --all")
        sys.exit(1)

    print(f"Merging {len(tickers)} tickers -> {output_dir}")

    success = 0
    for ticker in tickers:
        # Handle safe ticker names
        safe_ticker = ticker.replace("/", "_").replace("\\", "_")
        sec_path = os.path.join(sec_dir, f"{safe_ticker}.json")
        yf_path = os.path.join(yf_dir, f"{safe_ticker}.json")

        sec_data = json.load(open(sec_path)) if os.path.exists(sec_path) else None
        yf_data = json.load(open(yf_path)) if os.path.exists(yf_path) else None

        if not sec_data and not yf_data:
            print(f"  {ticker}: no data found")
            continue

        result = merge_data(ticker, sec_data, yf_data)
        save_merged(result, output_dir)
        success += 1

        n_income = sum(len(v) for v in result["income_statement"].values())
        n_balance = sum(len(v) for v in result["balance_sheet"].values())
        print(f"  {ticker}: {n_income} income pts, {n_balance} balance pts, "
              f"shares={result['shares_outstanding']['current'] or 'N/A'}")

    print(f"\nDone: {success}/{len(tickers)} merged")


if __name__ == "__main__":
    main()
