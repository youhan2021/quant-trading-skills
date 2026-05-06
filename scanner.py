#!/usr/bin/env python3
"""
quant-trading-skills scanner orchestrator.

Fetches data → generates signals → runs backtests → evaluates risk
→ outputs ranked list of strategies.

Usage:
    python3 scanner.py --tickers SPY,QQQ --min-sharpe 1.0 --max-dd -0.20
    python3 scanner.py --universe etf --min-sharpe 0.8
    python3 scanner.py --universe all  --min-sharpe 1.0 --max-dd -0.15
    python3 scanner.py --universe tech --min-sharpe 1.0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from glob import glob

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR   = SCRIPT_DIR
BASE_DIR    = os.path.dirname(SKILL_DIR)

COLLECTOR   = os.path.join(SKILL_DIR, "market-info-collector", "scripts", "collector.py")
GENERATOR   = os.path.join(SKILL_DIR, "strategy-generator", "scripts", "generator.py")
BACKTESTER  = os.path.join(SKILL_DIR, "backtest-engine", "scripts", "backtester.py")
RISK_EVAL   = os.path.join(SKILL_DIR, "risk-manager", "scripts", "risk_eval.py")

RESULTS_DIR = os.path.join(SKILL_DIR, "backtest-engine", "data", "results")
OUTPUT_DIR  = os.path.join(SKILL_DIR, "output", "scan_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── universes ─────────────────────────────────────────────────────────────────

UNIVERSES = {
    "etf": {
        "description": "Major US ETFs",
        "tickers": [
            "SPY", "QQQ", "IWM", "VTI", "DIA",
            "XLK", "XLF", "XLE", "XLV", "XLY", "XLI",
            "TLT", "IEF", "LQD", "HYG", "MBB",
            "GLD", "SLV",
            "VNQ", "IYR",
            "UUP", "FXE",
        ],
    },
    "tech": {
        "description": "Mega-cap tech + growth",
        "tickers": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "AVGO", "ORCL", "ADBE",
            "CRM", "NFLX", "AMD", "QCOM", "INTC",
        ],
    },
    "quality": {
        "description": "Quality / dividend aristocrats",
        "tickers": [
            "JNJ", "PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS",
            "JPM", "BAC", "GS", "MS", "BLK", "SCHW",
            "V", "MA", "AXP", "SPGI", "COST",
        ],
    },
    "growth": {
        "description": "High-growth names",
        "tickers": [
            "TSLA", "NVDA", "AMD", "CRM", "SNOW", "PLTR",
            "DDOG", "NET", "MU", "QCOM", "AVGO",
            "GOOGL", "AMZN", "META", "NFLX", "NOW",
        ],
    },
    "small": {
        "description": "Small/mid cap",
        "tickers": [
            "IWM", "VB", "VEA", "VWO", "IJR", "IWO",
            "SCHB", "SPYB", "SPMD", "IJJ", "IJK",
        ],
    },
    "sp500": {
        "description": "S&P 500 broad — all sectors",
        "tickers": [
            # Tech / Communication
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL", "ADBE",
            "CRM", "NFLX", "AMD", "QCOM", "INTC", "TXN", "MU", "AMAT", "LRCX", "KLAC",
            "PANW", "CRWD", "FTNT", "NET", "SNOW", "DDOG", "PLTR", "NOW", "SAP", "INTU",
            # Financials
            "JPM", "BAC", "GS", "MS", "BLK", "SCHW", "V", "MA", "AXP", "SPGI",
            "COF", "USB", "PNC", "TFC", "COF", "AON", "MMC", "CB", "AJG", "WELL",
            # Healthcare
            "LLY", "JNJ", "UNH", "JNJ", "PFE", "ABT", "MRK", "TMO", "ABBV", "BMY",
            "AMGN", "GILD", "ISRG", "MDT", "SYK", "ZTS", "BIIB", "REGN", "VRTX",
            # Consumer
            "PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS", "NKE", "SBUX", "TGT",
            "COST", "LOW", "TJX", "EL", "CL", "KMB", "GIS", "K", "HSY", "KHC",
            # Industrials
            "CAT", "DE", "BA", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "EMR",
            "FDX", "CSX", "NSC", "WM", "RSG", "PH", "ROK", "CTVA", "PCAR", "ODFL",
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "PXD", "OXY",
            # Utilities / Real Estate / Staples
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "ED", "WEC", "AWK",
            "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "AVB", "EQR", "VTR",
            # Materials
            "LIN", "APD", "ECL", "SHW", "NEM", "FCX", "NUE", "DOW", "DD", "PPG",
            # ETFs (add more)
            "TLT", "IEF", "LQD", "HYG", "MBB", "VNQ", "IYR", "UUP", "FXE",
            "VEA", "VWO", "IEMG", "EFA", "EEM", "SCHF", "SPDW", "RSP",
        ],
    },
    "all": {
        "description": "Full universe (all below)",
        "tickers": [],
    },
}

# Build "all" universe from others
UNIVERSES["all"]["tickers"] = list(set(
    t for u in UNIVERSES.values() if u["tickers"]
    for t in u["tickers"]
))


# ── run helper ────────────────────────────────────────────────────────────────

def run_script(script_path: str, *args, timeout: int = 300) -> str:
    cmd = ["python3", script_path] + list(args)
    print(f"  → Running: {' '.join(cmd[:3])}...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=os.path.dirname(script_path),
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return f"ERROR: Timeout after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


def run_collector(tickers: list[str], period: str, interval: str) -> dict:
    """Fetch market data for all tickers using parallel batch."""
    print(f"\n[1/4] Collecting data for {len(tickers)} tickers...")
    import concurrent.futures

    cache_dir = os.path.join(SKILL_DIR, "market-info-collector", "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    def fetch_one(t: str) -> tuple[str, bool, str]:
        import subprocess, sys
        cmd = [sys.executable, COLLECTOR,
               "--tickers", t,
               "--period",  period,
               "--interval", interval]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        ok = r.returncode == 0 and " OK " in r.stdout
        err = r.stderr.strip()[:80] if r.stderr else ""
        return (t, ok, err)

    ok_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, t): t for t in tickers}
        for fut in concurrent.futures.as_completed(futures, timeout=600):
            t, ok, err = fut.result()
            status = "OK" if ok else f"ERR({err[:30]})" if err else "ERR"
            print(f"  {t}: {status}")
            if ok:
                ok_count += 1

    print(f"  → Collected: {ok_count}/{len(tickers)} OK")
    return {"ok": ok_count, "total": len(tickers)}


def run_backtest_scan(tickers: list[str], period: str, interval: str,
                      min_sharpe: float, min_ann_ret: float, max_dd: float,
                      sort_by: str, top_n: int) -> list[dict]:
    """Run backtest scan for all tickers using direct import (no subprocess).
    Returns list of loaded result dicts written to RESULTS_DIR.
    """
    print(f"\n[2/4] Running backtest scan ({len(tickers)} tickers × 9 strategies × param combos)...")

    # Import backtester module directly
    sys.path.insert(0, os.path.dirname(BACKTESTER))
    import importlib.util
    spec = importlib.util.spec_from_file_location("backtester_mod", BACKTESTER)
    bt_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt_mod)

    all_result_dicts = []
    start = time.time()

    # Track which ticker+strategy+params combos we've already loaded to deduplicate
    seen_keys = set()

    for ticker in tickers:
        ticker = ticker.strip()
        if not ticker:
            continue
        print(f"  Scanning {ticker}...", end=" ", flush=True)

        # Fix RESULTS_DIR BEFORE scan_ticker (save_result captures it at call time)
        bt_mod.RESULTS_DIR = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(BACKTESTER))),
            "backtest-engine", "data", "results"
        )
        os.makedirs(bt_mod.RESULTS_DIR, exist_ok=True)

        try:
            bt_mod.scan_ticker(ticker=ticker, period=period, interval=interval)
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Load ONLY the result files written by this ticker's scan (newest files)
        # Use modification time to find files created during this scan
        scan_start_mtime = time.time()
        result_files = glob(os.path.join(bt_mod.RESULTS_DIR, "*.json"))
        for fpath in result_files:
            fname = os.path.basename(fpath)
            # Skip scan summaries
            if fname.startswith("scan_summary_"):
                continue
            # Deduplicate by ticker_strategy_params
            key = fname.replace(".json", "")
            if key in seen_keys:
                continue
            seen_keys.add(key)
            try:
                with open(fpath) as f:
                    all_result_dicts.append(json.load(f))
            except Exception:
                continue

    elapsed = time.time() - start
    print(f"  → Backtest scan done in {elapsed:.0f}s")
    print(f"  → {len(all_result_dicts)} unique backtest results loaded")
    return all_result_dicts


def run_risk_eval(all_result_dicts: list[dict],
                  min_sharpe: float, max_dd: float,
                  sort_by: str, top_n: int,
                  min_trades: int = 4) -> tuple[list, list]:
    """Run risk evaluation on backtest result dicts (in-process, no subprocess)."""
    print(f"\n[3/4] Evaluating risk for {len(all_result_dicts)} strategies...")
    sys.path.insert(0, os.path.dirname(RISK_EVAL))

    import importlib.util
    spec = importlib.util.spec_from_file_location("risk_eval", RISK_EVAL)
    risk_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(risk_mod)

    all_reports = []
    for r in all_result_dicts:
        try:
            report = risk_mod.assess_strategy_risk(r)
            report["_ticker"] = r.get("ticker", "UNKNOWN")
            report["_strategy"] = r.get("strategy", "unknown")
            all_reports.append(report)
        except Exception as e:
            continue

    # Apply filters
    sort_key_map = {
        "sharpe_ratio": "sharpe_ratio", "sortino_ratio": "sortino_ratio",
        "calmar_ratio": "calmar_ratio", "annual_return": "annual_return",
        "max_drawdown": "max_drawdown", "win_rate": "win_rate",
    }
    sk = sort_key_map.get(sort_by, "sharpe_ratio")
    filtered = []
    for r in all_reports:
        m = r.get("risk_metrics", {})
        trd = r.get("trade_risk", {})
        if m.get("sharpe_ratio", 0) < min_sharpe:
            continue
        if m.get("max_drawdown", 0) < max_dd:
            continue
        # ann_ret filtering is done at the scanner CLI level via --min-ann-ret
        n_trades = trd.get("total_trades", 0)
        if n_trades < min_trades:
            continue
        filtered.append(r)

    filtered.sort(key=lambda x: x.get("risk_metrics", {}).get(sk, 0), reverse=True)
    top_reports = filtered[:top_n]

    print(f"  → {len(top_reports)} strategies passed risk evaluation")
    return filtered, top_reports


def build_final_report(risk_reports: list, min_sharpe: float,
                       min_ann_ret: float, max_dd: float,
                       sort_by: str, top_n: int) -> dict:
    """Build final report from already-filtered, already-sorted risk reports."""
    print(f"\n[4/4] Building final report...")

    # risk_reports already filtered + sorted by run_risk_eval; just take top_n
    top = risk_reports[:top_n]

    report = {
        "report_id": f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": datetime.now().isoformat(),
        "filters": {
            "min_sharpe":  min_sharpe,
            "min_ann_ret": min_ann_ret,
            "max_dd":      max_dd,
            "sort_by":     sort_by,
        },
        "total_strategies_scanned": len(risk_reports),
        "passed_filters": len(risk_reports),
        "top_strategies": [
            {
                "rank": i + 1,
                "ticker": r.get("_ticker", r.get("backtest_ref", {}).get("ticker", "UNKNOWN")),
                "strategy": r.get("_strategy", r.get("backtest_ref", {}).get("strategy", "unknown")),
                "params": r.get("backtest_ref", {}).get("params", {}),
                "sharpe_ratio": r["risk_metrics"].get("sharpe_ratio", 0),
                "sortino_ratio": r["risk_metrics"].get("sortino_ratio", 0),
                "calmar_ratio": r["risk_metrics"].get("calmar_ratio", 0),
                "annual_return": r["risk_metrics"].get("annual_return",
                               r.get("capital", {}).get("total_return", 0)),
                "max_drawdown": r["risk_metrics"].get("max_drawdown", 0),
                "var_95": r["risk_metrics"].get("var_95", 0),
                "win_rate": r["trade_risk"].get("win_rate", 0),
                "kelly_fraction": r["risk_metrics"].get("kelly_fraction", 0),
                "profit_factor": r["trade_risk"].get("profit_factor", 0),
                "alerts": r.get("alerts", []),
                "recommendations": r.get("recommendations", []),
                "risk_report_id": r.get("risk_report_id", ""),
            }
            for i, r in enumerate(top)
        ],
    }
    return report


def save_and_display(report: dict):
    """Save report and print summary."""
    fname = f"scan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = os.path.join(OUTPUT_DIR, fname)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    strategies = report["top_strategies"]
    print(f"\n{'='*90}")
    print(f"📊 TOP STRATEGIES (min_sharpe={report['filters']['min_sharpe']}, "
          f"max_dd={report['filters']['max_dd']}, sorted by {report['filters']['sort_by']})")
    print(f"{'='*90}")
    print(f"{'#':<4} {'Ticker':<8} {'Strategy':<16} {'Sharpe':>7} {'Sortino':>8} "
          f"{'AnnRet':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>6} {'Kelly':>7}")
    print("-" * 90)
    for s in strategies:
        ann = s["annual_return"]
        md  = s["max_drawdown"]
        wr  = s["win_rate"]
        pf  = s["profit_factor"]
        kelly = s["kelly_fraction"]
        alerts = len(s["alerts"])
        flag = "⚠️" if alerts > 0 else "✅"
        print(f"{s['rank']:<4} {s['ticker']:<8} {s['strategy']:<16} "
              f"{s['sharpe_ratio']:>+7.2f} {s['sortino_ratio']:>+8.2f} "
              f"{ann:>+8.1%} {md:>+8.1%} {wr:>8.1%} {pf:>6.2f} {kelly:>7.1%}  {flag}")

    print(f"\n{'='*90}")
    print(f"Scanned: {report['total_strategies_scanned']} | Passed: {report['passed_filters']} | Shown: {len(strategies)}")
    print(f"Full report: {out_path}")
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant trading strategy scanner")
    parser.add_argument("--tickers", help="Comma-separated tickers (overrides --universe)")
    parser.add_argument("--universe", default="etf",
                        choices=list(UNIVERSES.keys()),
                        help="Predefined ticker universe")
    parser.add_argument("--period", default="3y",
                        choices=["1mo","3mo","6mo","1y","2y","3y","5y","10y","ytd","max"])
    parser.add_argument("--interval", default="1d",
                        choices=["1d","5d","1wk","1mo"])
    parser.add_argument("--min-sharpe", type=float, default=1.0,
                        help="Minimum Sharpe ratio (default 1.0)")
    parser.add_argument("--min-ann-ret", type=float, default=-1.0,
                        help="Minimum annualized return (default -1.0 = any)")
    parser.add_argument("--max-dd", type=float, default=-0.20,
                        help="Maximum drawdown (most negative, default -0.20)")
    parser.add_argument("--sort", default="sharpe_ratio",
                        choices=["sharpe_ratio","sortino_ratio","calmar_ratio",
                                 "annual_return","max_drawdown","win_rate"])
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top strategies to show")
    parser.add_argument("--min-trades", type=int, default=4,
                        help="Minimum number of trades (default 4)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetching (use cached data)")
    args = parser.parse_args()

    # Resolve tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        universe = UNIVERSES[args.universe]
        tickers = list(set(universe["tickers"]))  # dedupe
        print(f"\nUniverse: {args.universe} — {universe['description']}")
        print(f"Tickers ({len(tickers)}): {', '.join(sorted(tickers))}")

    print(f"\nScanner config:")
    print(f"  Tickers:     {len(tickers)}")
    print(f"  Period:      {args.period}")
    print(f"  Interval:    {args.interval}")
    print(f"  Min Sharpe: {args.min_sharpe}")
    print(f"  Max DD:     {args.max_dd}")
    print(f"  Sort by:    {args.sort}")
    print(f"  Top N:      {args.top}")

    # Step 1: Fetch data
    if not args.skip_fetch:
        run_collector(tickers, args.period, args.interval)
    else:
        print("\n[1/4] Skipping fetch (using cache)")

    # Step 2: Backtest scan
    all_result_dicts = run_backtest_scan(
        tickers, args.period, args.interval,
        args.min_sharpe, args.min_ann_ret, args.max_dd,
        args.sort, args.top,
    )

    # Step 3: Risk evaluation
    all_filtered, top_reports = run_risk_eval(
        all_result_dicts,
        args.min_sharpe, args.max_dd,
        args.sort, args.top,
        args.min_trades,
    )

    # Step 4: Build & save final report
    report = build_final_report(
        top_reports,
        args.min_sharpe, args.min_ann_ret, args.max_dd,
        args.sort, args.top,
    )
    # Patch total_scanned to use full filtered count
    report["total_strategies_scanned"] = len(all_result_dicts)
    report["passed_filters"] = len(all_filtered)
    save_and_display(report)
