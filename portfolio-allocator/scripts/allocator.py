#!/usr/bin/env python3
"""
Portfolio Allocator — combines stock selection + timing signals into a ranked portfolio.

Workflow:
1. Load backtest results from backtest-engine/data/results/
2. For each ticker, pick the best-performing strategy (highest risk-adjusted score)
3. Rank tickers by their best strategy score
4. Apply allocation rules:
   - TOP tier (Sharpe≥1.5, AnnRet>0, MaxDD≥-15%): up to `max_positions` equal-weight slots
   - WATCH tier (Sharpe≥1.0, AnnRet≥0, MaxDD≥-20%): fill remaining slots if room
5. Output portfolio positions with entry/exit signal, size, and confidence score
"""

import argparse
import json
import glob
import os
import sys
from datetime import datetime

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(SKILL_DIR, "backtest-engine/data/results")
OUTPUT_DIR  = os.path.join(SKILL_DIR, "output/portfolio")

os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings_filtered = False

# ── helpers ──────────────────────────────────────────────────────────────────

def load_all_results():
    """Load all backtest result JSON files."""
    files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    results = []
    for fp in files:
        fname = os.path.basename(fp)
        if "summary" in fname or "scan" in fname:
            continue
        try:
            with open(fp) as f:
                d = json.load(f)
            results.append(d)
        except Exception:
            continue
    return results


def score_strategy(r: dict) -> float:
    """Risk-adjusted composite score for ranking strategies."""
    p  = r.get("performance", {})
    ts = r.get("trade_stats",  {})

    sharpe   = p.get("sharpe_ratio", 0)
    sortino  = p.get("sortino_ratio", 0)
    calmar   = p.get("calmar_ratio", 0)
    ann_ret  = p.get("annualized_return", 0)
    max_dd   = p.get("max_drawdown", 0)
    n_trades = ts.get("total_trades", 0)
    win_rate = p.get("win_rate", 0)
    pf       = p.get("profit_factor", 0)

    if n_trades < 3 or sharpe <= 0 or max_dd == 0:
        return -999

    # Composite: weighted blend — prioritise Sharpe, penalise drawdown
    score = (sharpe * 0.35
           + sortino * 0.10
           + max(ann_ret * 100 * 0.20, 0)
           + max(calmar, 0) * 0.10
           + win_rate * 0.10
           + max(pf - 1, 0) * 0.05)

    # Drawdown penalty: exponential
    dd_penalty = abs(min(max_dd, 0)) ** 1.5 * 0.5
    score -= dd_penalty
    return score


def classify_strategy(r: dict) -> str:
    """Classify strategy into TOP / WATCH / SKIP tiers."""
    p  = r.get("performance", {})
    ts = r.get("trade_stats", {})
    ann_ret  = p.get("annualized_return", 0)
    max_dd   = p.get("max_drawdown", 0)
    sharpe   = p.get("sharpe_ratio", 0)
    n_trades = ts.get("total_trades", 0)

    if sharpe >= 1.5 and ann_ret > 0 and max_dd >= -0.15 and n_trades >= 3:
        return "TOP"
    elif sharpe >= 1.0 and ann_ret >= 0 and max_dd >= -0.20 and n_trades >= 3:
        return "WATCH"
    return "SKIP"


def select_best_per_ticker(results: list) -> dict:
    """For each ticker, pick the single best strategy by composite score.
    Strategy must have >= 3 trades to be considered at all.
    """
    best = {}
    for r in results:
        ticker   = r.get("ticker", "UNKNOWN")
        strategy = r.get("strategy", "unknown")
        params   = r.get("params", {})
        p  = r.get("performance", {})
        ts = r.get("trade_stats",  {})

        n_trades = ts.get("total_trades", 0)
        if n_trades < 3:
            continue

        score = score_strategy(r)
        if score <= -999:
            continue

        tier = classify_strategy(r)
        if tier == "SKIP":
            continue

        cur = best.get(ticker)
        if cur is None or score > cur["score"]:
            best[ticker] = {
                "ticker":   ticker,
                "strategy": strategy,
                "params":   params,
                "score":    score,
                "tier":     tier,
                "sharpe":   p.get("sharpe_ratio", 0),
                "sortino":  p.get("sortino_ratio", 0),
                "calmar":   p.get("calmar_ratio", 0),
                "ann_ret":  p.get("annualized_return", 0),
                "max_dd":   p.get("max_drawdown", 0),
                "n_trades": n_trades,
                "win_rate": p.get("win_rate", 0),
                "pf":       p.get("profit_factor", 0),
                "kelly":    0.0,
                "var_95":   0.0,
            }
    return best


def allocate_portfolio(best_by_ticker: dict, max_positions: int = 6) -> list:
    """
    Build a ranked portfolio from best strategies per ticker.

    Rules:
    - TOP tier fills up to max_positions (equal weight each)
    - Remaining slots go to WATCH tier
    - Weight = 1/N normalisation within each tier
    - Confidence = normalised score within tier
    """
    tops   = sorted([x for x in best_by_ticker.values() if x["tier"] == "TOP"],
                    key=lambda x: x["score"], reverse=True)
    watches = sorted([x for x in best_by_ticker.values() if x["tier"] == "WATCH"],
                      key=lambda x: x["score"], reverse=True)

    portfolio = []
    slots_used = 0

    # TOP tier
    for t in tops:
        if slots_used >= max_positions:
            break
        portfolio.append(t)
        slots_used += 1

    # WATCH tier — fill remaining
    remaining = max_positions - slots_used
    for w in watches[:remaining]:
        portfolio.append(w)

    # Assign weights
    n = len(portfolio)
    for i, pos in enumerate(portfolio):
        pos["weight"]    = 1.0 / n if n > 0 else 0
        pos["rank"]      = i + 1
        pos["confidence"] = min(pos["score"] / 20.0, 1.0) if pos["score"] > 0 else 0

    return portfolio


def render_portfolio(portfolio: list) -> str:
    """Render portfolio as a text table."""
    if not portfolio:
        return "\n⚠️  No strategies qualify for the portfolio.\n"

    header = (
        f"\n{'='*100}\n"
        f"📈 PORTFOLIO — {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {len(portfolio)} positions (equal-weight)\n"
        f"{'='*100}\n"
        f"{'#':<3} {'Ticker':<8} {'Strategy':<18} {'Tier':<6} {'Score':>7} {'Sharpe':>7} "
        f"{'AnnRet':>8} {'MaxDD':>8} {'Trades':>7} {'Kelly':>7} {'Weight':>7} {'Conf':>6}\n"
        f"{'-'*100}"
    )
    lines = [header]
    for p in portfolio:
        flag = "🟢" if p["tier"] == "TOP" else "🟡"
        lines.append(
            f"{p['rank']:<3} {p['ticker']:<8} {p['strategy']:<18} {p['tier']:<6} "
            f"{p['score']:>7.2f} {p['sharpe']:>7.2f} {p['ann_ret']:>+8.1%} "
            f"{p['max_dd']:>+8.1%} {p['n_trades']:>7} "
            f"{p['kelly']:>7.1%} {p['weight']:>7.1%} {p['confidence']:>6.1%}  {flag}"
        )
    lines.append(f"{'='*100}\n")
    return "\n".join(lines)


def build_report(portfolio: list, best_by_ticker: dict,
                  min_sharpe: float, max_dd: float) -> dict:
    tops   = [x for x in portfolio if x["tier"] == "TOP"]
    watches = [x for x in portfolio if x["tier"] == "WATCH"]

    return {
        "report_id":    f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": datetime.now().isoformat(),
        "config": {
            "min_sharpe":    min_sharpe,
            "max_dd":        max_dd,
            "max_positions": 6,
        },
        "universe_size":    len(best_by_ticker),
        "qualifying_tops":  len(tops),
        "qualifying_watch": len(watches),
        "portfolio": portfolio,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio allocator — rank + allocate")
    parser.add_argument("--min-sharpe", type=float, default=1.0)
    parser.add_argument("--max-dd",     type=float, default=-0.20)
    parser.add_argument("--max-pos",    type=int,   default=5)
    parser.add_argument("--skip-scan",   action="store_true",
                        help="Skip re-running scanner, use existing results")
    args = parser.parse_args()

    print("\n[Portfolio Allocator]")
    print(f"  Min Sharpe: {args.min_sharpe}  |  Max DD: {args.max_dd}  |  Max positions: {args.max_pos}")

    # Step 1: Load all backtest results
    results = load_all_results()
    print(f"  → Loaded {len(results)} backtest results")

    if not results:
        print("  ⚠️  No results found — run scanner first with --skip-scan")
        sys.exit(1)

    # Step 2: Select best strategy per ticker
    best = select_best_per_ticker(results)
    print(f"  → {len(best)} tickers with qualifying strategies")

    tiers = {"TOP": 0, "WATCH": 0, "SKIP": 0}
    for v in best.values():
        tiers[v["tier"]] += 1
    print(f"     TOP: {tiers['TOP']}  |  WATCH: {tiers['WATCH']}  |  SKIP: {tiers['SKIP']}")

    # Step 3: Build portfolio
    portfolio = allocate_portfolio(best, max_positions=args.max_pos)
    print(render_portfolio(portfolio))

    # Step 4: Save
    report = build_report(portfolio, best, args.min_sharpe, args.max_dd)
    out_path = os.path.join(OUTPUT_DIR, f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  💾 Portfolio saved → {out_path}")
