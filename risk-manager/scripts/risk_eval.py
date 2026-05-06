#!/usr/bin/env python3
"""
risk-manager/scripts/risk_eval.py
Portfolio / strategy risk evaluation.

Usage:
    python3 risk_eval.py                                       # eval all results in results dir
    python3 risk_eval.py --result /path/to/bt_result.json
    python3 risk_eval.py --ticker SPY --strategy ma_cross
    python3 risk_eval.py --best-of N  --min-sharpe 1.0 --max-dd -0.20
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from glob import glob

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(SKILL_DIR), "backtest-engine", "data", "results")
OUTPUT_DIR  = os.path.join(SKILL_DIR, "data", "risk")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd


# ── risk metrics ───────────────────────────────────────────────────────────────

def var_historic(returns: list[float], confidence: float = 0.95) -> float:
    """Historic VaR: the loss at the given confidence percentile."""
    if not returns:
        return 0.0
    r = np.array(returns)
    return float(np.percentile(r, (1 - confidence) * 100))


def cvar_historic(returns: list[float], confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall: average loss beyond VaR."""
    if not returns:
        return 0.0
    r = np.array(returns)
    var = var_historic(returns, confidence)
    return float(np.mean(r[r <= var])) if any(r <= var) else var


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly criterion: optimal fraction of capital to bet."""
    if avg_loss == 0 or win_rate >= 1 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1 - win_rate
    f = (b * win_rate - q) / b
    return max(0.0, min(f, 1.0))


def position_size_kelly(capital: float, win_rate: float, avg_win: float,
                        avg_loss: float, max_risk: float = 0.02) -> float:
    """Position size using Kelly, capped at max_risk fraction of capital."""
    kelly = kelly_fraction(win_rate, avg_win, avg_loss)
    # Kelly can be aggressive; use half-Kelly for safety
    fraction = kelly * 0.5
    return round(capital * min(fraction, max_risk), 2)


def assess_strategy_risk(result: dict, confidence: float = 0.95) -> dict:
    """
    Full risk assessment for a backtest result.
    Returns a RiskReport dict.
    """
    trades = result.get("trades", [])
    equity = result.get("equity_curve", [])
    perf   = result.get("performance", {})
    ticker = result.get("ticker", "?")
    strat  = result.get("strategy", "?")
    params = result.get("params", {})

    capital     = result.get("initial_capital", 100_000)
    final_cap   = result.get("final_capital", capital)

    # ── per-trade returns ────────────────────────────────────────────────────
    if trades:
        pnls = [t["pnl"] for t in trades]
        rets = pnls  # already fraction of entry price
    else:
        pnls = []
        rets = []

    # ── VaR / CVaR ───────────────────────────────────────────────────────────
    var_95  = var_historic(rets, 0.95)
    cvar_95 = cvar_historic(rets, 0.95)
    var_99  = var_historic(rets, 0.99)
    cvar_99 = cvar_historic(rets, 0.99)

    # ── portfolio risk from equity curve ──────────────────────────────────────
    eq_vals = [e["equity"] for e in equity]
    eq_rets = []
    for i in range(1, len(eq_vals)):
        if eq_vals[i-1] > 0:
            eq_rets.append((eq_vals[i] - eq_vals[i-1]) / eq_vals[i-1])

    eq_var_95  = var_historic(eq_rets, 0.95) if eq_rets else 0.0
    eq_cvar_95 = cvar_historic(eq_rets, 0.95) if eq_rets else 0.0

    # ── win/loss stats ────────────────────────────────────────────────────────
    if trades:
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        loss = [t["pnl"] for t in trades if t["pnl"] <= 0]
        avg_win_v  = float(np.mean(wins)) if wins else 0.0
        avg_loss_v = abs(float(np.mean(loss))) if loss else 0.0
        wr = len(wins) / len(trades)
    else:
        avg_win_v = avg_loss_v = 0.0
        wr = 0.0

    kelly_frac = kelly_fraction(wr, avg_win_v, avg_loss_v)

    # ── concentration / drawdown stats ───────────────────────────────────────
    peak = eq_vals[0] if eq_vals else capital
    running_peak = peak
    dd_series = []
    for e in eq_vals:
        if e > running_peak:
            running_peak = e
        dd = (e - running_peak) / running_peak
        dd_series.append(dd)
    max_drawdown = min(dd_series) if dd_series else 0.0

    # ── volatility ─────────────────────────────────────────────────────────────
    annual_vol = perf.get("volatility", 0.0)
    if annual_vol == 0 and eq_rets:
        annual_vol = float(np.std(eq_rets) * math.sqrt(252)) if len(eq_rets) > 1 else 0.0

    # ── risk/reward ───────────────────────────────────────────────────────────
    ann_ret  = perf.get("annualized_return", 0.0)
    sharpe   = perf.get("sharpe_ratio", 0.0)
    sortino  = perf.get("sortino_ratio", 0.0)
    calmar   = perf.get("calmar_ratio", 0.0)

    # ── alerts ────────────────────────────────────────────────────────────────
    alerts = []

    # Sharpe below 0.5
    if sharpe < 0.5 and sharpe >= 0:
        alerts.append({
            "level": "warning",
            "type": "low_sharpe",
            "message": f"Sharpe ratio {sharpe:.2f} is below 0.5 — risk-adjusted return is weak",
        })
    elif sharpe < 0:
        alerts.append({
            "level": "critical",
            "type": "negative_sharpe",
            "message": f"Sharpe ratio {sharpe:.2f} is negative — strategy loses money per unit risk",
        })

    # Large drawdown
    if max_drawdown < -0.20:
        alerts.append({
            "level": "critical",
            "type": "large_drawdown",
            "message": f"Max drawdown {max_drawdown:.1%} exceeds -20% — high risk of extended loss",
        })
    elif max_drawdown < -0.10:
        alerts.append({
            "level": "warning",
            "type": "moderate_drawdown",
            "message": f"Max drawdown {max_drawdown:.1%} exceeds -10% — monitor closely",
        })

    # Low win rate
    if trades and wr < 0.40:
        alerts.append({
            "level": "warning",
            "type": "low_win_rate",
            "message": f"Win rate {wr:.1%} is below 40% — requires high profit factor to be viable",
        })

    # Low profit factor
    pf = perf.get("profit_factor", 0)
    if pf > 0 and pf < 1.5:
        alerts.append({
            "level": "info",
            "type": "low_profit_factor",
            "message": f"Profit factor {pf:.2f} is below 1.5 — wins are barely outweighing losses",
        })

    # VaR breach
    if abs(var_95) > 0.05:
        alerts.append({
            "level": "warning",
            "type": "high_var",
            "message": f"Daily VaR(95%) {var_95:.2%} is large — potential for significant daily loss",
        })

    # ── recommendations ──────────────────────────────────────────────────────
    recommendations = []

    if kelly_frac > 0:
        half_kelly = round(kelly_frac * 0.5, 4)
        recommendations.append({
            "action": "position_sizing",
            "details": {
                "kelly_full": round(kelly_frac, 4),
                "kelly_half": half_kelly,
                "max_risk_per_trade": half_kelly,
                "reason": "Kelly criterion suggests optimal position size; use half-Kelly for safety margin",
            },
        })

    if max_drawdown < -0.15 and trades:
        recommendations.append({
            "action": "add_stop_loss",
            "details": {
                "suggested_stop_pct": round(abs(max_drawdown) * 0.5, 4),
                "reason": "Large historical drawdown — stop-loss rule would reduce exposure",
            },
        })

    if ann_vol := annual_vol:
        vol_target = 0.15  # 15% target
        if ann_vol > vol_target:
            recommendations.append({
                "action": "reduce_position",
                "details": {
                    "current_vol": round(ann_vol, 4),
                    "target_vol": vol_target,
                    "scale_factor": round(vol_target / ann_vol, 4),
                    "reason": f"Strategy volatility {ann_vol:.1%} exceeds target {vol_target:.0%}",
                },
            })

    # ── build report ─────────────────────────────────────────────────────────
    report = {
        "risk_report_id": f"risk_{datetime.now().strftime('%Y%m%m_%H%M%S')}",
        "backtest_ref": {
            "ticker":    ticker,
            "strategy":  strat,
            "params":    params,
            "backtest_id": result.get("backtest_id", ""),
        },
        "generated_at": datetime.now().isoformat(),
        "capital": {
            "initial":     capital,
            "final":       round(final_cap, 2),
            "total_return": perf.get("total_return", 0),
        },
        "risk_metrics": {
            "var_95":           round(var_95, 6),
            "cvar_95":         round(cvar_95, 6),
            "var_99":          round(var_99, 6),
            "cvar_99":         round(cvar_99, 6),
            "portfolio_var_95": round(eq_var_95 * final_cap, 2),
            "portfolio_cvar_95": round(eq_cvar_95 * final_cap, 2),
            "max_drawdown":    round(max_drawdown, 6),
            "annual_volatility": round(annual_vol, 6),
            "sharpe_ratio":    sharpe,
            "sortino_ratio":   sortino,
            "calmar_ratio":    calmar,
            "kelly_fraction": round(kelly_frac, 4),
        },
        "trade_risk": {
            "win_rate":       round(wr, 4),
            "avg_win":        round(avg_win_v, 6),
            "avg_loss":       round(avg_loss_v, 6),
            "profit_factor":  perf.get("profit_factor", 0),
            "total_trades":   len(trades),
        },
        "alerts":            alerts,
        "recommendations":   recommendations,
    }

    return report


def save_report(report: dict, prefix: str = "") -> str:
    tag  = f"{report['backtest_ref']['ticker']}_{report['backtest_ref']['strategy']}"
    out  = os.path.join(OUTPUT_DIR, f"{prefix}{tag}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    return out


# ── batch evaluation ──────────────────────────────────────────────────────────

def eval_all_results(results_dir: str = RESULTS_DIR,
                      min_sharpe: float = -999,
                      max_dd: float = 0.0,
                      sort_by: str = "sharpe_ratio") -> list[dict]:
    """Load all backtest results and evaluate risk."""
    files = glob(os.path.join(results_dir, "*.json"))
    # Filter out summary files
    files = [f for f in files if "summary" not in f]

    reports = []
    for fpath in files:
        with open(fpath) as f:
            bt = json.load(f)
        r = assess_strategy_risk(bt)
        # Apply filters
        sm = r["risk_metrics"]["sharpe_ratio"]
        md = r["risk_metrics"]["max_drawdown"]
        if sm < min_sharpe:
            continue
        if md < max_dd:
            continue
        reports.append(r)
        save_report(r)

    reports.sort(key=lambda x: x["risk_metrics"].get(sort_by, 0), reverse=True)
    return reports


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk evaluation for backtest results")
    parser.add_argument("--result", help="Path to a single backtest result JSON")
    parser.add_argument("--ticker", help="Filter by ticker")
    parser.add_argument("--strategy", help="Filter by strategy")
    parser.add_argument("--min-sharpe", type=float, default=-999.0)
    parser.add_argument("--max-dd", type=float, default=-999.0,
                         help="Min (most negative) drawdown allowed, e.g. -0.20")
    parser.add_argument("--sort", default="sharpe_ratio",
                         choices=["sharpe_ratio","sortino_ratio","calmar_ratio",
                                  "var_95","max_drawdown","win_rate"])
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    if args.result:
        with open(args.result) as f:
            bt = json.load(f)
        r = assess_strategy_risk(bt)
        out = save_report(r)
        m = r["risk_metrics"]
        print(f"\nRisk Report: {bt.get('ticker','?')} / {bt.get('strategy','?')}")
        print(f"  Sharpe ratio:       {m['sharpe_ratio']:+.3f}")
        print(f"  Sortino ratio:      {m['sortino_ratio']:+.3f}")
        print(f"  Calmar ratio:       {m['calmar_ratio']:+.3f}")
        print(f"  Max drawdown:       {m['max_drawdown']:+.1%}")
        print(f"  VaR(95%):          {m['var_95']:+.2%}")
        print(f"  CVaR(95%):         {m['cvar_95']:+.2%}")
        print(f"  Portfolio VaR:     ${m['portfolio_var_95']:,.2f}")
        print(f"  Annual vol:         {m['annual_volatility']:+.1%}")
        print(f"  Kelly fraction:     {m['kelly_fraction']:.2%} (half-Kelly: {m['kelly_fraction']*0.5:.2%})")
        print(f"  Win rate:          {r['trade_risk']['win_rate']:.1%}")
        print(f"  Profit factor:     {r['trade_risk']['profit_factor']:.2f}")
        if r["alerts"]:
            print(f"\n  ⚠️  Alerts:")
            for a in r["alerts"]:
                print(f"    [{a['level'].upper()}] {a['message']}")
        if r["recommendations"]:
            print(f"\n  💡 Recommendations:")
            for rec in r["recommendations"]:
                print(f"    • {rec['action']}: {rec['details']}")
        print(f"\nSaved to: {out}")

    else:
        reports = eval_all_results(
            RESULTS_DIR,
            min_sharpe=args.min_sharpe,
            max_dd=args.max_dd,
            sort_by=args.sort,
        )

        if args.ticker:
            reports = [r for r in reports if r["backtest_ref"]["ticker"] == args.ticker]
        if args.strategy:
            reports = [r for r in reports if r["backtest_ref"]["strategy"] == args.strategy]

        top = reports[:args.top]

        print(f"\n{'='*80}")
        print(f"TOP {len(top)} RISK-REPORTED STRATEGIES (sorted by {args.sort})")
        print(f"{'='*80}")
        print(f"{'Ticker':<8} {'Strategy':<18} {'Sharpe':>7} {'Sortino':>8} "
              f"{'MaxDD':>8} {'VaR95':>8} {'WinRate':>8} {'Kelly':>7} {'Alerts':>6}")
        print("-" * 80)
        for r in top:
            m = r["risk_metrics"]
            tr = r["trade_risk"]
            ticker = r["backtest_ref"]["ticker"]
            strat  = r["backtest_ref"]["strategy"]
            n_alerts = len(r["alerts"])
            print(f"{ticker:<8} {strat:<18} {m['sharpe_ratio']:>+7.2f} "
                  f"{m['sortino_ratio']:>+8.2f} {m['max_drawdown']:>+8.1%} "
                  f"{m['var_95']:>+8.2%} {tr['win_rate']:>8.1%} "
                  f"{m['kelly_fraction']:>7.1%} {n_alerts:>6}")

        print(f"\nTotal reports: {len(reports)}")
        print(f"Output dir: {OUTPUT_DIR}")
