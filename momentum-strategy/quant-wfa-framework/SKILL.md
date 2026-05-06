---
name: quant-wfa-framework
description: Rolling Walk-Forward Analysis framework for quantitative stock strategy validation. Built through trial-and-error with bugs, data issues, and counter-intuitive findings.
---

# Quant WFA Framework

## Core Architecture: 5-Step WFA

1. **Universe** → Large pool (e.g., SPX 117 stocks)
2. **Basic screen** → ROE>0, D/E<80% (removes zombie companies)
3. **Val IC** → Use 2016-2021 data to determine factor direction and weights
4. **Rolling Test** → Apply LOCKED weights to 2021/2022/2023/2024 each Q1 independently
5. **No adjustment** → Factor weights NEVER change based on Test results

## Key Findings (验证过的经验)

### ⚠️ Critical Bug: NoneType in Fundamentals
```python
# WRONG (crashes when roe is None)
if fund[t].get('roe', -999) > 0:

# CORRECT
if (fund[t].get('roe') or 0) > 0:
```
yfinance fundamentals can return `None` for any field. Always use `or 0` pattern.

### ⚠️ Critical Bug: MultiIndex from yfinance
```python
# auto_adjust=False returns MultiIndex (PriceType, Ticker)
df = yf.download(tickers, auto_adjust=False)
# df['Close'] first to get price layer
df_close = df['Close']  # shape (N, M), columns are tickers
```

### ⚠️ Factor Direction from IC needs quality threshold
```python
# Only determine direction if n >= 30 to avoid sample bias
if n >= 30:
    direction = 1 if mean_ic > 0 else -1
```

### ⚠️ Vol Factor Data Problem
- `vol20` needs 20 weeks of price history — many stocks don't have it
- `vol60` mostly NaN for滚动 window
- Minimum data requirement: n >= 30 before trusting IC

### ⚠️ Static Selection Problem (Discovered 2026-05-06)
When factor weights are LOCKED after Val, the same stocks get selected every year:
```
2021 → ['MU', 'NEM', 'NVDA', 'AMAT', 'RTX']
2022 → ['MU', 'NEM', 'NVDA', 'AMAT', 'RTX']  ← identical!
2023 → ['MU', 'NEM', 'NVDA', 'AMAT', 'RTX']  ← identical!
```
This means IC-based weighting is not truly "rolling" — it's a static filter applied to different universes. True rolling would rebalance based on changing factor relevance.

### ⚠️ Look-Aheck Bias in Fundamentals
Current yfinance fundamentals = most recent reported values, not historical point-in-time values. Using current ROE/D/E in a 2021 test window = look-ahead bias.

### ⚠️ Survivorship Bias in Current Universe
- Current SPX survivors (117 stocks) ≠ historical constituents
- Wikipedia 2011 snapshot (497 stocks including bankrupt companies) gives more honest results

## Results Summary

### SPX 2011 Historical Universe (Wikipedia snapshot, ~497 stocks)
- Val Sharpe = **0.00** for pure momentum strategies
- Test Sharpe = **-4.0**
- Conclusion: Pure momentum **does not work** in honest historical backtest

### Current Universe (117 survivors, yfinance)
- Rolling WFA average Sharpe = **1.31** vs SPY **1.38** (underperforms)
- 2022: nearly matches SPY (-0.73 vs -0.75)
- 2024: significantly underperforms (-0.88 gap)

### IC Statistics (Val 2016-2021, 117-stock survivor)
- roc20: IC=+0.035, frac=60%, n=72 ✓
- roc60: IC=+0.059, frac=69%, n=72 ✓
- roc120: IC=+0.077, frac=67%, n=72 ✓
- vol20: IC=-0.145, but n=8 < 30 ✗

## Framework Files

- `~/.hermes/skills/quant-trading/momentum-strategy/scripts/rolling_wfa.py` — 5-step rolling WFA
- `~/.hermes/skills/quant-trading/momentum-strategy/scripts/clean_backtest.py` — Historical SPX backtest

## Improvement Directions

1. **Rebalance annually**: Not just LOCKED weights, but recalculate IC direction each year from expanding window
2. **Industry diversification**: Cap at 1-2 stocks per GICS sector
3. **Historical SEC XBRL**: Need point-in-time fundamental data instead of current yfinance
4. **Mean-reversion overlay**: Individual stock selection uses reversal signal within momentum basket
