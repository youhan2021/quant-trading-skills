---
name: quant-wfa
description: Quantitative strategy Walk-Forward Analysis — fixed vs rolling retrain, info leak pitfalls, and recommended constrained rolling approach
---

# Quantitative Strategy Walk-Forward Analysis

## Overview
Empirical comparison of fixed strategy vs rolling retrain for quantitative stock selection. Based on 75-strategy comprehensive backtest across 2011-2026 with 28 rolling WFA windows.

## Key Finding: Fixed Strategy ≈ Rolling Retrain

```
固定策略（训一次跑5年） vs 每周滚动重训练
──────────────────────────────────────────
固定 Test Sharpe:   均值=0.90  最大=1.28
WFA Sharpe 均值:    均值=1.51  最大=2.00
WFA Sharpe 标准差:  均值=1.42  最大=1.80
WFA 5%分位数:       平均≈-0.9  ← 最差情况可能为负
WFA通过率:          75/75 (100%)
```

**核心结论：WFA 均值仅比固定高 0.05，但标准差 1.42 意味着策略高度不稳定。**

## WFA vs Fixed Strategy: True Costs and Benefits

### Expanding Window WFA（我们跑的方法）
```
窗口1: Train=2011-2014  Val=2014-2016  → 记Sharpe₁
窗口2: Train=2011-2015  Val=2015-2017  → 记Sharpe₂
...
窗口28: Train=2011-2025  Val=2025-2026  → 记Sharpe₂₈

最终报告: mean=1.33, std=1.42, p5≈-0.9
```
- 无 info leak，每个验证窗口完全 held-out
- 标准差大 = 策略表现高度不稳定

### Fixed Strategy（训一次跑5年）
```
Train: 2011-2016  Val: 2016-2021  Test: 2021-2026
→ Sharpe = 1.28
```
- 计算成本低
- 不需要每周重训练
- 但可能错过 regime 切换

## Subtle Info Leak: Two-Step Select + Validate on Same Data

用户提议的方法：
```
步骤1: Train(A) → Select on B
步骤2: Retrain(A+B) → Validate on B  ← BUG: 已用于选择，不能再验证
```

**问题：B既参与了策略选择，又参与了最终验证，验证 Sharpe 向上偏。**

正确做法：
```
Train(A) → Val(B)  → 报告Sharpe₁
Train(A+B) → Val(C)  → 报告Sharpe₂
Train(A+B+C) → Val(D)  → 报告Sharpe₃
最终报告: mean, std, p5
```

## Recommended Practice: Constrained Rolling Retrain

如果要做 rolling retrain，最小化噪音的方案：

1. **固定因子组合**（如 roc120+vol20），不换因子
2. 每周用最近 24 个月滚动验证这个固定组合的 Sharpe 是否仍为正
3. **只有当 Sharpe 连续 4 周 < 0 时才触发策略切换**
4. 切换时平滑过渡（50%旧仓+50%新仓，给2周缓冲）

这样 rolling retrain 的收益是**减少极端 regime 损失**，而不是提高平均收益。

## Data Files
- `~/.hermes/skills/quant-trading/fundamental-data-collector/data/comprehensive_v2_results.json` — 75策略完整回测结果（train/val/test/WFA/Monte Carlo）
- `~/.hermes/skills/quant-trading/fundamental-data-collector/data/fundamental_data.json` — SEC XBRL 基本面数据

## Scripts
- `comprehensive_v2.py` — 75策略 × 3种组合 × WFA + MC
- `comprehensive_backtest.py` — Daily/Weekly/Monthly 频率对比

## Key Numbers (2026-05-06)
```
固定策略 Test Sharpe:
  Price-roc120+vol20 top5:  1.28 (Ann=34.6%, MaxDD=-37.6%)
  Multi-roc120+book_per_share top10: 1.12 (Ann=18.2%, MaxDD=-26.6%)
  Single-book_per_share top10: 1.10 (Ann=18.8%, MaxDD=-15.4%)
  Single-roe top10: 0.98 (Ann=16.9%, MaxDD=-27.9%)
  SPY B&H: ~0.69 (Ann≈16.7%, MaxDD≈-26%)

WFA Sharpe 分布（28个滚动窗口）:
  Price-roc120+vol20 top5: mean=1.33, std=1.43, p5≈-1.0
  Single-roc120 top5: mean=1.51, std=1.39
```

## Related Skills
- `momentum-strategy` — 动量策略核心逻辑和 cron 周频执行
- `quant-backtest-validation` — info leak 防护和动态组合再平衡
