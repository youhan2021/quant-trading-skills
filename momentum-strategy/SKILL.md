---
name: quant-trading-momentum-archive
description: 【已归档】动量策略历史研究记录 — 所有结论已整合到父级 SKILL.md
tags: []
related_skills: []
required_environment_variables: []
required_commands: []
missing_required_environment_variables: []
missing_required_commands: []
setup_needed: false
setup_skipped: false
readiness_status: deprecated
---

# momentum-strategy（已归档）

> ⚠️ **已归档**：所有结论已整合到 `quant-trading/SKILL.md`。本文档保留历史研究过程。

## 归档时间

2026-05-06

## 主要贡献（已迁移）

### Bug 修复
- 发现并修复了 Look-Ahead Return Matrix bug（因子 j + 收益 j overlap）
- 发现 numpy array truthiness 问题
- 发现 monthly_rets.dropna() 删除过多行的问题

### 证伪结论（已保留在父级 SKILL.md）
- 11 只"优选股"Sharpe 7.79 → Selection Bias，真实约 1.16
- Val Sharpe>0 筛选 corr = -0.173，无预测力
- yfinance .info 基本面数据有 look-ahead bias
- Volatility Scaling 不推荐

## 文件

```
scripts/momentum_backtest.py   # 原始 MomentumBacktest 类
```

## 参考

- 完整结论：``quant-trading/SKILL.md``
