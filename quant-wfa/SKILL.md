---
name: quant-wfa-archive
description: 【已归档】Walk-Forward Analysis 方法论 — 已整合到父级 SKILL.md
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

# quant-wfa（已归档）

> ⚠️ **已归档**：WFA 方法论已整合到 `quant-trading/SKILL.md`。

## 归档时间

2026-05-06

## 核心方法论（来自 zachisit/july-backtester）

### Walk-Forward Analysis
- 测试期 80/20 分割：前 80% 是 IS，后 20% 是 OOS
- Verdict 标准：
  - Sign flip（IS>0, OOS<0）→ "Likely Overfitted"
  - OOS 年化比 IS 差 >75% → "Likely Overfitted"
  - 否则 → "Pass"

### Monte Carlo Block-Bootstrap
- 保留交易序列自相关性（block resampling，block = √N）
- 3维评分：Performance Robustness / Drawdown Realism / Tail Risk
- Score ≥4: Robust; Score ≤0: High Risk

## 本目录无独立脚本

WFA 逻辑已在 `fixed_weekly_backtest.py` 的年度分解中直接实现。

## 参考

- 完整结论：``quant-trading/SKILL.md``
