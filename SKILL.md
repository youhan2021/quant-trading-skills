---
name: quant-trading
description: 量化交易系统 — 固定动量策略 + 严格三段式回测，无信息泄漏。
tags: []
related_skills: []
required_environment_variables: []
required_commands: []
missing_required_environment_variables: []
missing_credential_files: []
missing_required_commands: []
setup_needed: false
setup_skipped: false
readiness_status: available
---

# quant-trading

量化交易系统 — 固定动量策略 + 严格三段式回测，无信息泄漏。

## 核心成果（2026-05-06 最终版）

### 当前最优策略：固定 Weekly Momentum

**策略规则（无需优化，直接可用）：**
- **Universe**: 13 只股票 — SPY, QQQ, GLD, TLT, EFA, AMZN, LMT, V, NVDA, WMT, GOOGL, UUP, JPM
- **因子**: roc120 + vol20（rank-zscore 等权组合）
- **持仓**: top5，等权
- **频率**: 周频（每周最后一个交易日算因子 → 下周持仓）
- **收益计算**: `(C[t+1] - C[t]) / C[t]`，因子在 t 用 t 日收盘（含全量历史）
- **换手**: 平均每周 11.5%，年化约 6x

**回测结果（真实滚动模拟，无任何参数优化）：**

| 指标 | 无TC | 10bps | 20bps | 30bps |
|------|------|-------|-------|-------|
| Sharpe | **1.25** | 1.19 | 1.13 | 1.08 |
| 年化收益 | 26.4% | 25.2% | 24.0% | 22.8% |
| MaxDD | -44.0% | | | |

- **SPY B&H**: Sharpe 0.77，年化 12.5%
- **最终净值**: 21.83x（vs SPY 5.44x）
- **胜率**: 11/13 年正收益（2018 和 2022 亏损）

**年度明细（无TC）：**

| 年份 | 收益 | Sharpe | MaxDD | 周换手 |
|------|------|--------|-------|--------|
| 2013 | +43.0% | 2.98 | -6.4% | 13.0% |
| 2014 | +2.1% | 0.14 | -11.0% | 6.4% |
| 2015 | +40.6% | 2.11 | -6.7% | 8.2% |
| 2016 | +33.7% | 1.65 | -8.6% | 12.8% |
| 2017 | +42.8% | 3.35 | -3.7% | 5.7% |
| 2018 | -6.6% | -0.26 | -29.7% | 8.3% |
| 2019 | +31.8% | 2.17 | -10.0% | 16.7% |
| 2020 | +37.8% | 1.20 | -28.7% | 12.0% |
| 2021 | +36.0% | 1.89 | -6.9% | 7.1% |
| 2022 | -36.2% | -1.15 | -38.8% | 11.5% |
| 2023 | +41.2% | 2.73 | -7.8% | 21.0% |
| 2024 | +47.7% | 2.62 | -10.2% | 20.0% |
| 2025 | +29.2% | 1.20 | -24.3% | 8.3% |

**交易成本估算：**
- Annual TC = `2 × tc_one_way × avg_weekly_turnover × 52`
- 周换手约 11.5%，20bps 单程下年 TC ≈ 2.4%，对收益影响约 9%

### 结论：没有圣杯，没有 Regime Switch

| 对比项 | 固定策略 | RegimeSwitch | 结论 |
|--------|---------|--------------|------|
| WFA Sharpe（12年平均） | **1.50** | 1.44 | 固定胜 |
| B 跑赢 A 的年份 | — | 4/12 | 固定胜 |
| MaxDD 控制 | -44.0% | — | 固定胜 |
| 复杂 度 | 低 | 高 | 固定胜 |

Regime Switch 在 WFA 验证中不 work（固定胜 4/12 年），且 2022 年债券也跌（加息环境），防御仓位反而更差。

## 三段式分离原则（防 Info Leak）

| 阶段 | 数据 | 用途 |
|------|------|------|
| Train | 2011-2016 | 因子 IC 方向 |
| Val | 2016-2021 | 策略筛选（已证伪：Val-Test 相关性 = -0.173） |
| Test | 2021-2026 | 最终评估，只跑一次 |

**重要发现：Val Sharpe>0 筛选对 Test 表现几乎无预测能力（corr = -0.173），该条件本质是过拟合源。**

## 已证伪的策略

1. **Regime Switch（股票/债券切换）**: WFA 12 年平均 Sharpe 1.44 < 固定 1.50，B 只赢 4/12 年
2. **11 只"优选股"**: Sharpe 7.79 → Selection Bias + Info Leak，真实约 1.16
3. **yfinance .info 基本面数据**: Look-ahead bias，最新季度数据含未来信息
4. **Val Sharpe>0 筛选**: 预测能力几乎为零
5. **Volatility Scaling**: 降低收益或增加 MaxDD，不推荐

## 关键 Bug 记录

### Look-Ahead Return Matrix（已修复）
```
错误: 因子在 j 计算，收益也用 j 月收益（overlap 20天）
修复: 因子在 j → 收益从 j 到 j+1（次月）
影响: Sharpe 7.79 → 1.16
```

### numpy array truthiness（已修复）
```
错误: if not rets:  → numpy array 判断歧义
修复: if len(rets) == 0:
```

## 文件结构

```
quant-trading/
├── SKILL.md                    # 本文件
├── README.md                    # repo 说明
├── fixed_weekly_backtest.py     # 核心回测脚本（固定策略，真实滚动）
├── momentum-strategy/
│   └── SKILL.md                # 历史策略研究记录
├── fundamental-data-collector/
│   └── scripts/
│       ├── rolling_retrain_weekly.py  # RegimeSwitch 实验脚本
│       ├── sec_xbrl_fetch.py          # SEC XBRL 数据获取（备用）
│       └── comprehensive_backtest.py   # 三段式综合实验
├── quant-wfa/
│   └── SKILL.md                # WFA 方法论
└── data/                        # 本地数据（不 push）
```

## 操作节奏（cron 配置）

```
Signal：每周最后一个交易日收盘 → 算 roc120+vol20 → 预设 top5 清单
执行：下一个交易日开盘 → 市价单买入
持仓：本周全程不动
重复：每周末重复
```

**无参数可调，无需干预。**

## 参考来源

- **Walk-Forward Analysis + Monte Carlo**: [zachisit/july-backtester](https://github.com/zachisit/july-backtester)
- **Volatility Scaling**: `/tmp/volatility-scaled-momentum-mean-reversion-strategy`
- **HMM Regime Detection**: [Abdullah-BA/RegimeSwitchingMomentumStrategy](https://github.com/Abdullah-BA/RegimeSwitchingMomentumStrategy)
