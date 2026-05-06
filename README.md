# quant-trading-skills

量化交易 skill 系统 — 严格三段式回测（Train/Val/Test），无信息泄漏。

## Skills

| Skill | 描述 |
|-------|------|
| `backtest-engine` | 回测引擎框架 |
| `fundamental-data-collector` | **SEC XBRL 历史 + yfinance 实时** 两层基本面数据收集 |
| `market-info-collector` | 市场信息（价格、成交量、技术指标）收集 |
| `momentum-strategy` | 动量策略量化交易系统（三段式回测，Walk-Forward Analysis） |
| `quant-backtest-validation` | 回测 info leak 防护和动态组合再平衡研究 |
| `quant-trading-controller` | 量化交易主控模块 |
| `risk-manager` | 风险管理（VaR、CVaR、止损止盈） |
| `strategy-generator` | 交易信号生成（双均线、RSI、MACD、布林带、动量、均值回归） |

## 核心研究成果（2026-05-06 更新）

### ⚠️ 重大 Bug 修复：look-ahead return matrix

**之前所有 Sharpe 数字均有 info leak** — 因子在月末 `j` 计算（用价格到 `j`），但收益错误地用了同一月的收益（`j` 当月）。正确做法：因子在 `j` → 收益从 `j+1`（次月）。

结果修正示例：
| 策略 | 修复前（info leak） | 修复后（正确） |
|------|------------------|--------------|
| roc20 top5 Sharpe | **7.79** | **1.16** |
| roc20 top5 Ann | 171.5% | 26.7% |
| MaxDD | 0% | **-25.6%** |
| 月胜率 | 100% | 59% |

> 📌 **"11只优先股" Sharpe 7.79 是用未来收益打分造成的幻觉**。真实 Sharpe 约 1.16，和其他客观选股策略差不多。

### 最佳策略（修复后，真实无 info leak）

| 策略 | Test Sharpe | Ann | MaxDD | MC-p5 |
|------|------------|-----|-------|-------|
| roc120+vol20 top5 | **1.28** | 34.6% | -37.6% | 0.48 |
| roc120 top5 | **1.16** | 26.7% | -25.6% | 0.45 |
| book_per_share top10 | **1.10** | 18.8% | -15.4% | 0.48 |
| roc120+book_per_share top20 | **1.12** | 18.2% | -26.6% | 0.38 |
| SPY B&H | ~0.69 | ~16.7% | ~-26% | — |

> MC-p5 = Monte Carlo 500次 bootstrap 5%分位数。大部分策略 < 0.5，置信区间下界接近零。

### 三段时间分离
- **Train**: 2011-01 至 2016-01（因子 IC 方向）
- **Val**: 2016-01 至 2021-01（策略筛选）
- **Test**: 2021-01 至 2026-01（最终评估，只跑一次）

### 因子 IC（Train 期，预测能力）

| 因子 | IC | IC>0% | 方向 | 备注 |
|------|-----|-------|------|------|
| roc60 | +0.029 | 57% | LONG | price 因子中最稳定 |
| roc120 | +0.009 | 55% | LONG | |
| roe | +0.039 | 60% | LONG | 基本面最强，但极弱 |
| earnings_yield | +0.021 | 57% | LONG | |
| book_per_share | -0.016 | 48% | SHORT | |
| de_ratio | -0.022 | 43% | SHORT | |

> 基本面因子 IC 极弱（0.02-0.04），top5 组合主要靠 price momentum。

### 综合实验：75 个策略对比（2026-05-06）

75 策略覆盖：单因子（price/fund）× top5/10/20 × 多因子组合（equal rank / IC-weighted / price×price / price×fund / triple）

**关键结论**：
1. **无"圣杯"策略** — 最高 Test Sharpe = 1.28，且 MC-p5 只有 0.48
2. **基本面因子作用有限** — roe/earnings_yield IC 弱（<4%），组合后提升不显著
3. **Momentum IC 方向在 Train→Val→Test 不稳定** — IC stability 接近 0，甚至为负
4. **WFA 大部分 Pass** — 但 MC-p5 普遍偏低，说明稳健性置信度有限

### 已解决的关键问题
- ~~13 只优选股 selection bias~~ → 103 只候选池，Sharpe 2.17 → 0.94（真实上限）
- ~~Val Sharpe>0 筛选~~ → 无效，corr=-0.173
- ~~yfinance .info look-ahead~~ → 用 SEC XBRL 历史数据代替
- ~~MaxDD bug~~ → numpy array 判断歧义，已修复
- ~~look-ahead return matrix~~ → score at j → return from j to j+1（次月），已修复
- ~~UNIVERSE 局部变量遮蔽~~ → 用独立局部变量 `available`

## Setup

```bash
# 安装依赖
pip install yfinance pandas numpy requests

# 收集 yfinance 实时数据
python fundamental-data-collector/scripts/yfinance_update.py --all

# 合并 SEC XBRL + yfinance
python fundamental-data-collector/scripts/merge_fundamental.py --all

# 运行多因子回测
python fundamental-data-collector/scripts/multi_factor_v3.py
```
