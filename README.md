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

## 核心研究成果（2026-05-06）

### 最佳策略：vol20 + ROE 多因子
```
Strategy              Sharpe  Ann%   MaxDD%
───────────────────────────────────────────
Multi-vol20+roe top5  1.42   +48.2%  -18.4%  ← 最佳
Multi-vol20+earnings  1.37  +47.0%  -18.4%
Baseline vol20 top10   1.31  +35.0%  -17.6%
SPY B&H               0.97  +14.6%  -23.9%
```

### 三段时间分离
- **Train**: 2011-01 至 2016-01（策略发现）
- **Val**: 2016-01 至 2021-01（股票筛选 & 因子方向）
- **Test**: 2021-01 至 2026-01（最终评估）

### 数据架构
```
~/.hermes/fundamental_data/       ← 本地数据目录
  sec_xbrl/                       ← SEC XBRL 历史（107 只，最早 2005）
  yfinance/                       ← yfinance 实时更新
  merged/                         ← 合并后数据
```

### 因子 IC 稳定性（Val 期）
```
因子       Val Mean IC  IC>0%   Stability
─────────────────────────────────────────
roc120     +0.151      66.7%    0.600  ✓
vol60       +0.108      66.1%    0.455  ✓
vol20       +0.105      62.7%    0.389  ✓
roe        -0.043       42.4%   -0.241  ✗ 反转
earnings   -0.040       42.4%   -0.232  ✗ 反转
```

### 已解决的关键问题
- ~~13 只优选股 selection bias~~ → 103 只候选池，Sharpe 2.17 → 0.94（真实上限）
- ~~Val Sharpe>0 筛选~~ → 无效，corr=-0.173
- ~~yfinance .info look-ahead~~ → 用 SEC XBRL 历史数据代替
- ~~MaxDD bug~~ → numpy array 判断歧义，已修复
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
