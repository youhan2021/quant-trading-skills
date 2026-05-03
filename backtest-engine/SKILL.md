---
name: backtest-engine
description: 量化交易回测引擎。根据历史数据对交易策略进行回测，计算收益率、夏普比率、最大回撤等绩效指标，输出详细回测报告。支持多种策略类型和参数扫描。
metadata:
  author: youhan
  version: 0.1.0
  tags: [quant, backtest, strategy, performance]
---

# Backtest Engine

量化交易回测引擎。对历史市场数据执行策略信号模拟，计算完整绩效指标。

## 回测流程

```
历史数据 → 策略信号 → 模拟成交 → 绩效计算 → 报告输出
```

## 支持的策略类型

| 策略 | 描述 | 参数 |
|------|------|------|
| `ma_cross` | 双均线交叉 | fast_period, slow_period |
| `rsi` | RSI 超买超卖 | period, overbought, oversold |
| `macd` | MACD 趋势 | fast, slow, signal |
| `bollinger` | 布林带策略 | period, std_dev |
| `momentum` | 动量策略 | lookback, threshold |
| `mean_reversion` | 均值回归 | period, z_score_threshold |
| `breakout` | 突破策略 | lookback_period |
| `custom` | 自定义策略 | 传入策略逻辑 |

## 绩效指标

### 核心指标

| 指标 | 描述 | 计算方式 |
|------|------|----------|
| `total_return` | 总收益率 | (期末净值-期初净值)/期初 |
| `annualized_return` | 年化收益率 | (1+总收益)^(252/交易日)-1 |
| `sharpe_ratio` | 夏普比率 | (年化收益-无风险利率)/年化波动率 |
| `max_drawdown` | 最大回撤 | max(Peak - Trough)/Peak |
| `calmar_ratio` | 卡玛比率 | 年化收益/最大回撤 |
| `win_rate` | 胜率 | 盈利交易数/总交易数 |
| `profit_factor` | 盈亏比 | 总盈利/总亏损 |
| `sortino_ratio` | 索提诺比率 | (年化收益-目标收益)/下行波动率 |
| ` calmar_ratio` | 卡玛比率 | 年化收益/最大回撤 |

### 交易统计

| 指标 | 描述 |
|------|------|
| `total_trades` | 总交易次数 |
| `avg_trades_per_year` | 年均交易次数 |
| `avg_profit_per_trade` | 单笔平均收益 |
| `avg_holding_period` | 平均持仓周期 |
| `max_consecutive_losses` | 最大连续亏损次数 |

## 输出格式

### 回测结果 JSON

```json
{
  "backtest_id": "bt_20240101_001",
  "strategy": {
    "name": "ma_cross",
    "params": {"fast_period": 5, "slow_period": 20},
    "market": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01"
  },
  "performance": {
    "total_return": 0.2534,
    "annualized_return": 0.2341,
    "sharpe_ratio": 1.85,
    "max_drawdown": -0.1234,
    "calmar_ratio": 1.89,
    "sortino_ratio": 2.41,
    "win_rate": 0.62,
    "profit_factor": 2.15,
    "volatility": 0.1823
  },
  "trades": [
    {
      "trade_id": 1,
      "entry_date": "2023-02-15",
      "exit_date": "2023-03-20",
      "side": "long",
      "entry_price": 152.30,
      "exit_price": 158.75,
      "pnl": 0.0423,
      "holding_days": 33
    }
  ],
  "equity_curve": [
    {"date": "2023-01-01", "equity": 100000},
    {"date": "2023-01-02", "equity": 100250}
  ],
  "drawdown_series": [
    {"date": "2023-05-01", "drawdown": -0.0512}
  ]
}
```

### 回测报告 Markdown

```markdown
# 回测报告: {strategy_name}

## 策略信息
- 标的: {ticker}
- 周期: {start_date} ~ {end_date}
- 策略类型: {strategy_type}

## 绩效摘要
| 指标 | 值 |
|------|-----|
| 总收益率 | {total_return}% |
| 年化收益率 | {annualized_return}% |
| 夏普比率 | {sharpe_ratio} |
| 最大回撤 | {max_drawdown}% |
| 卡玛比率 | {calmar_ratio} |
| 胜率 | {win_rate}% |

## 交易统计
- 总交易次数: {total_trades}
- 年均交易: {avg_trades_per_year}
- 盈亏比: {profit_factor}
```

## 回测引擎函数

### 主函数

```
backtest(strategy_config, price_data, initial_capital=100000)
  → BacktestResult

run_parameter_scan(strategy_type, param_grid, price_data)
  → List[BacktestResult] + best_params
```

### 信号生成

```
generate_signals(strategy_config, price_data)
  → DataFrame: Date, Signal(-1/0/1), Price
```

### 模拟成交

```
simulate_trades(signals, price_data, commission=0.001)
  → List[Trade]
```

### 绩效计算

```
calculate_performance(trades, initial_capital)
  → PerformanceMetrics
```

## 使用示例

### 标准回测流程

```
1. 加载 market-info-collector 获取历史数据
2. 配置策略参数
3. 调用 backtest() 执行回测
4. 输出回测报告
```

### 参数扫描

```
param_grid = {
  "fast_period": [5, 10, 15],
  "slow_period": [20, 30, 50]
}
results = run_parameter_scan("ma_cross", param_grid, price_data)
best = max(results, key=lambda x: x.sharpe_ratio)
```

## 目录结构

```
data/backtest/
├── results/
│   └── {strategy}_{timestamp}.json
└── reports/
    └── {strategy}_{timestamp}.md
```

## 依赖

```bash
pip install pandas numpy
```
