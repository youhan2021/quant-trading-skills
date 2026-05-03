---
name: strategy-generator
description: 量化交易策略生成模块。根据市场数据和技术指标生成交易信号。支持双均线、RSI、MACD、布林带、动量、均值回归等多种内置策略，也支持用户自定义策略逻辑。
metadata:
  author: youhan
  version: 0.1.0
  tags: [quant, strategy, signals, trading]
---

# Strategy Generator

量化交易策略生成器。根据市场数据和技术指标生成交易信号。

## 内置策略

### 趋势跟踪类

| 策略 | 描述 | 逻辑 |
|------|------|------|
| `ma_cross` | 均线交叉 | 快线从上穿越慢线→买入，反之→卖出 |
| `ema_cross` | 指数均线交叉 | 同上，用 EMA |
| `macd` | MACD 趋势 | MACD>0 金叉买入，<0 死叉卖出 |
| `momentum` | 动量策略 | 过去 N 日收益率 > 阈值 → 买入 |
| `breakout` | 突破策略 | 价格突破 N 日高点 → 买入 |

### 均值回归类

| 策略 | 描述 | 逻辑 |
|------|------|------|
| `rsi` | RSI 超买超卖 | RSI<30 买入，>70 卖出 |
| `bollinger` | 布林带策略 | 价格触及下轨买入，上轨卖出 |
| `mean_reversion` | 均值回归 | 价格偏离均线 N 个标准差 → 回归 |
| `kdj` | KDJ 随机 | K<20 金叉买入，>80 死叉卖出 |

### 套利类

| 策略 | 描述 | 逻辑 |
|------|------|------|
| `pairs_trading` | 配对交易 | 两资产价差偏离均值 → 收敛交易 |
| `index_arbitrage` | 指数套利 | 期现价差 > 成本 → 套利 |

## 信号格式

### Signal DataFrame

```python
# columns: Date, Open, High, Low, Close, Volume, Signal
# Signal: 1 (买入), -1 (卖出), 0 (持有)
```

### 信号输出 JSON

```json
{
  "signal_id": "sig_20240101_001",
  "strategy": "ma_cross",
  "generated_at": "2024-01-01T10:00:00Z",
  "signals": [
    {
      "date": "2024-01-01",
      "ticker": "AAPL",
      "signal": 1,
      "strength": 0.85,
      "price": 185.50,
      "reason": "MA5 crosses above MA20"
    }
  ]
}
```

## 策略配置

### 策略配置格式

```json
{
  "strategy_name": "ma_cross",
  "params": {
    "fast_period": 5,
    "slow_period": 20,
    "market": "AAPL"
  },
  "entry_rules": {...},
  "exit_rules": {...},
  "filters": [...]
}
```

### 参数说明

| 参数 | 适用策略 | 描述 |
|------|----------|------|
| `fast_period` | ma_cross, ema_cross | 快线周期 |
| `slow_period` | ma_cross, ema_cross | 慢线周期 |
| `period` | rsi, bollinger, kdj | 指标周期 |
| `overbought` | rsi | 超买阈值 |
| `oversold` | rsi | 超卖阈值 |
| `std_dev` | bollinger | 标准差倍数 |
| `lookback` | momentum, breakout | 回溯周期 |
| `threshold` | momentum | 动量阈值 |

## 技术指标计算

### 内置指标

```
SMA(data, period) → Series
EMA(data, period) → Series
RSI(data, period) → Series
MACD(data, fast, slow, signal) → DataFrame
BOLL(data, period, std_dev) → DataFrame
KDJ(high, low, close, period) → DataFrame
MOM(data, period) → Series
```

## 策略生成函数

### 主函数

```
generate_signal(strategy_config, price_data)
  → SignalResult

generate_multi_signal(strategy_list, price_data)
  → List[SignalResult]
```

### 指标计算

```
calculate_indicators(price_data, indicators)
  → DataFrame with indicators
```

### 信号过滤

```
apply_filters(signals, filter_config)
  → FilteredSignals

# 可用过滤器:
# - volume_filter: 成交量放大确认
# - trend_filter: 顺应主趋势
# - volatility_filter: 波动率过滤
```

## 使用示例

### 生成单策略信号

```python
config = {
  "strategy_name": "ma_cross",
  "params": {"fast_period": 5, "slow_period": 20}
}
result = generate_signal(config, price_data)
```

### 生成多策略信号

```python
strategies = ["ma_cross", "rsi", "macd"]
results = generate_multi_signal(strategies, price_data)
```

### 带过滤器

```python
filters = {
  "volume_filter": {"enabled": True, "threshold": 1.5},
  "trend_filter": {"enabled": True, "ma_period": 200}
}
result = apply_filters(base_signals, filters)
```

## 目录结构

```
data/strategies/
├── signals/
│   └── {ticker}_{date}.json
├── configs/
│   └── {strategy_name}.json
└── indicators/
    └── {ticker}_{date}.json
```

## 策略开发指南

### 添加新策略

1. 在策略注册表注册策略名称
2. 实现信号生成函数
3. 定义参数 schema
4. 添加回测支持
5. 更新文档

### 策略优先级

当多策略信号冲突时：
- 趋势跟踪 > 均值回归（趋势明确时）
- 均值回归 > 趋势跟踪（震荡市场中）
- 可配置优先级权重

## 扩展

### 自定义策略接口

```python
class BaseStrategy:
  def generate(self, price_data: DataFrame) -> DataFrame:
    """生成信号，必须返回包含 Signal 列的 DataFrame"""
    raise NotImplementedError

  def get_params(self) -> dict:
    """返回策略参数"""
    raise NotImplementedError
```

### 策略注册

```
用户可注册自定义策略类
系统自动计算指标并调用策略
```

## 依赖

```bash
pip install pandas numpy
```
