---
name: risk-manager
description: 量化交易风险管理模块。计算持仓风险敞口，评估VaR和CVaR，动态调整仓位，执行止损止盈。支持实时风控和回测风控两种模式。
metadata:
  author: youhan
  version: 0.1.0
  tags: [quant, risk, portfolio, position-sizing]
---

# Risk Manager

量化交易风险管理模块。控制仓位、计算风险敞口、执行止损止盈。

## 风险管理框架

```
入场前 → 计算仓位 → 持仓中 → 实时监控 → 出场 → 绩效归因
```

## 风险管理工具

### 仓位管理

| 工具 | 描述 | 公式 |
|------|------|------|
| `fixed_size` | 固定仓位 | 每笔固定金额 |
| `kelly` | Kelly 公式 | f = (bp - q) / b |
| `fixed_fractional` | 固定比例 | 账户 * 固定比例 |
| `volatility_target` | 波动率目标 | 目标波动率 / 资产波动率 |
| `max_loss` | 最大损失法 | (账户 * 最大损失率) / 单笔最大损失 |

### 风险指标

| 指标 | 描述 | 用途 |
|------|------|------|
| `VaR` | Value at Risk | 持仓最大损失（置信水平） |
| `CVaR` | Conditional VaR | 超过 VaR 的平均损失 |
| `volatility` | 年化波动率 | 风险衡量 |
| `beta` | Beta 系数 | 相对市场风险 |
| `correlation` | 相关性 | 组合分散化 |
| `drawdown` | 回撤 | 当前浮亏 |

## 风控函数

### 仓位计算

```
calculate_position_size(account_capital, risk_per_trade, entry_price, stop_loss)
  → position_size

calculate_kelly_fraction(win_rate, avg_win, avg_loss)
  → fraction
```

### 止损止盈

```
set_stop_loss(entry_price, strategy, atr_multiplier=2.0)
  → stop_loss_price

set_take_profit(entry_price, risk_reward_ratio=2.0, stop_loss=None)
  → take_profit_price

check_exit_conditions(position, current_price, stop_loss, take_profit)
  → should_exit: bool, reason: str
```

### 风险评估

```
assess_portfolio_risk(positions, market_data)
  → RiskReport

calculate_var(returns, confidence=0.95)
  → var_value

calculate_cvar(returns, confidence=0.95)
  → cvar_value
```

### 持仓风险

```
calculate_position_risk(position, current_price)
  → PositionRisk {
    exposure: 金额敞口,
    max_loss: 最大损失,
    var: VaR,
    beta: Beta
  }

evaluate_risk_reward(entry, stop, target)
  → {risk: amount, reward: amount, ratio: float}
```

## 风控规则

### 入场前检查

| 检查项 | 阈值 | 动作 |
|--------|------|------|
| 单笔风险 | ≤ 2% 账户 | 拒绝超限仓位 |
| 日累计风险 | ≤ 6% 账户 | 暂停新交易 |
| 组合总敞口 | ≤ 30% 单票 | 分散投资 |
| 相关性 | ≤ 0.7 同向仓位 | 避免过度集中 |

### 持仓中监控

| 检查项 | 阈值 | 动作 |
|--------|------|------|
| 浮动亏损 | ≤ 5% 账户 | 预警 |
| 浮动亏损 | ≤ 8% 账户 | 强制止损 |
| VaR 突破 | > 95% 置信 VaR | 减仓 |
| 波动率异常 | 3x 历史均值 | 预警 |

### 出场规则

| 条件 | 动作 |
|------|------|
| 触及止损 | 立即平仓 |
| 触及止盈 | 部分/全部平仓 |
| 到达目标时间 | 收盘前平仓 |
| 趋势反转 | 跟踪止损 |

## 风控报告

### Risk Report JSON

```json
{
  "risk_report_id": "risk_20240101_001",
  "generated_at": "2024-01-01T16:00:00Z",
  "portfolio": {
    "total_value": 1000000,
    "cash": 300000,
    "positions_value": 700000,
    "exposure": 0.70
  },
  "risk_metrics": {
    "portfolio_var": 28000,
    "portfolio_cvar": 42000,
    "portfolio_volatility": 0.1523,
    "max_drawdown": -0.0834
  },
  "positions": [
    {
      "ticker": "AAPL",
      "quantity": 1000,
      "entry_price": 175.00,
      "current_price": 178.50,
      "unrealized_pnl": 3500,
      "exposure": 0.1785,
      "position_var": 5340
    }
  ],
  "alerts": [
    {
      "level": "warning",
      "type": "concentration",
      "message": "AAPL 仓位占比 17.85% 超过目标 15%"
    }
  ],
  "recommendations": [
    {
      "action": "reduce",
      "ticker": "AAPL",
      "quantity": 100,
      "reason": "降低集中度风险"
    }
  ]
}
```

## 风险模式

### 实盘风控（实时）

```
实时监控：
- 持仓价值变化
- 触发止损止盈
- 组合风险指标
- 异常波动预警

执行动作：
- 发送告警
- 自动平仓（可选）
```

### 回测风控（模拟）

```
回测时加入风控层：
- 每根 K 线检查止损
- 计算未实现盈亏
- 记录最大回撤
- 模拟滑点和佣金
```

## 使用示例

### 实盘风控流程

```
1. 加载持仓数据
2. 获取实时行情
3. 评估风险
4. 检查告警
5. 执行建议或人工确认
```

### 回测风控流程

```
1. 设置风控参数
2. 回测引擎调用风控模块
3. 每根 K 线执行风控检查
4. 记录风控事件
5. 输出带风控的回测结果
```

## 目录结构

```
data/risk/
├── reports/
│   └── {date}_risk_report.json
├── alerts/
│   └── {date}_alerts.json
└── logs/
    └── {date}_risk_log.csv
```

## 扩展

### 自定义风控规则

```python
class RiskRule:
  def evaluate(self, position, market_data) -> RuleResult:
    raise NotImplementedError

  def get_name(self) -> str:
    raise NotImplementedError
```

### 规则注册

```
系统支持注册自定义风控规则
规则按优先级执行
触发后记录日志并告警
```

## 依赖

```bash
pip install pandas numpy
```
