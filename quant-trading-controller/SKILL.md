---
name: quant-trading-controller
description: 量化交易主控模块。协调市场信息收集、策略生成、信号执行、风险管理等各子模块。接收用户指令（如"分析苹果股票"、"回测双均线策略"），调度子skill完成任务，结果汇总输出。
metadata:
  author: youhan
  version: 0.1.0
  tags: [quant, trading, controller, orchestration]
---

# Quant Trading Controller

量化交易系统的中央协调模块。通过调度子模块完成完整的量化分析流程。

## 模块架构

```
quant-trading-controller (主控)
├── market-info-collector (数据层)
├── strategy-generator (策略生成)
├── backtest-engine (回测引擎)
├── risk-manager (风险管理)
└── execution-agent (执行代理) [预留]
```

## 工作流程

```
用户指令 → 主控解析 → 调度子skill → 汇总结果 → 输出报告
```

### 标准任务流程

| 步骤 | 调用 | 输出 |
|------|------|------|
| 1 | `market-info-collector` | 原始市场数据 |
| 2 | `strategy-generator` | 策略逻辑/信号 |
| 3 | `backtest-engine` | 回测报告 |
| 4 | `risk-manager` | 风险评估 |
| 5 | 汇总 | 最终建议 |

## 指令解析

### 支持的指令类型

| 指令 | 描述 | 示例 |
|------|------|------|
| `analyze` | 分析标的 | "分析苹果股票" |
| `backtest` | 回测策略 | "回测双均线策略" |
| `scan` | 扫描机会 | "扫描A股低估股票" |
| `monitor` | 监控信号 | "监控科技股动量" |
| `report` | 生成报告 | "生成月度报告" |

### 指令格式

```
{action}:{target}:{params}
analyze:AAPL:technicals,fundamentals
backtest:ma_cross:fast=5,slow=20,market=US
scan:china:aio,pe<20,roe>15
```

## 子skill调用

### 加载 market-info-collector

```
加载 market-info-collector skill 获取市场数据
```

### 调用数据收集

```python
# 收集价格数据
collect_price(ticker="AAPL", start="2024-01-01", interval="1d")

# 收集基本面
collect_fundamental(ticker="AAPL", report_type="quarterly")

# 收集新闻
collect_news(ticker="AAPL", days=7)
```

## 输出结构

所有任务输出到 `output/` 目录：

```
output/
├── analysis/
│   └── {ticker}_{timestamp}.json
├── backtest/
│   └── {strategy}_{timestamp}.json
└── reports/
    └── {report_type}_{timestamp}.md
```

## 状态管理

### 任务状态

| 状态 | 描述 |
|------|------|
| `pending` | 等待执行 |
| `running` | 执行中 |
| `completed` | 完成 |
| `failed` | 失败 |

### 任务记录

```json
{
  "task_id": "task_20240101_001",
  "type": "analyze",
  "target": "AAPL",
  "status": "completed",
  "subtasks": [...],
  "result": {...},
  "created_at": "2024-01-01T10:00:00Z"
}
```

## 与其他子skill的协作

### market-info-collector

数据收集模块。主控需要数据时调用其接口。

```
需要：市场数据
调用：load skill market-info-collector
获取：DataFrame/JSON 格式数据
```

### 子模块协作

已实现的子模块（通过 skill 调度）：

```
market-info-collector: ✅ 可用（数据收集）
strategy-generator:    ✅ 可用（信号生成）
backtest-engine:        ✅ 可用（回测引擎）
risk-manager:          ✅ 可用（风险管理）
execution-agent:       🔜 待开发（执行代理）
```

## 使用示例

### 分析单只股票

```
用户: 分析苹果股票
主控: 
  1. 加载 market-info-collector
  2. 获取 AAPL 价格数据 (1年)
  3. 获取基本面数据
  4. 计算技术指标
  5. 获取近期新闻
  6. 生成分析报告
```

### 回测策略

```
用户: 回测双均线策略
主控:
  1. 解析策略参数 fast=5, slow=20
  2. 加载 market-info-collector 获取历史数据
  3. 生成策略信号
  4. 执行回测
  5. 输出绩效指标
```

## 扩展

新增子模块时，在 SKILL.md 中添加模块说明，并在主控逻辑中注册其调用接口。
