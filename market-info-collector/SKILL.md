---
name: market-info-collector
description: 收集股票、外汇、大宗商品等金融市场的实时和历史数据。包括价格、成交量、技术指标、基本面数据、新闻事件等。支持多种数据源，输出结构化数据供量化策略使用。
metadata:
  author: youhan
  version: 0.1.0
  tags: [quant, trading, data, market-data]
---

# Market Info Collector

金融市场经济数据收集模块。负责从各种数据源获取市场信息并结构化输出。

## 数据类型

| 类型 | 内容 |
|------|------|
| `price` | 实时/历史价格 OHLCV |
| `fundamental` | 财报、估值、宏观数据 |
| `news` | 新闻、公告、事件 |
| `macro` | 利率、CPI、GDP 等宏观指标 |
| `sentiment` | 市场情绪、资金流向 |

## 数据源

### 免费数据源

| 数据源 | 类型 | API |
|--------|------|-----|
| Yahoo Finance (yfinance) | 股票/指数/外汇 | pip install yfinance |
| Alpha Vantage | 股票/外汇/加密 | 免费 key |
| FRED | 宏观经济 | fredapi |
| Tushare | A股 | tu需要 token |
| AKShare | A股/期货/基金 | akshare |

### 数据获取函数

```
collect_price(ticker, start_date, end_date, interval="1d")
  → DataFrame: Date, Open, High, Low, Close, Volume

collect_fundamental(ticker, report_type="quarterly")
  → DataFrame: Revenue, Profit, EPS, PE, PB, etc.

collect_macro(indicator, country="US")
  → Series: Date, Value

collect_news(ticker, days=7)
  → DataFrame: Date, Headline, Source, Sentiment

collect_market_sentiment()
  → Dict: VIX, PutCallRatio, MarketBreadth, MoneyFlow
```

## 输出格式

所有数据输出为标准化 JSON 格式，存储在 `data/market/` 目录：

```
data/market/
├── prices/
│   ├── {ticker}_{date}.json
├── fundamentals/
│   ├── {ticker}_{quarter}.json
├── news/
│   ├── {ticker}_{date}.json
└── macro/
    └── {indicator}_{date}.json
```

## 使用方式

```
当需要获取市场数据时，加载此 skill。
根据任务选择合适的数据源函数。
数据自动保存到 data/market/ 目录。
```

## 依赖

```bash
pip install yfinance akshare pandas fredapi
```
