---
name: fundamental-data-collector
description: 收集股票基本面数据的两层架构 — SEC XBRL 历史 + yfinance 实时更新
trigger: "fundamental data collection, XBRL, SEC data, yfinance基本面"
---

# Fundamental Data Collector Skill

## Overview
收集股票基本面数据的两层架构：
- **历史层**：SEC EDGAR XBRL JSON API（完整历史，最早到2005，无 look-ahead）
- **更新层**：yfinance（实时更新）

数据存储在 skill 的 `data/` 子目录下（SEC XBRL 原始数据需单独收集到 `data/sec_xbrl/`，yfinance 数据到 `data/yfinance/`，合并数据到 `data/merged/`）。

## Directory Structure
```
~/.hermes/fundamental_data/
  meta.json                    # 全局元数据（最后更新时间、版本）
  tickers.json                 # 股票列表和 CIK 映射
  sec_xbrl/                    # SEC XBRL 历史数据 (AAPL.json, MSFT.json, ...)
  yfinance/                    # yfinance 实时数据
  merged/                      # 合并后的最终数据
  logs/                        # fetch 日志
```

## Data Fields (Merged Output)
```json
{
  "ticker": "AAPL",
  "cik": "0000320193",
  "last_updated": "2026-05-06",
  "data_source": "sec_xbrl+yfinance",
  "shares_outstanding": {"current": 15728960000, "source": "yfinance"},
  "income_statement": {
    "revenues": [{"end": "2024-09-28", "filed": "2024-10-31", "val": 391035000000}],
    "net_income_loss": [...]
  },
  "balance_sheet": {"stockholders_equity": [...]},
  "ratios": {"roe": [{"date": "2024-09-28", "val": 1.52}]}
}
```

## CIK Lookup (Priority Order)
1. `https://www.sec.gov/files/company_tickers.json` — SEC 官方，206KB，秒回
2. `https://efts.sec.gov/LATEST/search-index?q={ticker}` — search-index fallback
3. Hardcoded fallback（见下文）

## SEC XBRL API
- **Endpoint**: `https://data.sec.gov/api/xbrl/companyfacts/CIK{ZERO_PADDED_10}.json`
- **CIK 格式**: 必须 zero-padded 到 10 位，如 `0000320193`
- **Rate limit**: 0.1s between requests, 429 时等待 5s 重试（最多 3 次）
- **User-Agent**: `HermesBot/1.0 (fundamental data collection)`

### Look-Ahead Protection
```python
# 只保留 filed_date ≤ effective_date 的数据
FILING_LAG_DAYS = 60
effective_date = 回测截面日期（每月末）
```

### Filtering Rules
1. 只保留 `unit == "USD"` 的记录
2. 排除有 Dimensions 的复合指标（保留纯量）
3. 目标 concepts:
   - Income: `Revenues`, `NetIncomeLoss`, `GrossProfit`, `OperatingIncome`
   - Balance: `StockholdersEquity`, `Assets`, `Liabilities`
   - Shares: `WeightedAverageSharesOutstandingDiluted`

## Hardcoded CIK Fallback
```json
{
  "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044",
  "AMZN": "0001018724", "NVDA": "0001045810", "META": "0001326801",
  "TSLA": "0001318605", "BRK": "0001067983", "JPM": "0000019617",
  "V": "0001362108", "JNJ": "0000200406", "WMT": "0000104169",
  "PG": "0000080424", "MA": "0001141391", "UNH": "0000731766",
  "HD": "0000035495", "DIS": "0001744486", "PYPL": "0001585185",
  "NFLX": "0001065280", "ADBE": "0000796343", "CRM": "0001108524",
  "INTC": "0000050863", "AMD": "0000002488", "QCOM": "0000806328"
}
```

## Scripts (in `scripts/`)

### `sec_xbrl_fetch.py` — SEC XBRL Historical Fetch
```bash
python scripts/sec_xbrl_fetch.py --tickers AAPL,MSFT,GOOGL --output ~/.hermes/fundamental_data/sec_xbrl/
python scripts/sec_xbrl_fetch.py --all --tickers-file ~/.hermes/fundamental_data/tickers.json --output ~/.hermes/fundamental_data/sec_xbrl/
```
- CIK lookup: `https://www.sec.gov/files/company_tickers.json` → hardcoded fallback
- 请求 `companyfacts/CIK{ZERO_PADDED_10}.json`，解析 `data.{concept}.USD`
- 过滤 USD/unit，保留 10-K/10-Q，过滤异常 filed<end 数据
- 0.1s delay between requests，429 时 exponential backoff (5s, 10s, 20s)

### `yfinance_update.py` — yfinance Realtime Update
```bash
python scripts/yfinance_update.py --tickers AAPL,MSFT --output ~/.hermes/fundamental_data/yfinance/
python scripts/yfinance_update.py --all --tickers-file ~/.hermes/fundamental_data/tickers.json --output ~/.hermes/fundamental_data/yfinance/
```
- 用 `yf.Ticker(ticker).info` + `fast_info` 提取 market_cap/shares/PE/ROE/DE 等
- 提取字段: marketCap, sharesOutstanding, trailingPE, forwardPE, trailingEps, forwardEps, bookValue, totalRevenue, netIncomeToCommon, returnOnEquity, debtToEquity, grossMargins, profitMargins, revenueGrowth, earningsGrowth 等

### `merge_fundamental.py` — Merge SEC + yfinance
```bash
python scripts/merge_fundamental.py --ticker AAPL --sec-dir ... --yf-dir ... --output merged/
python scripts/merge_fundamental.py --all --sec-dir ... --yf-dir ... --output ...
```
- yfinance 覆盖"最新"值，XBRL 提供历史时间序列
- 输出结构: `income_statement{revenues, net_income_loss, gross_profit, operating_income}`, `balance_sheet{stockholders_equity, assets, liabilities}`, `shares_outstanding`, `market_cap`, `ratios{roe, debt_to_equity, pe_trailing}`

### `compute_ratios.py` — (TODO) Compute Derived Ratios
```python
# ROE = NetIncomeLoss_4Q_sum / StockholdersEquity_latest
# Earnings Yield = EPS / Price
# Debt/Equity = Liabilities / Equity
```

### `validate_lookahead.py` — (TODO) Verify No Look-Ahead Bias
```bash
python validate_lookahead.py --ticker AAPL --merged-dir ~/.hermes/fundamental_data/merged/
# Output: "OK" or "VIOLATION: N points found"
```

## Update Schedule
- **手动**: `python yfinance_update.py --all`（需要最新数据时）
- **自动**: cronjob 每周运行一次
- **SEC XBRL**: 一次性历史收集，不需要定期更新

## Verification
```bash
# 1. 文件数量
ls ~/.hermes/fundamental_data/sec_xbrl/ | wc -l  # 应等于 tickers 数量

# 2. Look-ahead 检查
python validate_lookahead.py --all --merged-dir ~/.hermes/fundamental_data/merged/
# 期望: 0 violations

# 3. 数据完整性
python -c "
import json, glob
data = json.load(open(glob.glob('~/.hermes/fundamental_data/merged/AAPL.json')[0]))
print('Revenues:', len(data['income_statement']['revenues']))
print('NetIncome:', len(data['income_statement']['net_income_loss']))
"
# 期望: AAPL 应有 10+ 年 Revenue 数据（2008+）
```

## Common Issues
1. **company_tickers.json 403**: 加 User-Agent header，兜底用 hardcoded CIK
2. **XBRL 有 Dimensions**: 过滤掉有 Dimensions key 或 abstract 的记录
3. **filed_date 早于 end_date**: 跳过异常数据点
4. **yfinance .info 返回 None**: 用 `get("field", None)` 优雅处理
5. **429 Rate limit**: 严格单线程 + exponential backoff

```bash
# Data lives under the skill's data/ directory
~/.hermes/skills/quant-trading/fundamental-data-collector/data/
  baseline_results.json           # vol20 top10 baseline 回测结果
  multi_factor_results.json       # 多因子 v3 回测结果
  rolling_ic_results.json         # 滚动 IC 因子稳定性分析
  obj_pool_results.json          # 103 只候选池回测结果
  ticker_cik_map.json            # 107 只股票 CIK 映射
```

## Relevant Files (in `data/` and `scripts/`)
- `/tmp/sec_xbrl_fetch.py` — 现有 XBRL fetcher（107 只股票，company_tickers.json 做 CIK lookup）
- `/tmp/fundamental_data.json` — 现有 XBRL 原始数据（Revenues/NetIncomeLoss/StockholdersEquity/SharesOutstanding）
- `/tmp/ticker_cik_map.json` — 现有 CIK 映射（107 只股票）
- `/tmp/multi_factor_v3.py` — 多因子策略（使用 XBRL 数据计算 roe/earnings_yield/de_ratio）
