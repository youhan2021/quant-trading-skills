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

量化交易系统 — 多因子动量策略 + 严格滚动 Walk-Forward 验证，无信息泄漏。

## ⚠️ 历史版本（有 Selection Bias，废弃）

### 13股"优选"Universe（已废弃）

**问题**：这13只股票是历史上存活 + 表现良好的样本，用它们回测等同于用后验知识选股。

### 75策略挑最优（已废弃）

从75个策略里挑Test Sharpe最高 → "Sharpe 1.28" → 本质是 **selection bias**：75个策略里总有一个运气好，1.28是偶然，不是能力。

### comprehensive_v2_results.json（已废弃）

103只幸存者股票 + 75个策略 → 双重selection bias。所有数字不可信。

## 严格 Walk-Forward 验证方法（2026-05-06 确立）

### 核心原则

**每个滚动窗口完全独立，不共享任何"未来"信息：**

```
Train 2009~2014 → Test 2014
Train 2010~2015 → Test 2015
Train 2011~2016 → Test 2016
...
Train 2019~2024 → Test 2024
```

| 阶段 | 数据范围 | 用途 |
|------|---------|------|
| **Train** | 5年滑动窗口 | 计算因子IC → 确定因子方向和权重 |
| **Test** | 1年 | 用Train期LOCKED的因子权重选股，**完全不调整** |

### 5步走流程（每个窗口）

```
1. Train期月度IC分析 → 因子方向（n≥30 & frac>55%）
2. 用IC均值作为权重
3. 用Test期起点前120周数据算截面因子
4. 综合得分 → 选top5
5. Test期回测 → Sharpe vs SPY
```

### 有效因子标准

- **n ≥ 30**：数据点足够（60个月 ≈ 5年）
- **frac > 55%**：IC为正的月份超过55%
- 同时满足才作为有效因子，否则跳过该窗口

### 判定策略是否有效

- **胜率**：跑赢SPY的窗口占比
- **平均超额**：Avg(策略Sharpe - SPY Sharpe)
- **严格标准**：胜率>60% 且 平均超额>0.3 才算通过

### 脚本位置

```
momentum-strategy/scripts/rolling_wfa_v2.py
```

### 运行方法

```bash
cd ~/.hermes/skills/quant-trading
python momentum-strategy/scripts/rolling_wfa_v2.py
```

### 验证结果（2026-05-06）

10个滚动窗口，严格时序分离：

| 窗口 | Train期 | Test期 | 策略Sharpe | SPY | 超额 | 有效因子 |
|------|---------|--------|-----------|-----|------|---------|
| 2014 | 09~14 | 2014 | 0.43 | 1.36 | -0.93 | roc20(-) |
| 2015 | 10~15 | 2015 | -0.95 | 0.08 | -1.03 | roc60(-), roc120(-) |
| 2016 | 11~16 | 2016 | **2.87** | 0.85 | **+2.03** | roc120(-) |
| 2017 | 12~17 | 2017 | 跳过 | — | — | 无有效因子 |
| 2018 | 13~18 | 2018 | -1.00 | -0.44 | -0.57 | roc20(-), roc60(-), roc120(-) |
| 2019 | 14~19 | 2019 | 0.88 | 2.72 | -1.83 | roc120(-) |
| 2020 | 15~20 | 2020 | **3.57** | 0.65 | **+2.92** | roc20(+), roc60(+) |
| 2021 | 16~21 | 2021 | 1.18 | 2.21 | -1.04 | roc20(+), roc60(+), roc120(+) |
| 2022 | 17~22 | 2022 | -0.81 | -0.75 | -0.06 | roc20(+), roc60(+), roc120(+) |
| 2023 | 18~23 | 2023 | -0.05 | 1.90 | -1.95 | roc20(+), roc60(+), roc120(+) |
| 2024 | 19~24 | 2024 | **2.97** | 2.33 | **+0.65** | roc20(+), roc60(+), roc120(+), vol20(-) |

**汇总**：
- 平均Sharpe：0.91 vs SPY 1.09
- 超额收益：-0.18
- **胜率：30%（3/10）**
- **结论：纯粹动量策略在严格WFA验证下跑不赢SPY**

### 实盘推荐（Train 2020~2024）

```
选股: ['NVDA', 'AVGO', 'NFLX', 'META', 'GE']
因子权重: roc20=16%, roc60=20%, roc120=23%, vol20=20%, vol60=20%
```

⚠️ 注意：这是基于历史WFA验证的结果，但WFA显示策略整体无效（胜率30%），实盘应谨慎。

### 为什么以前能跑出Sharpe 1.28？

**四重selection bias叠加**：
1. 从75个策略里挑最优（最大bias）
2. 103只股票全是当前幸存者（无倒闭股）
3. Val期选参数，Test期验证（本质是同一回测）
4. benchmark计算有bug（MultiIndex）

### 已证伪的策略

| 策略 | 问题 |
|------|------|
| 13股"优选"Universe | Selection bias — 用后验知识选股 |
| 75策略挑最优 | Selection bias — 75个里总有一个运气好 |
| Regime Switch | WFA验证：固定胜4/12年，不推荐 |
| Volatility Scaling | 降低收益或增加MaxDD |
| Val Sharpe>0筛选 | 预测能力几乎为零（corr=-0.173） |

## yfinance 数据处理规范

**MultiIndex陷阱**：`auto_adjust=False` 返回 `[(PriceType, Ticker)]` MultiIndex，必须用 `df['Close']` 取收盘价列再flatten。

```python
def yf_close(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return {}
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']  # 只取收盘价，columns变成ticker names
    if df.index.tz is not None:
        df = df.tz_localize(None)
    result = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 100:
            result[str(col)] = s
    return result
```

**假历史数据**：对2011年之前IPO的股票，yfinance可能返回虚假历史。需用 `firstTradeDateEpochGregorian` 验证。

## 文件结构

```
quant-trading/
├── SKILL.md                           # 本文件
├── momentum-strategy/
│   └── scripts/
│       ├── rolling_wfa_v2.py         # ★ 严格滚动WFA标准脚本
│       └── clean_backtest.py          # Wikipedia历史快照回测
└── fundamental-data-collector/
    └── data/
        ├── ticker_cik_map.json         # 117只候选股票
        └── comprehensive_v2_results.json  # 废弃（selection bias）
```

## 操作节奏

```
Signal：每周最后一个交易日收盘 → 算因子得分 → 预设top5清单
执行：下一个交易日开盘 → 市价单买入
持仓：本周全程不动
重复：每周末重复
```

**无参数可调，无需干预。**

## 参考来源

- **Walk-Forward Analysis + Monte Carlo**: [zachisit/july-backtester](https://github.com/zachisit/july-backtester)
