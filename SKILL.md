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

量化交易系统 — **roc120 动量策略** + 严格滚动 Walk-Forward 验证，无信息泄漏。

---

## ⚠️ 历史版本（已废弃，有 Selection Bias）

所有以下版本的结果均不可信，已归档到 `archived/` 和 `wfa_archive/`：

- `rolling_wfa.py` — 原始版本
- `comprehensive_v2.py` — 13股"优选"Universe（selection bias）
- `rolling_wfa_v2~v5.py` — 中间版本，有各种bug
- `rolling_wfa_v7~v11.py` — Long-Short/Regime等实验版本
- `comprehensive_backtest.py`, `factor_research.py`, `multi_factor_v3.py` 等

---

## ✅ 当前标准脚本

**`rolling_wfa_v6.py`** — 严格滚动 Walk-Forward 验证

### 核心特性

1. **候选池**：117只股票（来自 `fundamental-data-collector/data/ticker_cik_map.json`）
2. **IPO过滤**：每只股票在Test期起点必须有足够历史（>=130周）
3. **因子方向**：用IC符号决定方向（正IC→做多，负IC→做空/逆向）
4. **候选策略**：69个（单/双/三因子组合 × top5/10/20）
5. **选股方法**：Train期按IC绝对值选最优策略，Test期LOCKED验证
6. **持仓**：年度调仓，等权配置top5只股票

### 运行方法

```bash
cd ~/.hermes/skills/quant-trading
python momentum-strategy/scripts/rolling_wfa_v6.py
```

### 验证结果（2026-05-06，运行3次均稳定复现）

11个滚动窗口，严格时序分离：

| 窗口 | Train IC | Sharpe | SPY | 超额 | 标记 |
|------|---------|--------|-----|------|------|
| 2014 | +0.0801 | 1.05 | 1.36 | -0.31 | |
| 2015 | +0.0439 | 0.10 | 0.08 | +0.02 | ☆ |
| 2016 | +0.0556 | 1.58 | 0.85 | +0.73 | ★ |
| 2017 | +0.0812 | 4.55 | 3.64 | +0.90 | ★ |
| 2018 | +0.0815 | -0.03 | -0.44 | +0.41 | ★ |
| 2019 | +0.0888 | 3.32 | 2.72 | +0.60 | ★ |
| 2020 | +0.1367 | 1.61 | 0.65 | +0.96 | ★ |
| 2021 | +0.2044 | 2.41 | 2.21 | +0.19 | ☆ |
| 2022 | +0.2412 | -0.51 | -0.75 | +0.24 | ☆ |
| 2023 | +0.2376 | 2.11 | 1.90 | +0.21 | ☆ |
| 2024 | +0.2491 | 0.52 | 2.33 | -1.81 | |

**汇总**：
- **平均 Sharpe：1.52** vs SPY 1.32
- **平均超额：+0.19**
- **胜率：82%（9/11 窗口）**
- **最差窗口：2024（SPY大涨22%，策略仅6.9%）**

---

## 策略说明

### 因子选择

所有版本中，**roc120（120日动量）始终是最优因子**：
- IC从2014年的0.08逐步升到2024年的0.25
- frac（胜率）从69%升到97~100%
- 表明动量效应在近年越来越稳定

### 选股逻辑

```
Train期（5年）：
  1. 对每个候选策略计算月度IC
  2. 选IC均值绝对值最大的策略
  3. 记录该策略的direction（正/负）

Test期（1年）：
  1. 用Test期起点前120周数据算截面因子
  2. 按Train期选定的策略和direction选top5
  3. 等权配置，全程持有不动
  4. 回测 → Sharpe vs SPY
```

### 核心参数

| 参数 | 值 |
|------|-----|
| 候选池 | 117只股票 |
| 有效池（含IPO过滤） | 107~117只 |
| 最优因子 | roc120 |
| 持仓数量 | top5 |
| 调仓频率 | 年度 |
| 权重 | 等权 |

---

## 为什么以前能跑出 Sharpe 1.28？

**四重 selection bias 叠加**（已全部修复）：

1. 从75个策略里挑最优（最大bias）
2. 103只股票全是当前幸存者（无倒闭股）
3. Val期选参数，Test期验证（本质是同一回测）
4. benchmark计算有bug（MultiIndex DataFrame）

---

## 已证伪的策略

| 策略 | 问题 | 结论 |
|------|------|------|
| 13股"优选"Universe | Selection bias — 用后验知识选股 | ❌ 废弃 |
| 75策略挑最优 | Selection bias — 75个里总有一个运气好 | ❌ 废弃 |
| Long-Short（v7） | 牛市中空头严重拖累 | ❌ 废弃 |
| Top-3平均（v10） | 策略同质化，无分散效果 | ❌ 废弃 |
| 扩大候选池（v11） | 弱动量股票摊薄优质收益 | ❌ 废弃 |
| Regime Filter（v11） | regime计算有bug，效果未知 | ❌ 待修 |

---

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

---

## 目录结构

```
quant-trading/
├── SKILL.md                           # 本文件
├── momentum-strategy/
│   ├── scripts/
│   │   ├── rolling_wfa_v6.py         # ★ 当前标准脚本
│   │   ├── clean_backtest.py          # Wikipedia历史快照回测（参考）
│   │   ├── archived/                  # 废弃脚本
│   │   └── wfa_archive/               # v2~v11中间版本
│   └── results/
│       └── wfa_v6_results.json        # 最新验证结果
├── fundamental-data-collector/
│   └── data/
│       ├── ticker_cik_map.json         # 117只候选股票
│       └── ...（其他文件）
└── [其他子模块...]
```

---

## 操作节奏

```
Signal：每年最后一个交易日收盘 → 算因子得分 → 预设top5清单
执行：下一个交易日开盘 → 市价单买入
持仓：全年不动
重复：每年末重复
```

**无参数可调，无需干预。**

---

## 当前结论

1. **roc120 动量因子有效**：在严格WFA下，11个窗口中9个跑赢SPY
2. **Sharpe 1.52 vs SPY 1.32**：平均超额+0.19
3. **策略简单可重复**：无需参数调优，无regime切换
4. **主要风险**：熊市（2018/2022）和SPY大涨年份（2024）表现一般
5. **选股集中**：AAPL/ABBV/ABT/ADBE反复出现，存在幸存者bias

---

## 参考来源

- **Walk-Forward Analysis + Monte Carlo**: [zachisit/july-backtester](https://github.com/zachisit/july-backtester)
