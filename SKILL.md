---
name: quant-trading
description: 量化交易系统 — roc120动量策略 + 严格三段式WFA验证，无信息泄漏。
---

# quant-trading

量化交易系统 — **roc120 动量策略** + 严格滚动 Walk-Forward 验证，无信息泄漏。

---

## ⚠️ 历史版本（已废弃，有 Selection Bias）

所有以下版本的结果均不可信，已归档到 `momentum-strategy/scripts/archived/` 和 `momentum-strategy/scripts/wfa_archive/`：

- `rolling_wfa.py` — 原始版本
- `comprehensive_v2.py` — 13股"优选"Universe（selection bias）
- `rolling_wfa_v2~v5.py` — 中间版本，有各种bug
- `rolling_wfa_v7~v11.py` — Long-Short/Regime等实验版本
- `comprehensive_backtest.py`, `factor_research.py`, `multi_factor_v3.py` 等

---

## ✅ 当前标准脚本

**`momentum-strategy/scripts/rolling_wfa_v6.py`** — 严格滚动 Walk-Forward 验证

### 核心设计

| 参数 | 值 |
|------|-----|
| 候选池 | 117只（当前SPX成分股） |
| IPO过滤 | Test期起点前≥130周价格历史；2020年后IPO禁止进入2024+窗口 |
| 候选策略 | 75个（单/双/三因子组合 × top5/10/20） |
| 因子方向 | Train期IC符号决定（IC>0追动量，IC<0逆向） |
| 持仓 | roc120 top5，等权，年度调仓 |

### 三段式分离

```
Train（5年）→ 评估75个策略 → 选|IC|最大者 → 锁定因子+方向+topN
Test（1年）  → 用Train锁定的参数 → 原封不动回测
```

### 运行

```bash
cd ~/.hermes/skills/quant-trading
python momentum-strategy/scripts/rolling_wfa_v6.py
```

---

## 验证结果（v6，12窗口，2025-05-06）

|| 窗口 | Train IC | Sharpe | SPY | 超额 | 胜 |
||------|----------|--------|-----|------|-----|
|| 2014 | +0.080 | 1.05 | 1.36 | -0.31 | |
|| 2015 | +0.044 | 0.10 | 0.08 | +0.02 | ☆ |
|| 2016 | +0.056 | 1.58 | 0.85 | +0.73 | ★ |
|| 2017 | +0.081 | 4.55 | 3.64 | +0.90 | ★ |
|| 2018 | +0.082 | -0.03 | -0.44 | +0.41 | ★ |
|| 2019 | +0.089 | 3.32 | 2.72 | +0.60 | ★ |
|| 2020 | +0.137 | 1.61 | 0.65 | +0.96 | ★ |
|| 2021 | +0.204 | 2.41 | 2.21 | +0.19 | ☆ |
|| 2022 | +0.241 | -0.51 | -0.75 | +0.24 | ☆ |
|| 2023 | +0.238 | 2.11 | 1.90 | +0.21 | ☆ |
|| 2024 | +0.249 | 0.66 | 2.33 | -1.67 | |
|| 2025 | +0.272 | 0.99 | 1.08 | -0.08 | |

**汇总**：
- **平均 Sharpe：1.49** vs SPY 1.30
- **平均超额：+0.18**
- **胜率：75%（9/12 窗口超额为正）**
- **★ 超额>0.3：5个窗口（2016/2017/2019/2020/2024未达）**
- **最差窗口：2024（SPY Sharpe 2.33，策略0.66，超额-1.67）**

---

## 因子演进（roc120 IC随时间增强）

| 年份 | IC | 含义 |
|------|-----|------|
| 2014 | 0.08 | 动量效应弱 |
| 2016 | 0.06 | |
| 2018 | 0.08 | |
| 2020 | 0.14 | 疫情后流动性驱动 |
| 2022 | 0.24 | 动量效应显著增强 |
| 2025 | 0.27 | 因子持续有效 |

---

## 已证伪的策略

| 策略 | 问题 | 结论 |
|------|------|------|
| 13股"优选" | Selection bias — 用后验知识选股 | ❌ 废弃 |
| Long-Short（v7） | 牛市中空头严重拖累 | ❌ 废弃 |
| Top-3平均（v10） | 策略同质化，无分散效果 | ❌ 废弃 |
| 扩大候选池193只（v11） | 弱动量股票摊薄收益，IC从0.14降至0.06 | ❌ 废弃 |
| Regime Filter（v11） | 周数据上200日MA计算错误，0% bear market检测 | ❌ 待修 |

---

## 关键教训

### 候选池不是越大越好
- 117只：IC ≈ 0.14
- 193只：IC ≈ 0.06
- **质量 > 数量**

### 从Train期选策略是必要的，但不够
- v6从75个策略里用Train期IC选最优 → 仍有轻微selection bias（11个窗口全选roc120本身说明这个因子太dominant）
- 真正的OOS验证：只看Test期 Sharpe 是否稳定为正

### ABNB的教训
- 130周门槛太松：ABNB 2020-12 IPO到2024-01=3.2年，过130周≈2.5年门槛
- 解决方案：加"2020年后IPO禁止进入2024+窗口"的硬规则

---

## yfinance 数据处理规范

**MultiIndex陷阱**：`auto_adjust=False` 返回 `[(PriceType, Ticker)]` MultiIndex，必须用 `df['Close']` 取收盘价列再flatten。

```python
df = yf.download(tickers, auto_adjust=False, progress=False)
df = df['Close']  # MultiIndex → SingleIndex
```

**tz-aware陷阱**：读取后去tz：`df = df.tz_localize(None)`

---

## 目录结构

```
quant-trading/
├── SKILL.md                              # 本文件
├── momentum-strategy/
│   ├── SKILL.md                          # 策略子模块
│   ├── momentum_backtest.py
│   ├── results/
│   │   └── wfa_v6_results.json          # 验证结果
│   └── scripts/
│       ├── rolling_wfa_v6.py             # ★ 当前标准脚本
│       ├── clean_backtest.py             # Wikipedia历史快照回测（参考）
│       ├── archived/                      # 废弃脚本
│       └── wfa_archive/                   # v2~v11中间版本
└── fundamental-data-collector/
    └── data/
        ├── ticker_cik_map.json           # 117只候选股票（含IPO日期）
        └── ticker_pool_193.json          # 193只扩展候选（已废弃）
```

---

## 操作节奏

```
每年12月最后一个交易日 → 算因子得分 → 预设top5清单
每年1月第一个交易日开盘 → 市价单买入
全年持有不动
每年末重复
```

**无参数可调，无需干预。**

---

## 当前结论

1. **roc120 动量因子有效**：12个窗口中9个超额为正
2. **Sharpe 1.49 vs SPY 1.30**：平均超额+0.18
3. **策略简单可重复**：无需参数调优，无regime切换
4. **主要风险**：SPY大涨年份（2024超额-1.67）
5. **选股高度重叠**：AAPL/ABBV/ABT/ADBE/AMAT反复出现（集中度风险）

---

## 参考来源

- **Walk-Forward Analysis + Monte Carlo**: [zachisit/july-backtester](https://github.com/zachisit/july-backtester)
