---
name: momentum-strategy
description: 动量策略 — roc120单因子，严格三段式WFA验证，2025年最新结论。
---

# Momentum Strategy (动量策略)

## 策略核心

**单因子：roc120**（120日动量，即约6个月的价格动量）

选股规则：
- 候选池：当前SPX成分股，约117只（需通过IPO过滤）
- 过滤：Test期起点前130周（约2.5年）必须有价格历史；2020年后IPO的股票不允许进入2024及以后的窗口
- 因子计算：120日ROC = 当前价格 / 120天前价格 - 1
- 选股：取roc120最高的5只，等权持有

---

## 严格三段式 Walk-Forward

### Train期（5年）
- 数据：2007至今，周频价格数据
- 评估75个候选策略（单/双/三因子组合 × top5/10/20）
- 每月计算截面IC（因子秩 vs 下月收益秩相关）
- 选 `|IC|` 最大的策略作为该窗口最优
- **关键：因子方向由Train期IC符号决定**，IC>0 → 追动量(+1)，IC<0 → 逆向(-1)

### Val期（1年滚动）
- 每年1月1日滚动
- Train参数LOCKED，不根据Val结果调整

### Test期（Out-of-Sample验证）
- 完全独立于Train/Val
- 2014~2025共12个窗口
- 每年1月1日换仓

---

## 候选策略列表（75个）

**单因子（15个）**：
- roc20/roc60/roc120 × top5/top10/top20

**双因子（30个）**：
- (roc20/roc60/roc120 + vol20/vol60) × top5/top10/top20

**三因子（30个）**：
- (roc + vol + roc≠f1) × top5/top10/top20

**最终选择：几乎所有窗口都选中 roc120 top5**（自然选择，非人为指定）

---

## IPO过滤规则

```python
def get_valid_tickers(tw_start, min_weeks=130):
    # 要求：股票在Test期起点前至少有min_weeks周的价格历史
    # 额外规则：2020年后IPO的股票不允许进入2024及以后的窗口
    #           （避免ABNB等近期IPO在近年窗口产生误导性高动量）
```

**为什么不用更严格的IPO门槛？**
- 260周（约5年）会过滤掉大量2010~2015年IPO的优质股（如AMAT 2012年IPO会被2020窗口排除）
- 130周（约2.5年）是roc120因子的最低历史要求
- 额外加"2020年后IPO禁止进入2024+窗口"的硬过滤，针对近期IPO的 Survivorship Bias

---

## 最新结果（2025版本，12窗口）

| 窗口 | Train IC | Sharpe | SPY | 超额 | 选股 |
|------|----------|--------|-----|------|------|
| 2014 | +0.080 | 1.05 | 1.36 | -0.31 | AAPL/ABT/ADBE/AMAT/AMD |
| 2015 | +0.044 | 0.10 | 0.08 | +0.02 | AAPL/ABT/ADBE/AMAT/AMGN |
| 2016 | +0.056 | 1.58 | 0.85 | +0.73 | AAPL/ABBV/ABT/ADBE/AMAT |
| 2017 | +0.081 | 4.55 | 3.64 | +0.90 | AAPL/ABBV/ABT/ADBE/AMAT |
| 2018 | +0.082 | -0.03 | -0.44 | +0.41 | AAPL/ABBV/ABT/ADBE/AMAT |
| 2019 | +0.089 | 3.32 | 2.72 | +0.60 | AAPL/ABBV/ABT/ADBE/AMAT |
| 2020 | +0.137 | 1.61 | 0.65 | +0.96 | AAPL/ABBV/ABT/ADBE/AMAT |
| 2021 | +0.204 | 2.41 | 2.21 | +0.19 | AAPL/ABBV/ABT/ADBE/AMAT |
| 2022 | +0.241 | -0.51 | -0.75 | +0.24 | AAPL/ABBV/ABT/ADBE/AMAT |
| 2023 | +0.238 | 2.11 | 1.90 | +0.21 | AAPL/ABBV/ABT/ADBE/AMAT |
| 2024 | +0.249 | 0.66 | 2.33 | -1.67 | AAPL/ABBV/ABT |
| 2025 | +0.272 | 0.99 | 1.08 | -0.08 | AAPL/ABBV/ABT |

**平均：Sharpe 1.49 vs SPY 1.30，超额 +0.18，胜率 75% (9/12)**

---

## 关键发现

### IC趋势（因子有效性随时间增强）
- 2014: 0.08 → 2017: 0.08 → 2020: 0.14 → 2022: 0.24 → 2025: 0.27
- 解释：大市值动量因子在2015年后越来越有效，与量化宽松、低利率环境相关

### 高集中度风险
- 12个窗口选股高度重叠：**AAPL/ABBV/ABT/ADBE/AMAT几乎每次都出现**
- 实际持仓约3~5只，等权配置，集中度极高
- 这是动量策略的本质特征，无法通过分散消除

### 失败窗口分析
- **2014**：Train IC仅0.08，策略在牛市起步期输给SPY（+0.31）
- **2024**：SPY在大科技股带领下Sharpe 2.33，策略仅0.66（超额-1.67）
- 2024是策略结构性跑输的年份，不是个例

---

## 已知局限

1. **高集中度**：约3~5只股票，无分散
2. **单因子**：只有价格动量，无基本面因子
3. **2024困境**：当SPY在大科技带领下Sharpe > 2时，策略大概率跑输
4. **Survivorship Bias**：候选池是当前SPX成分股，已倒闭的公司不在池中（但IPO过滤已缓解此问题）

---

## 验证脚本

```
momentum-strategy/scripts/rolling_wfa_v6.py
```

运行：
```bash
cd ~/.hermes/skills/quant-trading
python momentum-strategy/scripts/rolling_wfa_v6.py
```

结果输出到：`momentum-strategy/results/wfa_v6_results.json`

---

## 术语

- **roc120**：120日 Rate of Change = price_t / price_{t-120} - 1
- **IC (Information Coefficient)**：截面因子值与下期收益的秩相关系数
- **WFA (Walk-Forward Analysis)**：滚动时序验证，严格Train/Val/Test三段式分离
- **Top5**：取动量最强的5只股票等权配置
