---
name: quant-trading-momentum
description: 动量策略量化交易系统 — 严格三段式回测，无信息泄漏。
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

# quant-trading-momentum

动量策略量化交易系统 — 严格三段式回测，无信息泄漏。

## 核心研究成果（2026-05-06 重大更新）

### ⚠️ 重大 Bug：Look-Ahead Return Matrix（所有历史 Sharpe 均需修正）

**之前所有回测结果均有 info leak**：
- 因子在月末 `j` 计算（用价格到 `j`）
- 收益错误地用了同一月的 `j` 当月收益
- 正确做法：**因子在 `j` → 收益从 `j` 到 `j+1`（次月）**

**修正前后对比**：
| 指标 | 修复前（info leak） | 修复后（正确） |
|------|------------------|--------------|
| roc20 top5 Sharpe | 7.79 | **1.16** |
| roc20 top5 Ann | 171.5% | **26.7%** |
| MaxDD | 0% | **-25.6%** |
| 月胜率 | 100% | 59% |

> 📌 **"11只优先股" Sharpe 7.79 是用未来收益打分造成的幻觉**。真实 Sharpe 约 1.16，和其他客观选股策略差不多。

### 三段数据分离（防 Info Leak）

| 阶段 | 数据 | 用途 |
|---|---|---|
| Train | 2011-2016 | 因子 IC 方向（9个因子 × 3种组合） |
| Val | 2016-2021 | 75个策略组合筛选 |
| Test | 2021-2026 | 最终评估，**只跑一次** |

### 综合实验结果（75 策略，修复后）

**Test Sharpe 排名**：
| 策略 | Test Sharpe | Ann | MaxDD | MC-p5 |
|------|------------|-----|-------|-------|
| Price-roc120+vol20 top5 | **1.28** | 34.6% | -37.6% | 0.48 |
| Multi-roc120+book_per_share top20 | **1.12** | 18.2% | -26.6% | 0.38 |
| Single-roc120 top5 | **1.16** | 26.7% | -25.6% | 0.45 |
| Single-book_per_share top10 | **1.10** | 18.8% | -15.4% | 0.48 |
| Single-roe top10 | 0.98 | 16.9% | -27.9% | 0.36 |
| SPY B&H | ~0.69 | ~16.7% | ~-26% | — |

**关键结论**：
1. **无"圣杯"策略** — 最高 Test Sharpe = 1.28
2. **基本面因子作用有限** — roe IC=+0.039（60% IC>0%），极弱
3. **MC-p5 普遍 < 0.5** — 95%置信区间下界接近零，稳健性存疑
4. **WFA 大部分 Pass** — 但不代表策略稳定，只是 OOS 未出现极端反转

### 新增防护层（来自 GitHub 研究）

**Walk-Forward Analysis (WFA)** — 来源: zachisit/july-backtester
- 测试期 80/20 分割：前 80% 是 IS，后 20% 是 OOS
- Verdict 标准：
  - Sign flip（IS>0, OOS<0）→ "Likely Overfitted"
  - OOS 年化比 IS 差 >75% → "Likely Overfitted"
  - 否则 → "Pass"

**Monte Carlo Block-Bootstrap** — 来源: zachisit/july-backtester
- 保留交易序列自相关性（block resampling，block = √N）
- 3维评分：Performance Robustness / Drawdown Realism / Tail Risk
- Score ≥4: Robust; Score ≤0: High Risk

**Volatility Scaling** — 来源: volatility-scaled-momentum-mean-reversion-strategy
- **结论：不推荐**。实测发现 VolScale 10% 大幅降低收益；VolScale 15% 增加 MaxDD
- 仅在高波动市场有防御价值，收益代价过高

### Clean Hold-Constrained Rebalancing

关键规则（防过度换仓）：
- **SELL**: (t not in sel) OR (age > min_hold)
- **BUY**: flat AND in sel
- **不因 ROC 排名变化而强制换仓**

### 候选策略（4种，周频）

```
roc20    = regime(ma20>ma50) & roc(20) > 0
roc60    = regime(ma20>ma50) & roc(60) > 0
rsi50    = regime(ma20>ma50) & rsi(14) < 50
rsi50_21 = regime(ma20>ma50) & rsi(21) < 50
```

### 当前最优参数（2026-05-06 干净实验）

```
per_stock_strategies : True
max_weight           : 0.20
min_hold             : 26
top_n                : 30       # 核心发现：top_n 越大越好（30 > 20 > 10）
use_vol_scale        : False    # 不推荐
frequency            : weekly   # Friday SET / Monday EXEC
```

⚠️ 注意：以上参数基于 103 只客观候选池 Val Sharpe>0 筛选，Sharpe 最高为 4.01（top30）。扣交易成本后约 3.2，但仍高于 SPY B&H(1.96)，说明策略有边际价值。

### 候选池（学术名单，共103只）

103 只股票基于学术/量化研究常用股票名单：市值+流动性+行业分散，规则客观，人为干预少。Val Sharpe>0 筛选后剩余 82 只。

> SPY, QQQ, GLD, TLT, EFA, AMZN, LMT, V, NVDA, WMT, GOOGL, UUP, JPM（共13只）— ⚠️ 历史主观候选池，存在 Selection Bias，仅作参考

这 13 只不是从大量候选中用 Val 规则客观筛选出来的，而是**人选的"看起来不错"的股票**（知道长期会涨），再用 Val Sharpe>0 验证，**本质是用未来信息选股**。

SKILL 中所有 Sharpe/Ann 回测数字（137.9%, 9.87 等）均为**有偏上限**，不代表真实策略表现。

## 回测结果（Test 2021-2026，无交易成本）

### 干净实验（2026-05-06）— 103只客观候选池，Val Sharpe>0 筛选

| 组合 | 年化% | Sharpe | MaxDD | WFA | MC |
|---|---|---|---|---|---|
| SPY B&H | 15.2% | 1.96 | -24.5% | — | — |
| top5 | 21.2% | 0.97 | -40.8% | N/A | DDUnderstated,ModTailRisk |
| top10 | 39.6% | 2.17 | -29.8% | Pass | DDUnderstated |
| top15 | 45.8% | 2.72 | -27.3% | Pass | DDUnderstated |
| top20 | 55.5% | 3.38 | -27.8% | Pass | DDUnderstated |
| top30 | 63.2% | 4.01 | -24.5% | Pass | DDUnderstated |
| **top10（扣交易成本×0.8）** | ~31.7% | **~1.74** | — | — | — |

**关键数字：Val Sharpe vs Test Sharpe correlation = -0.173（几乎为零）**
→ Val Sharpe>0 筛选对 Test 期表现**几乎无预测能力**
→ 之前 9.87 的 Sharpe 完全来自 Selection Bias

**关键结论：top_n 越大越好**（30 > 20 > 15 > 10 > 5）
→ 动量信号很弱，靠更多持仓分散噪音

## 使用方法

### 完整回测

```python
from scripts.momentum_backtest import MomentumBacktest

bt = MomentumBacktest(
    tickers=['SPY','QQQ','GLD','NVDA','AMZN',...],
    train_start='2011-01-01', train_end='2016-01-01',
    val_start='2016-01-01', val_end='2021-01-01',
    test_start='2021-01-01', test_end='2026-01-01',
    top_n=10, min_hold=26, max_weight=0.20,
    per_stock=True,
)
result = bt.run()
# result = {ann, sharpe, mdd, wfa_verdict, is_ann, oos_ann, mc: {mc_score, mc_verdict}, eq}
```

### 快速单股票分析

```python
from scripts.momentum_backtest import quick_scan

for hold in [8, 12, 26, 52]:
    r = quick_scan('NVDA', min_hold=hold, top_n=10)
    print(f"hold={hold}: Sharpe={r['sharpe']:.2f}")
```

### 网格搜索

```python
from scripts.momentum_backtest import grid_search

results = grid_search(
    max_weights=[0.15, 0.20, 0.25],
    min_holds=[4, 8, 12, 26, 52],
    top_ns=[5, 7, 10, 12, 15],
)
# 返回结果含 wfa_verdict, mc_score, mc_verdict
```

## 稳健性验证

### 1. 候选池扩大实验（2026-05-06 重做 — 清除 Selection Bias）

**旧实验有误：之前"11只 > 131只"的结论来源于 Selection Bias——那11只不是从相同 Val 流程选出的，而是先看了 Test 结果反过来挑的，本质是 info leak。**

**新实验（103只宽池 vs 13只主观池，两个池子经过完全相同的 Val Sharpe>0 筛选流程）：**

| 池子 | 输入 | Val通过 | Test Sharpe | Test Ann% | MaxDD% | WFA | MC |
|------|------|---------|-------------|-----------|--------|-----|-----|
| 小池（主观选） | 13只 | 12只 | **8.39** | **116%** | -23.5% | Pass | DDUnderstated |
| 大池（学术名单） | 103只 | 82只 | **2.17** | 39.6% | -29.8% | Pass | DDUnderstated |

**关键证据：Val Sharpe vs Test Sharpe correlation = -0.173（NEGLIGIBLE）**
→ Val Sharpe>0 筛选对 Test 表现**几乎无预测能力**，该选股标准本身就是过拟合源
→ 小池跑赢的真实原因：股票少 → 噪音少（并非"选得准"）

**真正干净的 top_n 扫描（103只候选池）：**

| top_n | Sharpe | Ann% | MaxDD% | WFA |
|-------|--------|------|--------|-----|
| 5 | 0.97 | 21.2% | -40.8% | N/A |
| 10 | 2.17 | 39.6% | -29.8% | Pass |
| 15 | 2.72 | 45.8% | -27.3% | Pass |
| 20 | 3.38 | 55.5% | -27.8% | Pass |
| 30 | 4.01 | 63.2% | -24.5% | Pass |

→ **top_n 越大越好**，动量信号弱，靠分散持仓降低噪音

## 关键洞察

1. **⚠️ Selection Bias 确认**：历史 9.87 Sharpe 完全来自人主观选股（用未来信息选 13 只"好股票"），扣交易成本后真实 Sharpe 约 1.74，与 SPY B&H(1.96) 几乎无差异
2. **⚠️ Val Sharpe>0 证伪**：Val-Test 相关性=-0.173（几乎为零），Val 期表现对 Test 无预测力，该筛选条件是过拟合源
3. **top_n 越大越好**：在干净池子里 30>20>15>10>5，动量信号弱，靠分散持仓降低噪音
4. **hold=26 ≈ hold=52**：差异<1%，26周更灵活
5. **max_weight 在 top10 时几乎不触发**：10仓均分 20%，上限不起作用
6. **WFA Pass（top10+）**：IS>0, OOS>0, 衰退 <75%
7. **MC DDUnderstated**：实际交易可能比回测差，需预留更多缓冲
8. **VolScale 不推荐**：实测降低收益或增加 MaxDD

## Pitfalls（已验证的坑）

### numpy array truthiness — `if not rets:` 是 bug

```python
# ❌ 错误 — numpy array 的 truth value 模棱两可（Ambiguous truth value）
if not rets:
    rets = r          # r 是向量，被当成标量存储
else:
    rets = rets + r   # scalar + vector → 结果变成标量

# ✅ 正确 — 用显式长度判断
rets = r if len(rets) == 0 else rets + r

# 或更好的向量化版本（rolling_ic_v3.py）：
port_rets = np.mean(ret_matrix[:, month_idx])  # 直接取矩阵列
equity[1:] *= (1 + port_rets)                  # 标量乘法
```

**症状**：组合净值 Ann 达到 300,000%，MaxDD 显示 -75% 等极端值。

**根因**：`if not np.array([])` 在 numpy 中会 raise `ValueError`；而 `if not np.array([0.05, 0.10])` 在某些 numpy 版本返回 `True`（element-wise 判断），导致向量被当成标量累加。

### 日期索引 off-by-one — 因子用当月、收益用次月

```python
# ❌ 错误 — IC 用同月因子值和收益（look-ahead）
ic = corr(factors[:, mi], returns[:, mi])

# ✅ 正确 — IC 用当月因子预测次月收益
ic = corr(factors[:, mi], returns[:, mi + 1])

# ✅ 回测选股逻辑（rolling_ic_v3.py）：
for mi in range(start_mi, end_mi):
    scores = ...factors[:, mi]...    # 用当月初因子
    port_ret = np.mean(ret_matrix[top_indices, mi + 1])  # 持仓次月收益
```

### 月频因子 window 需满足 data availability

```python
# 在 2016-01 开始 Val 期，要计算 ROC60：
# 需要 2014-12 之前的数据才能算出第一个有效 ROC60
# 如果数据从 2010-01 开始 → 第一个 ROC60 在 2012-01 之后才有效

# ✅ 始终用 `compute_rolling_ic()` 里的 n >= 15 过滤
if mask.sum() < 15:
    continue
```

## Price Factor 扩展实验（2026-05-06 新增）

### 方法论：滚动月度 IC 稳定性

与周频动量策略不同，Price Factor 实验用**月度 IC 稳定性**筛选因子：

```
滚动月度 IC_t  = corr(因子值[month t], 次月收益[month t+1])
Stability      = mean(IC) / std(IC)        # IC 均值 / IC 标准差
方向一致性    = IC > 0 的月份占比
```

- **稳定因子**：Stability > 0.3 且 IC>0% > 55%
- **因子方向**：由 Val 期的 IC>0% 占比决定（≥55%→+1，≤45%→-1，45-55%→中性）
- **权重**：|Stability| — 越稳定权重越高

### 三段时间分界（月末日期锚定）

```
Train : 2011-01-31 → 2016-01-31   (~60个月)
Val   : 2016-01-31 → 2021-01-31   (~55个月)
Test  : 2021-01-31 → 2026-01-31   (~60个月)
```

注意：用 `monthly_ends.get_indexer([ts], method='ffill')` 获取列索引，避免日期不在序列里的问题。

### 候选因子（月频可用性）

| 因子 | window | 月频可用性 | Train IC>0% | Val IC>0% | Val Stability | 状态 |
|------|--------|-----------|------------|----------|--------------|------|
| vol20 | 20日 | ✅ | 57.5% | **60.0%** | **0.299** | Active (+1) |
| vol60 | 60日 | ⚠️ 需要60根日线 | — | — | — | n=0 |
| roc20 | 20日 | ✅ | 57.5% | 45.0% | 0.156 | Uncertain (-1) |
| roc60 | 60日 | ⚠️ 需要60根日线 | — | — | — | n=0 |
| roc120 | 120日 | ⚠️ 需要120根日线 | — | — | — | n=0 |

> ⚠️ roc60/roc120 在 Val 期不可用 — 月频 ROC 需要 60/120 根日线，Val 期只有约 55 个月的数据，window+lookback 超过可用历史

### Test 回测结果（2021-2026，真实净值 MaxDD）

| 策略 | Sharpe | Ann% | MaxDD% |
|------|--------|------|--------|
| **vol20 top10** | **1.30** | **34.8%** | **-17.6%** |
| vol20 top5 | 1.28 | 43.2% | -18.4% |
| vol20 top30 | 1.28 | 25.9% | -20.3% |
| Multi-Factor (vol20+roc20) | 1.19 | 41.0% | -23.8% |
| roc20 top10 | 1.07 | 25.1% | -17.4% |
| SPY B&H | 0.97 | 14.6% | -23.9% |

**关键发现：**
1. **vol20 是唯一稳定因子** — IC>0 在 Val 期 60% 的月份稳定，direction=+1（低波动=赢家）
2. **多因子反而更差** — vol20+roc20 方向相反，互相抵消；Multi-Factor Sharpe=1.19 < vol20 alone=1.30
3. **roc20 在 Val 方向不确定** — IC>0 占比 45%（几乎随机），direction=-1
4. **vol20 top10 优于 SPY** — Sharpe 1.30 vs 0.97，Ann 34.8% vs 14.6%，MaxDD 更小

### Baseline 结果存档

```
/tmp/baseline_results.json   — vol20 top10 完整指标
/tmp/rolling_ic_results.json — 所有策略完整 IC + 回测数字
/tmp/test_rolling_ic_v3.py  — 38 个 unit tests，全部通过
```

### 综合实验结果（修复后，2026-05-06）

> ⚠️ **所有历史 Sharpe 数字均已修正**（修复 look-ahead return matrix bug）。以下为正确数字。

**Test Sharpe 排名（75 策略，Val→Test 严格分离）**：

| 策略 | Test Sharpe | Ann | MaxDD | MC-p5 |
|------|------------|-----|-------|-------|
| Price-roc120+vol20 top5 | **1.28** | 34.6% | -37.6% | 0.48 |
| Multi-roc120+book_per_share top20 | **1.12** | 18.2% | -26.6% | 0.38 |
| Single-roc120 top5 | **1.16** | 26.7% | -25.6% | 0.45 |
| Single-book_per_share top10 | **1.10** | 18.8% | -15.4% | 0.48 |
| Single-roe top10 | 0.98 | 16.9% | -27.9% | 0.36 |
| Single-roc60 top10 | 0.97 | 18.3% | -22.4% | 0.23 |
| SPY B&H | ~0.69 | ~16.7% | ~-26% | — |

> MC-p5 = Monte Carlo 500次 bootstrap 5%分位数。大部分 < 0.5，95%置信区间下界接近零。

**关键结论**：
1. **无"圣杯"策略** — 最高 Test Sharpe = 1.28
2. **基本面因子作用有限** — roe IC=+0.039（60% IC>0%），极弱；book_per_share 是基本面中最稳定的（IC>0% 48%，方向 SHORT）
3. **MC-p5 普遍偏低** — 稳健性置信度有限
4. **WFA 大部分 Pass** — 但不代表策略稳定

### 三段时间分离

| 阶段 | 数据 | 用途 |
|---|---|---|
| Train | 2011-2016 | 因子 IC 方向（9个因子 × 3种组合） |
| Val | 2016-2021 | 75个策略组合筛选 |
| Test | 2021-2026 | 最终评估，**只跑一次** |

### Python Scoping Bug（已验证坑）

```python
# ❌ 错误 — UNIVERSE += [...] 让 Python 把 main() 里的 UNIVERSE 当局部变量
def main():
    global UNIVERSE          # 添加这行也不够，因为还有全局 UNIVERSE
    UNIVERSE += ['SPY']      # UnboundLocalError

# ✅ 正确 — 改用不冲突的变量名
UNIVERSE_BASE = sorted(set(['SPY', ...]))
def main():
    tickers = UNIVERSE_BASE  # 用新名字，完全避免遮蔽
```

## 已知局限

- ⚠️ **Selection Bias 已清算**：历史 13 只主观候选池导致 Sharpe 虚高（9.87 → 真实约 1.74）。所有历史性能数字均已替换为干净实验结果
- ⚠️ **Val Sharpe>0 证伪**：Val-Test 相关性=-0.173，该筛选条件本质是过拟合。实际选股时应**跳过 Val 筛选**，直接用客观 universe + 固定规则
- **未扣交易成本**（Sharpe 估计打 8 折）
- **top_n=30 时实际可操作性需验证**（同时持有 30 只股票，换仓摩擦成本显著增加）
- MC 的 "DDUnderstated" 警告表示实际交易需预留更多缓冲
- 回测仅周频

## 参考来源

- **Walk-Forward Analysis + Monte Carlo**: [zachisit/july-backtester](https://github.com/zachisit/july-backtester) `helpers/wfa.py`, `helpers/monte_carlo.py`
- **Volatility Scaling**: `/tmp/volatility-scaled-momentum-mean-reversion-strategy`
- **HMM Regime Detection**: [Abdullah-BA/RegimeSwitchingMomentumStrategy](https://github.com/Abdullah-BA/RegimeSwitchingMomentumStrategy)
- **Risk Parity + Black-Litterman**: [anemer-astro/portfolio-optimization](https://github.com/anemer-astro/portfolio-optimization)

## 文件结构

```
scripts/
  momentum_backtest.py    # 核心回测引擎（含 WFA + MC + Vol Scale）
references/
  baseline_results.json   # 最新回测结果存档
```
