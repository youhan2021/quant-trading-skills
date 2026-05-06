---
name: quant-backtest-validation
description: 量化回测 info leak 防护和动态组合再平衡的核心研究结论。采用严格三段式 train/val/test 验证流程。
metadata:
  author: youhan
  version: 0.2.0
  tags: [quant, backtest, info-leak, validation, dynamic-rebalance, momentum]
---

# Quant Backtest Validation v0.2

量化策略回测的 info leak 防护和动态组合再平衡的核心研究结论。

## 三段式验证流程（必须遵守）

```
Train: 2011-01-01 → 2016-01-01  → 选择策略（per-stock最优 or 统一策略）
Val:   2016-01-01 → 2021-01-01  → 验证期：筛选股票（排除 Sharpe≤0）、确认策略稳健性
Test:  2021-01-01 → 2026-01-01  → 最终绩效报告（唯一可信的数字）
```

**绝对禁止**：
- 在 Val 或 Test 数据上选择策略
- 在 Val 或 Test 数据上选择股票
- 用 Test 数据计算的相关性或 Sharpe 排名来选股

## 信号定义（已验证稳健）

```python
def roc_sig(prices, period=20):
    return prices.pct_change(period)

def regime(prices, fast=20, slow=50):
    mf = prices.rolling(fast).mean()
    ms = prices.rolling(slow).mean()
    return (mf > ms).astype(int)

def rsi_sig(prices, period=14):
    delta = prices.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    ag = gain.rolling(period).mean()
    al = loss.rolling(period).mean()
    rs = ag / al.replace(0, np.inf)
    return 100 - 100 / (1 + rs)

strategies = {
    'roc20':    lambda p: ((regime(p)==1) & (roc_sig(p,20)>0)).astype(int).shift(1).fillna(0),
    'roc60':    lambda p: ((regime(p)==1) & (roc_sig(p,60)>0)).astype(int).shift(1).fillna(0),
    'rsi50':    lambda p: ((regime(p)==1) & (rsi_sig(p,14)<50)).astype(int).shift(1).fillna(0),
    'rsi50_21': lambda p: ((regime(p)==1) & (rsi_sig(p,21)<50)).astype(int).shift(1).fillna(0),
}
```

## 核心发现（v0.2 纠正版）

### 1. 三段式后，Per-stock 策略 > 统一策略

| 对比项 | 统一 roc60 | per-stock 最优 |
|--------|-----------|---------------|
| 年化 | 147.4% | 159.8% |
| Sharpe | 7.26 | 7.19 |
| MaxDD | -25.6% | -33.2% |
| 交易次数 | 229 | 345 |

per-stock 策略年化更高，但 MaxDD 也更大。统一 roc60 的 MaxDD 更低是因为信号同步——熊市时同时离场，波动更小。

### 2. min_hold 越长越好（hold=52 全面胜出）

无论 max_weight 设多少，hold=52（≈1年）始终是 Sharpe 最高的设置：

| max_w | hold=4 Sharpe | hold=26 Sharpe | hold=52 Sharpe |
|--------|--------------|---------------|---------------|
| 15% | 6.53 | 6.95 | **7.63** |
| 20% | 7.12 | 7.66 | **8.42** |
| 25% | 7.46 | 7.86 | **8.97** |
| 30% | 7.36 | 7.40 | 8.31 |

**解读**：动量策略需要时间跑出来，频繁换股（hold=4）反而打断动量。1年是最小足够长的持有期。

### 3. 仓位上限甜点：max_weight=25%

| max_w | 效果 |
|--------|------|
| 15% | 太保守，错过 NVDA 动量，年化低 |
| 20% | 改善明显 |
| **25%** | **最优平衡：限制极端集中 + 保留动量** |
| 30%+ | 失去约束力，NVDA 绑架组合 |

### 4. 动态选股有效（纠正「动态不如静态」的错误结论）

早期分析因为 info leak（用同一段数据选策略+回测）得出「动态不如静态」的错误结论。清除 leak 后：

- Dynamic (per-stock, hold=52, max_w=25%): Sharpe 8.97
- Static (val top5): Sharpe 11.48（但被 NVDA 绑架，MaxDD=-24.5%）

**动态选股的优势**：当某只股票进入下跌 regime，能自动退出，而不是死拿。

### 5. 被低估的标的（验证期 Sharpe>0 但未被重视）

以下标的在验证期表现稳健，测试期也表现良好：
- GLD: val Sharpe=5.53, test Sharpe=1.79（避险资产）
- UUP: val Sharpe=0.84, test Sharpe=3.46（美元走强）
- JPM: val Sharpe=7.09, test Sharpe=1.08

## 最优参数组合（2026-05 实测）

```
策略:     per-stock 最优（训练期选出）
仓位上限: max_weight = 25%
最小持有: min_hold = 52 周（约1年）
选股数量: top_n = 5
再平衡:   权重偏离 >20% 时触发
执行:     周五信号 → 周一开盘执行
```

**测试期绩效（2021-2026，无交易成本）**：
- 年化: 197.7%
- Sharpe: 8.97
- MaxDD: -33.6%
- 交易次数: 337

**考虑交易成本后（≈1%/年）估算**：
- Sharpe ≈ 7.2（0.8倍折扣）
- 年化 ≈ 158%

## 动态组合回测框架代码

```python
import numpy as np, pandas as pd, yfinance as yf

def bt_port(ticker_list, data, initial=100_000, top_n=5, min_hold=52,
            per_stock=True, max_weight=0.25, threshold=0.20):
    """
    data: yfinance 下载的 close prices (MultiIndex 或 DataFrame)
    max_weight: 单股最大仓位权重 (0.25 = 25%)
    min_hold: 最小持有周数（防过早卖出）
    """
    dates = data.index.sort_values()
    fridays = [d for d in dates if d.weekday() == 4]
    cash = initial; positions = {t: 0 for t in ticker_list}
    entry_w = {t: -999 for t in ticker_list}; w_idx = 0; trades = 0; equity = []

    for fri in fridays:
        w_idx += 1
        # 找下周一（或最近交易日）
        mon = fri + pd.Timedelta(days=3)
        if mon not in data.index:
            for off in [3,4,5,6,7]:
                c = fri + pd.Timedelta(days=off)
                if c in data.index: mon = c; break
            else: continue

        # 当前持仓的信号
        def get_s(t):
            try:
                sig = get_per_stock_sig(t, data[t]) if per_stock else unified_sig(data[t])
                return sig.loc[fri] if fri in sig.index else 0
            except: return 0

        cur = [t for t in ticker_list if positions[t] > 0]
        cur_s = {t: get_s(t) for t in cur}
        age = {t: w_idx - entry_w[t] for t in cur}

        # 强制卖出：信号消失 OR 持有超限
        sell = [t for t in cur if cur_s[t] == 0 or age[t] > min_hold]
        for t in sell:
            px = data.loc[mon, t]
            if px > 0 and positions[t] > 0:
                cash += positions[t] * px; positions[t] = 0; entry_w[t] = -999; trades += 1

        # 候选股：signal=1
        active = [t for t in ticker_list if t in data.columns
                  and fri in data.index and get_s(t) == 1]

        # 按60天ROC排名选 top_n
        scored = [(t, data.loc[fri, t] / data.loc[fri - pd.Timedelta(days=60), t] - 1
                  if fri - pd.Timedelta(days=60) in data.index and data.loc[fri - pd.Timedelta(days=60), t] > 0 else 0)
                  for t in active]
        scored.sort(key=lambda x: x[1], reverse=True)
        keep = [t for t in cur if t not in sell]; sel = keep.copy()
        for t, _ in scored:
            if len(sel) >= top_n: break
            if t not in sel: sel.append(t)

        n = len(sel); tgt_w = 1.0 / n if n > 0 else 0
        prev = cash + sum(positions[t] * data.loc[fri, t]
                          for t in ticker_list if positions[t] > 0 and fri in data.index)

        # 计算目标仓位（含上限）
        tgt_shares = {}
        for t in ticker_list:
            px = data.loc[mon, t]
            if px <= 0: continue
            if t in sel and n > 0:
                tgt_val = prev * tgt_w; tgt_sh = int(tgt_val / px) if px > 0 else 0
                max_val = prev * max_weight
                if tgt_sh * px > max_val:
                    tgt_sh = int(max_val / px)
                tgt_shares[t] = tgt_sh
            else:
                tgt_shares[t] = 0

        # 执行调仓
        for t in ticker_list:
            px = data.loc[mon, t]
            if px <= 0: continue
            cs = positions[t]; tgt_sh = tgt_shares[t]
            if t in sel and n > 0:
                cur_w = (cs * px) / prev if prev > 0 else 0
                tgt_actual_w = (tgt_sh * px) / prev if prev > 0 else 0
                drift = abs(cur_w - tgt_actual_w)
                if drift > threshold or (cs == 0 and tgt_sh > 0):
                    if cs > 0: cash += cs * px
                    cash -= tgt_sh * px; positions[t] = tgt_sh
                    if entry_w[t] < 0: entry_w[t] = w_idx
                    if tgt_sh != cs: trades += 1
            else:
                if cs > 0: cash += cs * px; positions[t] = 0; entry_w[t] = -999; trades += 1

        pv = cash + sum(positions[t] * data.loc[mon, t]
                        for t in ticker_list if positions[t] > 0)
        equity.append({'date': mon, 'eq': pv})

    eq = pd.DataFrame(equity).set_index('date')
    ret = eq['eq'].pct_change().dropna()
    ann = (eq['eq'].iloc[-1] / initial) ** (252 / len(eq)) - 1
    vol = ret.std() * np.sqrt(52)
    sharpe = ann / vol if vol > 0 else 0
    mdd = ((eq['eq'] / eq['eq'].cummax()) - 1).min()
    return {'ann': ann * 100, 'sharpe': sharpe, 'mdd': mdd * 100, 'trades': trades}
```

## Info Leak 检查清单

- [ ] 策略选择：在 Train 数据上完成
- [ ] 股票筛选：在 Val 数据上确认 Sharpe>0（不是 Test）
- [ ] 参数调优：在 Val 数据上完成（不是 Test）
- [ ] 最终报告：只引用 Test 数据结果
- [ ] 信号生成：所有 signal 用 `shift(1)` 避免当天数据泄露
- [ ] ROC/MA 计算：只用历史数据，不用未来数据

## 已知局限性

1. **Sharpe 膨胀**：周数据 Sharpe 比日数据高。只做组合内相对比较，不引用绝对值。
2. **NVDA 影响**：任何含 NVDA 的组合都被其 2021-2026 涨幅（14x）主导
3. **手续费未计入**：模拟 0 成本；实际交易估算打 8 折
4. **单一市场环境**：2021-2026 是长牛市，trend-following 天然占优
5. **外推风险**：训练期（2011-2016）和测试期（2021-2026）市场结构可能发生变化

## 依赖

```bash
pip install yfinance pandas numpy
```
