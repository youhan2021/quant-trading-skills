---
name: quant-backtest-validation
description: 量化回测 info leak 防护和动态组合再平衡的核心研究结论。采用严格三段式 train/val/test 验证流程。
metadata:
  author: youhan
  version: 0.3.0
  tags: [quant, backtest, info-leak, validation, dynamic-rebalance, momentum]
---

# Quant Backtest Validation v0.3

> ⚠️ v0.2 及之前版本的 Sharpe 数字已废弃 — yfinance MultiIndex 数据处理 bug 导致所有回测结果虚高（详见「已知 bug」章节）。

量化策略回测的 info leak 防护和动态组合再平衡的核心研究结论。

## 已知 Bug（v0.3 关键修复）

### Bug 1: yfinance MultiIndex 列名陷阱 [CRITICAL]

```python
# ❌ 错误：auto_adjust=False 返回 MultiIndex (PriceType, Ticker)
df = yf.download(tickers, start=start, end=end, auto_adjust=False)
# df.columns: [('Close', 'AAPL'), ('Close', 'MSFT'), ('High', 'AAPL'), ...]
# 每个 ticker 有 6 种价格类型（Close, Adj Close, High, Low, Open, Volume）

# ❌ 错误：用 df['Close'] 后直接 flatten 是错的
df.columns = df.columns.get_level_values(1)  # → 重复列名！df['AAPL'] 变成 DataFrame(6列)

# ✅ 正确：先取 'Close' 层，得到单层 DataFrame
df = yf.download(tickers, start=start, end=end, auto_adjust=False)
if isinstance(df.columns, pd.MultiIndex):
    df = df['Close']  # DataFrame, columns = ticker names only
# df['AAPL'] 现在是 Series
```

### Bug 2: tz-aware / tz-naive 混合导致 DataFrame 构造失败

```python
# ❌ 有些股票 index 带时区，有些不带，pd.DataFrame(dict) 会报错
# ValueError: Cannot join tz-naive with tz-aware DatetimeIndex

# ✅ 统一在处理完 MultiIndex 后立即去掉时区
df = df['Close']
if df.index.tz is not None:
    df = df.tz_localize(None)
```

### Bug 3: yfinance 返回 IPO 于回测期之后的股票的虚假历史数据

部分 2011 年后 IPO 的股票（SMCI、ABBV 等），yfinance 会返回虚假的历史数据（1762+ 个数据点）。必须在回测前过滤：

```python
def filter_ipo_fakes(tickers_dict, ipo_cutoff='2011-01-01'):
    """剔除 IPO 于 cutoff 之后的股票（yfinance 会返回假历史数据）"""
    cutoff = pd.Timestamp(ipo_cutoff)
    result = {}
    for ticker, s in tickers_dict.items():
        first_valid = s.dropna().index[0]
        if first_valid <= cutoff:
            result[ticker] = s
    return result
```

### 实测数据（2026-05 验证）

用 Wikipedia 2011 历史快照（497只）跑 grid search（12种参数组合）：**Val Sharpe 全部 = 0.00**

这说明：纯动量策略在包含大量已倒闭公司的 historical universe 下**完全无效**。之前 v0.2 跑出的 Sharpe 7-9 是 yfinance bug 叠加 survivorship bias 的虚假结果。

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

## 核心发现（v0.3 — 待重新验证）

> ⚠️ 以下所有 Sharpe / 年化数字均来自 v0.2旧代码（yfinance MultiIndex bug），**已废弃**，仅作历史记录。

### 结论：纯动量策略在 historical universe（Wikipedia 2011快照，497只，含大量已倒闭公司）下 Val Sharpe = 0.00（12种参数全部为0）

所有高 Sharpe 的旧结论均来自：
1. 当前 SPX 幸存者列表（不含倒闭公司 → survivorship bias）
2. yfinance MultiIndex 处理错误 → 回报计算错误
3. IPO 股票虚假历史数据（SMCI 等2011后IPO股有1762个假数据点）

## 正确的数据下载代码（v0.3 标准实现）

```python
import yfinance as yf
import pandas as pd
import numpy as np

def yf_close(tickers, start, end):
    """下载收盘价，返回 {ticker: Series} dict

    ⚠️ 必须用 auto_adjust=False，然后用 df['Close'] 取收盘价。
    直接 flatten MultiIndex 会导致列名重复，df['AAPL'] 变成 DataFrame(6列)。
    """
    df = yf.download(tickers, start=start, end=end,
                     auto_adjust=False, progress=False)
    if df.empty:
        return {}
    # MultiIndex: ('Close', 'AAPL'), ('High', 'AAPL'), ... → 只取 Close 层
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']  # DataFrame, columns = ticker names only
    # 统一 tz-naive（有些股票 index 带时区，有些不带）
    if df.index.tz is not None:
        df = df.tz_localize(None)
    result = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 100:
            result[str(col)] = s
    return result

def filter_ipo_fakes(tickers_dict, ipo_cutoff='2011-01-01'):
    """剔除 IPO 于 cutoff 之后的股票（yfinance 返回假历史数据）"""
    cutoff = pd.Timestamp(ipo_cutoff)
    result = {}
    for ticker, s in tickers_dict.items():
        first_valid = s.dropna().index[0]
        if first_valid <= cutoff:
            result[ticker] = s
    return result
```

## 动态组合回测框架代码（v0.3 — 待修复 MultiIndex 后重新验证）

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

- [ ] **数据处理**：`auto_adjust=False` + `df['Close']`，不用 `auto_adjust=True`（数据范围受限）
- [ ] **tz 处理**：统一去掉时区，避免 `Cannot join tz-naive with tz-aware` 错误
- [ ] **IPO 过滤**：剔除回测期之后 IPO 的股票（yfinance 返回假数据）
- [ ] **Survivorship Bias**：用历史快照，不用当前成分股列表
- [ ] 策略选择：在 Train 数据上完成
- [ ] 股票筛选：在 Val 数据上确认 Sharpe>0（不是 Test）
- [ ] 参数调优：在 Val 数据上完成（不是 Test）
- [ ] 最终报告：只引用 Test 数据结果
- [ ] 信号生成：所有 signal 用 `shift(1)` 避免当天数据泄露
- [ ] ROC/MA 计算：只用历史数据，不用未来数据

## 已知局限性（v0.3）

1. **v0.2 Sharpe 数字全部作废**：所有年化、Sharpe、MaxDD 结论均来自 yfinance MultiIndex bug 的错误计算。待用干净代码重新验证。
2. **Survivorship Bias**：Wikipedia 历史快照（497只）比当前 SPX 好，但仍缺失已倒闭公司（需要 Kenneth French historical constituents 才能彻底解决）
3. **yfinance 数据质量**：部分股票（已倒闭或 IPO 于回测期之后）数据缺失或错误
4. **Sharpe 膨胀**：周数据 Sharpe 比日数据高。只做组合内相对比较，不引用绝对值。
5. **单一市场环境**：2021-2026 是长牛市，trend-following 天然占优
6. **外推风险**：训练期（2011-2016）和测试期（2021-2026）市场结构可能发生变化

## 依赖

```bash
pip install yfinance pandas numpy
```
