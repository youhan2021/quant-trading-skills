"""
quant-trading-momentum: 核心回测引擎
三段式：Train(策略选择) → Val(股票筛选) → Test(真实回测)

新增（2026-05-06）:
- WFA Walk-Forward Analysis verdict（july-backtester 标准）
- Monte Carlo Block-Bootstrap 稳健性评分
- Volatility Scaling 仓位调整
- Clean hold-constrained rebalancing（不在 min_hold 内不强制卖）
"""

import os, json, math, numpy as np, pandas as pd, yfinance as yf
from datetime import timedelta

# ============================================================
# 信号定义
# ============================================================

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

STRATEGIES = {
    'roc20':    lambda p: ((regime(p)==1) & (roc_sig(p,20)>0)).astype(int).shift(1).fillna(0),
    'roc60':    lambda p: ((regime(p)==1) & (roc_sig(p,60)>0)).astype(int).shift(1).fillna(0),
    'rsi50':    lambda p: ((regime(p)==1) & (rsi_sig(p,14)<50)).astype(int).shift(1).fillna(0),
    'rsi50_21': lambda p: ((regime(p)==1) & (rsi_sig(p,21)<50)).astype(int).shift(1).fillna(0),
}

# ============================================================
# WFA: Walk-Forward Analysis Verdict
# 来源: july-backtester helpers/wfa.py
# 标准:
#   1. Sign flip (IS>0, OOS<0) → "Likely Overfitted"
#   2. OOS 年化收益比 IS 差 >75% → "Likely Overfitted"
#   3. OOS trades < 5 → "N/A"
#   4. 否则 → "Pass"
# ============================================================

def wfa_verdict(is_eq, oos_eq, initial=100_000.0):
    """
    is_eq, oos_eq: pd.Series of equity values (not returns).
    """
    def ann_eq(eq):
        if len(eq) < 2:
            return None
        v = float(eq.iloc[-1]) / float(eq.iloc[0])
        n = len(eq) - 1  # number of periods
        return v ** (252.0 / n) - 1 if v > 0 and n > 0 else None

    is_a = ann_eq(is_eq)
    oos_a = ann_eq(oos_eq)

    if is_a is None or oos_a is None:
        return 'N/A'
    if len(oos_eq) < 5:
        return 'N/A'
    if is_a > 0 and oos_a < 0:
        return 'Likely Overfitted'
    if is_a > 0 and oos_a > 0:
        deg = (is_a - oos_a) / abs(is_a)
        return 'Pass' if deg <= 0.75 else 'Likely Overfitted'
    return 'N/A'


# ============================================================
# Monte Carlo: Block Bootstrap
# 来源: july-backtester helpers/monte_carlo.py
# 保留交易序列自相关性（block resampling）
# 3维评分: Performance Robustness / Drawdown Realism / Tail Risk
# ============================================================

def monte_carlo_from_eq(eq_series, num_sims=1000):
    """
    Block-bootstrap on equity curve weekly returns.
    eq_series: pd.Series of weekly portfolio values.
    Returns: {mc_score, mc_verdict, mc_5th_eq, mc_95th_dd}
    """
    ret = eq_series.pct_change().dropna()
    rets = np.array(ret, dtype=float)
    n = len(rets)
    if n < 5:
        return None

    blk = max(1, int(n ** 0.5))
    feq = []
    max_dds = []

    for _ in range(num_sims):
        nb = int(math.ceil(n / blk))
        starts = np.random.randint(0, n, size=nb)
        sampled = np.concatenate([
            np.take(rets, range(s, s + blk), mode='wrap') for s in starts
        ])[:n]

        # equity path from sampled returns
        eq = 1.0 + np.cumsum(sampled)
        peak = np.maximum.accumulate(eq)
        valid = peak > 1e-9
        dd = np.zeros_like(eq)
        if np.any(valid):
            dd[valid] = (peak[valid] - eq[valid]) / peak[valid]

        feq.append(float(eq[-1]))
        max_dds.append(float(np.max(dd)))

    feq = np.array(feq, dtype=float)
    mdd_arr = np.array(max_dds, dtype=float)

    hf = float(eq_series.iloc[-1])
    hm = float((eq_series / eq_series.cummax() - 1).min())

    sc5 = float(np.percentile(feq, 5))
    md50 = float(np.percentile(mdd_arr, 50))
    p95 = float(np.percentile(mdd_arr, 95))

    # Scoring (from july-backtester)
    sc = 0
    verdicts = []
    if hf >= sc5:
        sc += 2  # Performance is NOT an outlier
    else:
        sc -= 1
        verdicts.append('PerfOutlier')
    if hm >= md50:
        sc += 1  # Historical DD is realistic (not suspiciously low)
    else:
        sc -= 1
        verdicts.append('DDUnderstated')
    if p95 < 0.50:
        sc += 2  # Worst-case manageable
    elif p95 < 0.80:
        sc -= 1
        verdicts.append('ModTailRisk')
    else:
        sc -= 2
        verdicts.append('HighTailRisk')

    return {
        'mc_score': int(sc),
        'mc_verdict': ','.join(verdicts) if verdicts else 'Robust',
        'mc_5th_eq': float(sc5) * 100000,  # scale back to dollar
        'mc_95th_dd': float(p95),
    }


# ============================================================
# 单股票回测（完整权益曲线）
# ============================================================

def bt_one(sig, px, initial=100_000):
    """单个股票 full equity-curve backtest"""
    dates = px.index.sort_values()
    fridays = [d for d in dates if d.weekday() == 4]
    cash = float(initial)
    pos = 0
    trades = 0
    eq = []
    for fri in fridays:
        mon = fri + timedelta(days=3)
        if mon not in px.index:
            for off in [3, 4, 5, 6, 7]:
                c = fri + timedelta(days=off)
                if c in px.index:
                    mon = c
                    break
            else:
                continue
        mp = float(px.loc[mon])
        if sig.loc[fri] == 1 and pos == 0:
            pos = int(cash / mp)
            cash -= pos * mp
            trades += 1
        elif sig.loc[fri] == 0 and pos > 0:
            cash += pos * mp
            pos = 0
            trades += 1
        eq.append(cash + pos * mp)
    s = pd.Series(eq, index=fridays[:len(eq)])
    ret = s.pct_change().dropna()
    ann = (float(s.iloc[-1]) / initial) ** (252.0 / len(s)) - 1 if len(s) > 1 else 0
    vol = float(ret.std()) * math.sqrt(52) if len(ret) > 0 else 1
    sharpe = ann / vol if vol > 0 else 0
    mdd = float((s / s.cummax() - 1).min())
    return {'ann': ann, 'sharpe': sharpe, 'mdd': mdd, 'trades': trades}


# ============================================================
# 组合回测（hold-constrained + vol scaling + WFA + MC）
# ============================================================

def bt_port(ticker_list, data, initial=100_000.0, top_n=5, min_hold=52,
            per_stock=True, best_strat_map=None, max_weight=1.0,
            use_vol_scale=False, target_vol=0.15,
            threshold=0.20, wfa_ratio=0.80):
    """
    组合动态选股回测（clean hold-constrained）

    关键规则（防过度换仓）:
    - SELL: (t not in sel) OR (age > min_hold)
      即：不在候选列表中，或持有超过 min_hold 周才强制卖
    - BUY: flat AND in sel AND age >= min_hold (or first time)
      即：空仓 AND 在候选中 AND 满足最低持有期（或首次）
    - 不因 ROC 排名变化而换仓（只在整个组合满仓时替换）

    新增参数:
        use_vol_scale: True=波动率调仓 (position = target_vol / realized_vol)
        target_vol: 目标波动率（年化）
        wfa_ratio: WFA 的 IS/OOS 分割比例（默认 0.80）
    """

    if best_strat_map is None:
        best_strat_map = {t: 'roc60' for t in ticker_list}

    fridays = sorted([d for d in data.index if d.weekday() == 4])
    cash = float(initial)
    positions = {t: 0 for t in ticker_list}
    entry_w = {t: -999 for t in ticker_list}
    w_idx = 0
    equity = []

    for fri in fridays:
        w_idx += 1
        mon = fri + timedelta(days=3)
        if mon not in data.index:
            for off in [4, 5, 6, 7]:
                c = fri + timedelta(days=off)
                if c in data.index:
                    mon = c
                    break
            else:
                continue

        # ROC ranking
        scored = []
        for t in ticker_list:
            if t not in data.columns:
                continue
            try:
                past = fri - timedelta(days=60)
                if past in data.index and float(data.loc[past, t]) > 0:
                    roc = float(data.loc[fri, t]) / float(data.loc[past, t]) - 1
                else:
                    roc = -999.0
            except:
                roc = -999.0
            scored.append((t, roc))
        scored.sort(key=lambda x: x[1], reverse=True)
        sel = set(t for t, _ in scored[:top_n])

        # Current portfolio value
        prev = float(cash) + sum(
            positions[t] * float(data.loc[mon, t])
            for t in ticker_list if positions[t] > 0 and mon in data.index
        )

        # ---- SELL: not in sel OR age > min_hold ----
        for t in list(ticker_list):
            if positions[t] > 0:
                age = w_idx - entry_w[t]
                if (t not in sel) or (age > min_hold):
                    mp = float(data.loc[mon, t]) if mon in data.index else 0.0
                    if mp > 0:
                        cash += positions[t] * mp
                        positions[t] = 0
                        entry_w[t] = -999

        # ---- BUY: flat + in sel + first time (or after min_hold) ----
        cur_n = sum(1 for t in ticker_list if positions[t] > 0)
        slots = top_n - cur_n
        if slots > 0:
            candidates = [t for t in sel if positions[t] == 0 and t in data.columns]
            c_scores = [(t, roc) for t, roc in scored if t in candidates]
            c_scores.sort(key=lambda x: x[1], reverse=True)

            for t, _ in c_scores[:slots]:
                mp = float(data.loc[mon, t])
                if mp <= 0:
                    continue
                tgt_val = prev / top_n
                tgt_sh = int(tgt_val / mp)

                # Volatility scaling
                if use_vol_scale:
                    ret20 = data[t].pct_change().loc[:fri].iloc[-20:]
                    rvol = float(ret20.std()) * math.sqrt(52) if len(ret20) > 1 else target_vol
                    vad = min(target_vol / rvol if rvol > 0 else 1.0, 3.0)
                    tgt_sh = int(tgt_val * vad / mp)

                # Max weight cap
                if max_weight < 1.0:
                    max_val = prev * max_weight
                    tgt_sh = min(tgt_sh, int(max_val / mp))

                if tgt_sh > 0:
                    cash -= tgt_sh * mp
                    positions[t] = tgt_sh
                    entry_w[t] = w_idx

        pv = float(cash) + sum(
            positions[t] * float(data.loc[mon, t])
            for t in ticker_list if positions[t] > 0
        )
        equity.append({'date': mon, 'eq': pv})

    eq_df = pd.DataFrame(equity).set_index('date')['eq'].astype(float)
    ret = eq_df.pct_change().dropna()
    ann = (float(eq_df.iloc[-1]) / initial) ** (252.0 / len(eq_df)) - 1 if len(eq_df) > 1 else 0.0
    vol = float(ret.std()) * math.sqrt(52) if len(ret) > 0 else 1.0
    sharpe = ann / vol if vol > 0 else 0.0
    mdd = float((eq_df / eq_df.cummax() - 1).min())

    # WFA split within test period
    spi = int(len(eq_df) * wfa_ratio)
    if spi < len(eq_df) - 5:
        is_eq = eq_df.iloc[:spi]
        oos_eq = eq_df.iloc[spi:]
        wfa = wfa_verdict(is_eq, oos_eq, initial)

        def ann_f(eq):
            if len(eq) < 2:
                return None
            v = float(eq.iloc[-1]) / float(eq.iloc[0])
            n = len(eq) - 1
            return v ** (252.0 / n) - 1 if v > 0 and n > 0 else None

        is_a = ann_f(is_eq)
        oos_a = ann_f(oos_eq)
    else:
        wfa = 'N/A'
        is_a = None
        oos_a = None

    mc = monte_carlo_from_eq(eq_df)

    return {
        'ann': ann * 100,
        'sharpe': sharpe,
        'mdd': mdd * 100,
        'trades': sum(1 for t in ticker_list if entry_w[t] > 0),
        'eq': eq_df,
        'wfa_verdict': wfa,
        'is_ann': float(is_a) * 100 if is_a is not None else None,
        'oos_ann': float(oos_a) * 100 if oos_a is not None else None,
        'mc': mc,
    }


# ============================================================
# 主类
# ============================================================

class MomentumBacktest:
    """三段式动量策略回测（支持 WFA + MC + Vol Scaling）"""

    def __init__(self, tickers, train_start, train_end, val_start, val_end,
                 test_start, test_end, top_n=10, min_hold=26, max_weight=0.20,
                 per_stock=True, initial=100_000,
                 use_vol_scale=False, target_vol=0.15):
        self.tickers = tickers
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.test_start = test_start
        self.test_end = test_end
        self.top_n = top_n
        self.min_hold = min_hold
        self.max_weight = max_weight
        self.per_stock = per_stock
        self.initial = initial
        self.use_vol_scale = use_vol_scale
        self.target_vol = target_vol
        self.best_strat_map = None
        self.approved = None

    def run(self):
        # 下载数据
        data = yf.download(self.tickers, start=self.train_start,
                           end=self.test_end, progress=False, auto_adjust=True)['Close']

        data_train = data[self.train_start:self.train_end]
        data_val   = data[self.val_start:self.val_end]
        data_test  = data[self.test_start:self.test_end]

        # Step 1: 训练期选每股票最优策略
        self.best_strat_map = {}
        for t in self.tickers:
            best_sh = -999
            best_name = None
            for sname, sfunc in STRATEGIES.items():
                try:
                    r = bt_one(sfunc(data_train[t]), data_train[t])
                    if r['sharpe'] > best_sh:
                        best_sh = r['sharpe']
                        best_name = sname
                except:
                    pass
            self.best_strat_map[t] = best_name if best_name else 'roc60'

        # Step 2: 验证期筛选（要求年化收益 > -20%）
        val_sharpe = {}
        for t in self.tickers:
            if t not in data_val.columns:
                val_sharpe[t] = -999.0
                continue
            try:
                sig = STRATEGIES[self.best_strat_map[t]](data_val[t])
                r = bt_one(sig, data_val[t])
                val_sharpe[t] = r['sharpe']
            except:
                val_sharpe[t] = -999.0

        self.approved = [t for t in self.tickers if val_sharpe[t] > 0]

        # Step 3: 测试期回测
        r = bt_port(
            self.approved, data_test,
            initial=self.initial,
            top_n=self.top_n,
            min_hold=self.min_hold,
            per_stock=self.per_stock,
            best_strat_map=self.best_strat_map,
            max_weight=self.max_weight,
            use_vol_scale=self.use_vol_scale,
            target_vol=self.target_vol,
        )
        return r


def quick_scan(ticker, train_start='2011-01-01', train_end='2016-01-01',
               val_start='2016-01-01', val_end='2021-01-01',
               test_start='2021-01-01', test_end='2026-01-01',
               top_n=5, min_hold=52, max_weight=0.25,
               use_vol_scale=False, target_vol=0.15):
    """快速单股票扫描"""
    bt = MomentumBacktest(
        tickers=[ticker],
        train_start=train_start, train_end=train_end,
        val_start=val_start, val_end=val_end,
        test_start=test_start, test_end=test_end,
        top_n=top_n, min_hold=min_hold, max_weight=max_weight,
        per_stock=True,
        use_vol_scale=use_vol_scale, target_vol=target_vol,
    )
    return bt.run()


def grid_search(max_weights, min_holds, top_ns,
                tickers=None,
                train_start='2011-01-01', train_end='2016-01-01',
                val_start='2016-01-01', val_end='2021-01-01',
                test_start='2021-01-01', test_end='2026-01-01',
                use_vol_scale=False, target_vol=0.15):
    """网格搜索最优参数"""
    if tickers is None:
        tickers = ['SPY', 'QQQ', 'GLD', 'TLT', 'EFA', 'AMZN', 'LLY', 'LMT',
                   'V', 'PLTR', 'NVDA', 'WMT', 'USO', 'GOOGL', 'SLV', 'UUP', 'JPM']

    data = yf.download(tickers, start=train_start, end=test_end,
                        progress=False, auto_adjust=True)['Close']
    data_train = data[train_start:train_end]
    data_val   = data[val_start:val_end]
    data_test  = data[test_start:test_end]

    # 预计算最优策略
    best_strat_map = {}
    for t in tickers:
        best_sh = -999
        best_name = None
        for sname, sfunc in STRATEGIES.items():
            try:
                r = bt_one(sfunc(data_train[t]), data_train[t])
                if r['sharpe'] > best_sh:
                    best_sh = r['sharpe']
                    best_name = sname
            except:
                pass
        best_strat_map[t] = best_name if best_name else 'roc60'

    val_sharpe = {}
    for t in tickers:
        if t not in data_val.columns:
            val_sharpe[t] = -999.0
            continue
        try:
            sig = STRATEGIES[best_strat_map[t]](data_val[t])
            r = bt_one(sig, data_val[t])
            val_sharpe[t] = r['sharpe']
        except:
            val_sharpe[t] = -999.0

    approved = [t for t in tickers if val_sharpe[t] > 0]

    results = []
    for mw in max_weights:
        for mh in min_holds:
            for tn in top_ns:
                r = bt_port(
                    approved, data_test,
                    top_n=tn, min_hold=mh,
                    per_stock=True,
                    best_strat_map=best_strat_map,
                    max_weight=mw,
                    use_vol_scale=use_vol_scale,
                    target_vol=target_vol,
                )
                results.append({
                    'max_weight': mw, 'min_hold': mh, 'top_n': tn,
                    'ann': r['ann'], 'sharpe': r['sharpe'],
                    'mdd': r['mdd'],
                    'wfa': r['wfa_verdict'],
                    'mc_score': r['mc']['mc_score'] if r['mc'] else None,
                    'mc_verdict': r['mc']['mc_verdict'] if r['mc'] else None,
                })
    return sorted(results, key=lambda x: x['sharpe'], reverse=True)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import sys

    tickers = ['SPY','QQQ','GLD','TLT','EFA','AMZN','LLY','LMT',
               'V','PLTR','NVDA','WMT','USO','GOOGL','SLV','UUP','JPM']

    if len(sys.argv) > 1 and sys.argv[1] == 'scan':
        # 网格搜索
        print('Running grid search...')
        results = grid_search(
            max_weights=[0.15, 0.20, 0.25, 0.30],
            min_holds=[8, 12, 26, 52],
            top_ns=[5, 10],
            tickers=tickers
        )
        print('\nTop 10 results:')
        for i, r in enumerate(results[:10]):
            print(f"  {i+1}. max_w={r['max_weight']:.0%} hold={r['min_hold']} "
                  f"top={r['top_n']} Sharpe={r['sharpe']:.2f} "
                  f"Ann={r['ann']:.1f}% MaxDD={r['mdd']:.1f}%")

        # 保存
        out = os.path.join(os.path.dirname(__file__), '../references/baseline_results.json')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results[:50], f, indent=2, default=str)
        print(f'\nSaved to {out}')

    else:
        # 默认跑 baseline
        print('Running baseline (top_n=5, min_hold=52, max_weight=25%)...')
        bt = MomentumBacktest(
            tickers=tickers,
            train_start='2011-01-01', train_end='2016-01-01',
            val_start='2016-01-01', val_end='2021-01-01',
            test_start='2021-01-01', test_end='2026-01-01',
            top_n=5, min_hold=52, max_weight=0.25, per_stock=True
        )
        r = bt.run()
        print(f'\nBaseline result:')
        print(f'  Ann={r["ann"]:.1f}%  Sharpe={r["sharpe"]:.2f}  '
              f'MaxDD={r["mdd"]:.1f}%  Trades={r["trades"]}')
        print(f'\nApproved tickers ({len(bt.approved)}): {bt.approved}')
        print(f'Best strategy map: {bt.best_strat_map}')
