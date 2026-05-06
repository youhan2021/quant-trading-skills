"""
SPX Momentum Backtest — 三段式验证
====================================
Universe: SPX 成分股（503只，客观定义，不人为筛选）
三段式:
  Train 2011-01 → 2016-01: IC定方向 + 参数扫描
  Val   2016-01 → 2021-01: Sharpe选最优
  Test  2021-01 → 2026-02: 锁死只跑一次
因子: roc + vol rank-zscore
持仓: top5 等权
频率: Weekly rebal
"""

import json, math, time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
TRAIN_START = "2011-01-01"
TRAIN_END   = "2016-01-01"
VAL_START   = "2016-01-01"
VAL_END     = "2021-01-01"
TEST_START  = "2021-01-01"
TEST_END    = "2026-02-01"

TOP_N_LIST    = [3, 5, 7, 10]
ROC_WIN_LIST  = [60, 120, 240]
VOL_WIN_LIST  = [20, 60]
TC            = 0.002  # 20bps

DATA_START    = "2009-01-01"  # 多预留2年做 lookback
DATA_END      = "2026-02-10"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def rzscore(s):
    r = s.rank()
    return (r - r.mean()) / r.std()

def download_weekly(tickers, start, end):
    """下载日线 → 周线"""
    print(f"  下载 {len(tickers)} 只股票日线...", end=" ", flush=True)
    df = yf.download(tickers, start=start, end=end,
                      auto_adjust=False, progress=False)
    closes = df['Close'].dropna(how='all')
    weekly = closes.resample('W').last().dropna(how='all')
    print(f"周线 {weekly.shape}")
    return weekly

def get_factor_score(daily_prices, roc_win, vol_win):
    """某截面计算 roc + vol 因子分数"""
    n = daily_prices.shape[0]
    if n < max(roc_win, vol_win):
        return None, None

    roc = (daily_prices.iloc[-1] / daily_prices.iloc[-roc_win]) - 1
    vol = daily_prices.iloc[-vol_win:].std() * np.sqrt(252)

    combined = rzscore(roc) + rzscore(-vol)  # 低波高分
    return combined, vol

def compute_weekly_signals(daily_prices, weekly_dates, roc_win, vol_win, top_n):
    """
    对每个 weekly date 计算 signal
    用当日之前的历史数据（不含未来）
    返回 {date: {top5, scores}}
    """
    signals = {}
    min_history = max(roc_win, vol_win) + 5

    for i, date in enumerate(weekly_dates):
        if i < min_history:
            continue

        d = daily_prices.loc[:date]
        if d.shape[0] < min_history:
            continue

        combined, _ = get_factor_score(d, roc_win, vol_win)
        if combined is None:
            continue

        top = combined.nlargest(top_n)
        signals[date] = {
            'top5':   top.index.tolist(),
            'scores': top.to_dict()
        }

    return signals

def backtest(signals, weekly_prices, tc=0.0):
    """
    信号驱动回测（每周 rebal）
    signals[date] → 下周持仓
    """
    rets_list = []
    dates = sorted(signals.keys())
    holdings_list = [None]

    for i, date in enumerate(dates):
        if i == 0:
            holdings_list[0] = set(signals[date]['top5'])
            continue

        prev_date = dates[i-1]
        prev_holdings = holdings_list[-1]
        curr_holdings = set(signals[date]['top5'])

        sold   = prev_holdings - curr_holdings
        bought = curr_holdings - prev_holdings
        turnover = (len(sold) + len(bought)) / max(len(prev_holdings), 1)
        did_trade = (len(sold) + len(bought)) > 0

        w1 = weekly_prices.loc[prev_date]
        w2 = weekly_prices.loc[date]
        if w1.isna().any() or w2.isna().any():
            holdings_list.append(curr_holdings)
            continue

        ret = (w2 / w1) - 1
        hold = list(curr_holdings)
        port_ret = ret[hold].mean()
        port_ret -= tc * turnover if did_trade else 0

        rets_list.append({
            'date': date,
            'ret':   port_ret,
            'turnover': turnover if did_trade else 0,
            'did_trade': did_trade,
        })
        holdings_list.append(curr_holdings)

    if not rets_list:
        return pd.DataFrame()

    df = pd.DataFrame(rets_list).set_index('date')
    return df

def calc_sharpe(df_rets):
    if len(df_rets) < 10:
        return -999
    r = df_rets['ret']
    ann = r.mean() * 52
    std = r.std()  * math.sqrt(52)
    return ann / std if std > 1e-10 else -999

def calc_metrics(df_rets):
    if len(df_rets) < 5:
        return {}
    r = df_rets['ret']
    equity = (1 + r).cumprod()
    ann  = r.mean() * 52
    std  = r.std()  * math.sqrt(52)
    mdd  = float((equity / equity.cummax()).min() - 1)
    n_trades = int((df_rets['did_trade'] == True).sum())
    ann_tc = 2 * TC * n_trades / len(r) * 52  # 单程TC×2×年换手
    return {
        'sharpe':    round(ann / std, 2) if std > 1e-10 else -999,
        'ann_ret':   round(ann * 100, 1),
        'ann_vol':   round(std * 100, 1),
        'max_dd':    round(mdd * 100, 1),
        'n_weeks':   len(r),
        'n_trades':  n_trades,
        'ann_tc':    round(ann_tc * 100, 1),  # 年化TC损耗
    }

def annual_detail(df_rets):
    df = df_rets.copy()
    df['year'] = df.index.year
    rows = []
    for yr, g in df.groupby('year'):
        rows.append({
            'year': yr,
            'ret':  round(float((1+g['ret']).prod()-1)*100, 1),
            'n':    len(g),
            'sharpe': round(float(g['ret'].mean()*52 / (g['ret'].std()*math.sqrt(52))), 2)
                   if g['ret'].std() > 1e-10 else 0
        })
    return rows

# ─── Stage 1: Train — IC + 参数扫描 ─────────────────────────────────────────

def run_train(daily, weekly, tickers):
    print(f"\n{'='*60}")
    print(f"[Stage 1] Train {TRAIN_START} → {TRAIN_END}")
    print(f"{'='*60}")

    train_weekly = weekly[TRAIN_START:TRAIN_END]
    train_dates = train_weekly.index.tolist()
    print(f"  Train 周数: {len(train_dates)}")

    # IC: 每周算因子 → 下周收益 IC
    ic_rows = []
    for i in range(30, len(train_dates) - 1):
        d = daily.loc[:train_dates[i]]
        if d.shape[0] < 130:
            continue
        combined, _ = get_factor_score(d, 120, 20)
        if combined is None:
            continue

        # 下周收益
        w1 = train_weekly.loc[train_dates[i]]
        w2 = train_weekly.loc[train_dates[i+1]]
        ret = (w2 / w1) - 1
        ret = ret[combined.index].dropna()

        ic = combined[ret.index].corr(ret)
        ic_rows.append({'date': train_dates[i], 'ic': ic})

    ic_df = pd.DataFrame(ic_rows).set_index('date')
    mean_ic = ic_df['ic'].mean()
    ic_pos  = (ic_df['ic'] > 0).mean()
    print(f"  IC mean={mean_ic:.3f}, IC>0 ratio={ic_pos:.1%}")

    # 参数扫描
    results = []
    total = len(TOP_N_LIST) * len(ROC_WIN_LIST) * len(VOL_WIN_LIST)
    print(f"  参数扫描 {total} 种组合...")

    for top_n in TOP_N_LIST:
        for roc_win in ROC_WIN_LIST:
            for vol_win in VOL_WIN_LIST:
                sig = compute_weekly_signals(daily, train_dates, roc_win, vol_win, top_n)
                bt = backtest(sig, weekly[TRAIN_START:TRAIN_END])
                if bt.empty:
                    continue
                m = calc_metrics(bt)
                results.append({
                    'top_n': top_n, 'roc_win': roc_win, 'vol_win': vol_win,
                    'sharpe': m['sharpe'], 'ann_ret': m['ann_ret'],
                    'max_dd': m['max_dd'], 'n_weeks': m['n_weeks'],
                    'ic_mean': mean_ic, 'ic_pos': ic_pos,
                })

    df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print(f"\n  Train Top5:")
    print(df.head(5)[['top_n','roc_win','vol_win','sharpe','ann_ret','max_dd']].to_string(index=False))

    # 记录参数
    best = df.iloc[0]
    return {
        'ic_mean': round(mean_ic, 3),
        'ic_pos':  round(ic_pos, 3),
        'scan': df.to_dict('records'),
        'best_params': {
            'top_n':  int(best['top_n']),
            'roc_win': int(best['roc_win']),
            'vol_win': int(best['vol_win']),
        }
    }

# ─── Stage 2: Val — 选最优参数 ───────────────────────────────────────────────

def run_val(daily, weekly, train_result, tickers):
    print(f"\n{'='*60}")
    print(f"[Stage 2] Val {VAL_START} → {VAL_END}")
    print(f"{'='*60}")

    val_weekly = weekly[VAL_START:VAL_END]
    val_dates = val_weekly.index.tolist()
    print(f"  Val 周数: {len(val_dates)}")

    train_best = train_result['best_params']
    results = []

    for _, row in pd.DataFrame(train_result['scan']).iterrows():
        top_n  = int(row['top_n'])
        roc_w  = int(row['roc_win'])
        vol_w  = int(row['vol_win'])

        sig = compute_weekly_signals(daily, val_dates, roc_w, vol_w, top_n)
        bt = backtest(sig, weekly[VAL_START:VAL_END])
        if bt.empty:
            continue
        m = calc_metrics(bt)
        results.append({
            'top_n': top_n, 'roc_win': roc_w, 'vol_win': vol_w,
            'train_sharpe': row['sharpe'],
            **m
        })

    df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print(f"\n  Val Top5:")
    print(df.head(5)[['top_n','roc_win','vol_win','sharpe','ann_ret','max_dd','train_sharpe']].to_string(index=False))

    best = df.iloc[0]
    return {
        'sharpe': best['sharpe'],
        'best_params': {
            'top_n':  int(best['top_n']),
            'roc_win': int(best['roc_win']),
            'vol_win': int(best['vol_win']),
        },
        'all_results': df.to_dict('records'),
    }

# ─── Stage 3: Test — 锁死参数，只跑一次 ──────────────────────────────────────

def run_test(daily, weekly, val_result, tickers):
    print(f"\n{'='*60}")
    print(f"[Stage 3] Test {TEST_START} → {TEST_END}")
    print(f"{'='*60}")

    p = val_result['best_params']
    print(f"  锁死参数: top={p['top_n']}, roc={p['roc_win']}, vol={p['vol_win']}")

    test_weekly = weekly[TEST_START:TEST_END]
    test_dates = test_weekly.index.tolist()
    print(f"  Test 周数: {len(test_dates)}")

    sig = compute_weekly_signals(daily, test_dates, p['roc_win'], p['vol_win'], p['top_n'])
    bt_0 = backtest(sig, weekly[TEST_START:TEST_END], tc=0.0)
    bt_t = backtest(sig, weekly[TEST_START:TEST_END], tc=TC)

    m0 = calc_metrics(bt_0)
    mT = calc_metrics(bt_t)

    # SPY B&H
    spy = yf.download('SPY', start=TEST_START, end=TEST_END,
                      auto_adjust=False, progress=False)['Close'].squeeze()
    sw  = spy.resample('W').last().dropna()
    sr  = sw.pct_change().dropna()
    spy_sharpe = float(sr.mean()*52 / (sr.std()*math.sqrt(52)))
    spy_mdd    = float((sw/sw.cummax()).min()-1)
    spy_ann    = float(sr.mean()*52)

    print(f"\n  {'指标':<15} {'无TC':>10} {'20bps':>10} {'SPY B&H':>10}")
    print(f"  {'-'*47}")
    for k in ['sharpe','ann_ret','ann_vol','max_dd']:
        v0 = m0.get(k, '-')
        vT = mT.get(k, '-')
        vs = {'sharpe': round(spy_sharpe,2), 'ann_ret': round(spy_ann*100,1),
              'ann_vol': round(float(sr.std()*math.sqrt(52))*100,1),
              'max_dd': round(spy_mdd*100,1)}.get(k, '-')
        print(f"  {k:<15} {str(v0):>10} {str(vT):>10} {str(vs):>10}")

    # 年度明细
    ann = annual_detail(bt_t)
    print(f"\n  年度明细 (20bps):")
    print(f"  {'年份':<6} {'收益':>8} {'Sharpe':>8} {'换手':>6}")
    print(f"  {'-'*32}")
    for row in ann:
        sign = '+' if row['ret'] > 0 else ''
        print(f"  {row['year']:<6} {sign}{row['ret']:>6.1f}%  {row['sharpe']:>6.2f}  {row['n']}")

    return {
        'params': p,
        'metrics_0tc': m0,
        'metrics_20bps': mT,
        'annual': ann,
        'spy_sharpe': round(spy_sharpe, 2),
        'spy_ann_ret': round(spy_ann*100, 1),
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 读取 SPX ticker 列表
    with open('/tmp/spx_tickers.txt') as f:
        tickers = [l.strip() for l in f if l.strip()]

    print(f"SPX universe: {len(tickers)} 只")
    print(f"Train: {TRAIN_START}→{TRAIN_END}")
    print(f"Val:   {VAL_START}→{VAL_END}")
    print(f"Test:  {TEST_START}→{TEST_END}")

    # 下载数据（一次性下载，按需切片）
    # yfinance 批量下载 ticker 太多会失败，分批
    BATCH = 50
    all_close = {}
    failed = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        print(f"  下载 {i+1}-{min(i+BATCH, len(tickers))}...", end=" ", flush=True)
        try:
            df = yf.download(batch, start=DATA_START, end=DATA_END,
                             auto_adjust=False, progress=False)
            closes = df['Close'].dropna(how='all')
            for col in closes.columns:
                all_close[col] = closes[col]
            print(f"OK ({closes.shape[1]} 只)")
        except Exception as e:
            print(f"失败: {e}")
            failed.extend(batch)

    daily = pd.DataFrame(all_close)
    weekly = daily.resample('W').last().dropna(how='all')
    print(f"\n  总日线: {daily.shape}, 周线: {weekly.shape}")
    print(f"  下载失败: {len(failed)} 只")

    # 过滤：只保留有足够历史的股票（2011前有200+交易日）
    avail = daily.loc[:TRAIN_START].count()
    good_tickers = avail[avail >= 200].index.tolist()
    print(f"  过滤后（2011前有200+交易日）: {len(good_tickers)} 只")

    weekly = weekly[good_tickers].dropna(how='all')
    daily  = daily[good_tickers].dropna(how='all')
    print(f"  最终日线: {daily.shape}")
    print(f"  最终周线: {weekly.shape}")

    t0 = time.time()

    # Stage 1: Train
    train_result = run_train(daily, weekly, good_tickers)

    # Stage 2: Val
    val_result = run_val(daily, weekly, train_result, good_tickers)

    # Stage 3: Test
    test_result = run_test(daily, weekly, val_result, good_tickers)

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed/60:.1f} min")

    # 保存结果
    out = '/tmp/spx_momentum_results.json'
    result = {
        'universe': {'source': 'SPX constituents', 'n_tickers': len(good_tickers)},
        'train': train_result,
        'val':   val_result,
        'test':  test_result,
        'elapsed_min': round(elapsed/60, 1),
    }
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n✅ 结果 → {out}")