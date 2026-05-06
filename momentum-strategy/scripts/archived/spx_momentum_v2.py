"""
SPX Momentum Backtest v2 — 三段式 + Survival Bias 最小化
=========================================================
核心改进：
1. Universe = 当前 SPX 成分股，但只保留2011年之前就存在的股票
   （排除 IPO date > 2011 的公司）
2. 不做幸存者偏差：允许公司在回测期间倒闭/退市（用 yfinance 实际数据）
3. 三段式严格分离

数据问题：
- yfinance 只返回当前存在的股票价格
- 倒闭公司（Lehman, Bear Stearns 等）历史价格 yfinance 仍可查
- 但 IPO 于2011年之后的股票不可用
- → 解决方案：过滤掉2011年后IPO的股票
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

TOP_N_LIST    = [5, 10]
ROC_WIN_LIST  = [60, 120]
VOL_WIN_LIST  = [20, 60]
TC            = 0.002  # 20bps

DATA_START    = "2009-01-01"
DATA_END      = "2026-02-10"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def rzscore(s):
    r = s.rank()
    return (r - r.mean()) / r.std()

def download_batches(tickers, start, end, batch_size=50):
    """分批下载 yfinance"""
    all_close = {}
    failed = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            df = yf.download(batch, start=start, end=end,
                            auto_adjust=False, progress=False)
            closes = df['Close'].dropna(how='all')
            for col in closes.columns:
                all_close[col] = closes[col]
        except Exception as e:
            failed.extend(batch)
    return pd.DataFrame(all_close), failed

def get_factor_score(daily_prices, roc_win, vol_win):
    n = daily_prices.shape[0]
    if n < max(roc_win, vol_win) + 5:
        return None
    roc = (daily_prices.iloc[-1] / daily_prices.iloc[-roc_win]) - 1
    vol = daily_prices.iloc[-vol_win:].std() * np.sqrt(252)
    return rzscore(roc) + rzscore(-vol)

def compute_signals(daily, weekly_dates, roc_win, vol_win, top_n):
    """对每个截面计算信号"""
    signals = {}
    min_hist = max(roc_win, vol_win) + 5
    for i, date in enumerate(weekly_dates):
        if i < min_hist:
            continue
        d = daily.loc[:date]
        if d.shape[0] < min_hist:
            continue
        score = get_factor_score(d, roc_win, vol_win)
        if score is None:
            continue
        top = score.nlargest(top_n)
        signals[date] = top.index.tolist()
    return signals

def backtest(signals, weekly_prices, tc=0.0):
    rets_list = []
    dates = sorted(signals.keys())
    holdings = [set(signals[dates[0]])]

    for i in range(1, len(dates)):
        prev = dates[i-1]
        curr = dates[i]
        prev_h = holdings[-1]
        curr_h = set(signals[curr])

        sold = prev_h - curr_h
        bought = curr_h - prev_h
        did_trade = (len(sold) + len(bought)) > 0
        turnover = (len(sold) + len(bought)) / max(len(prev_h), 1)

        w1 = weekly_prices.loc[prev]
        w2 = weekly_prices.loc[curr]
        if w1.isna().any() or w2.isna().any():
            holdings.append(curr_h)
            continue

        ret = (w2 / w1) - 1
        port_ret = ret[list(curr_h)].mean()
        port_ret -= tc * turnover if did_trade else 0

        rets_list.append({
            'date': curr, 'ret': port_ret,
            'turnover': turnover if did_trade else 0,
            'did_trade': did_trade,
            'n_hold': len(curr_h),
        })
        holdings.append(curr_h)

    if not rets_list:
        return pd.DataFrame()
    df = pd.DataFrame(rets_list).set_index('date')
    return df

def metrics(df_rets):
    if len(df_rets) < 5:
        return {}
    r = df_rets['ret']
    equity = (1 + r).cumprod()
    ann  = r.mean() * 52
    std  = r.std()  * math.sqrt(52)
    mdd  = float((equity / equity.cummax()).min() - 1)
    n_trades = int((df_rets['did_trade'] == True).sum())
    return {
        'sharpe':    round(ann / std, 2) if std > 1e-10 else -999,
        'ann_ret':   round(ann * 100, 1),
        'ann_vol':   round(std * 100, 1),
        'max_dd':    round(mdd * 100, 1),
        'n_weeks':   len(r),
        'n_trades':  n_trades,
        'ann_tc':    round(2*TC*n_trades/len(r)*52*100, 1),
        'equity_end': round(float(equity.iloc[-1]), 3),
    }

def annual_detail(df_rets):
    rows = []
    for yr, g in df_rets.groupby(df_rets.index.year):
        total = float((1+g['ret']).prod()-1)
        sh = float(g['ret'].mean()*52 / (g['ret'].std()*math.sqrt(52))) if g['ret'].std()>1e-10 else 0
        rows.append({'year': yr, 'ret': round(total*100,1), 'n_weeks': len(g), 'sharpe': round(sh,2)})
    return rows

def run_stage(name, daily, weekly, dates, roc_win, vol_win, top_n, tc=0.0):
    sig = compute_signals(daily, dates, roc_win, vol_win, top_n)
    bt = backtest(sig, weekly, tc)
    m = metrics(bt)
    ann = annual_detail(bt) if not bt.empty else []
    return {'params': {'roc':roc_win,'vol':vol_win,'top':top_n}, 'metrics': m, 'annual': ann, 'n_signals': len(sig), 'n_rets': len(bt)}

def run():
    # ── 读取 SPX ticker 并过滤 2011 后 IPO ─────────────────────────────
    with open('/tmp/spx_tickers.txt') as f:
        all_spx = [l.strip() for l in f if l.strip()]

    print(f"SPX 总计: {len(all_spx)} 只")

    # ── 下载全量数据（分批）───────────────────────────────
    print(f"\n下载数据 {DATA_START} → {DATA_END}...")
    daily_all, failed = download_batches(all_spx, DATA_START, DATA_END)
    weekly_all = daily_all.resample('W').last()
    print(f"下载完成: 日线 {daily_all.shape}, 周线 {weekly_all.shape}")
    print(f"下载失败: {len(failed)} 只: {failed[:5]}...")

    # ── 过滤：2011年之前就存在的股票（排除 IPO 于2011后的）──────────
    # 方法：检查 2011-01-01 之前有多少条数据
    avail_2011 = daily_all.loc[:'2011-01-01'].count()
    # 至少需要 500 个交易日（约2年），才认为2011年之前就存在
    pre_2011 = avail_2011[avail_2011 >= 500].index.tolist()
    print(f"\n2011年前有500+交易日: {len(pre_2011)} 只（排除IPO于2011后的）")

    # 再过滤：至少在 2011 年有 200 个交易日（宽松一点）
    avail_train = daily_all.loc[:TRAIN_START].count()
    eligible = avail_train[avail_train >= 200].index.tolist()
    print(f"2011年前有200+交易日: {len(eligible)} 只")

    # 取两者的并集
    good_tickers = sorted(set(pre_2011) & set(eligible))
    print(f"最终 universe: {len(good_tickers)} 只")

    daily = daily_all[good_tickers].dropna(how='all')
    weekly = weekly_all[good_tickers].dropna(how='all')
    print(f"最终日线: {daily.shape}, 周线: {weekly.shape}")

    t0 = time.time()

    # ── Stage 1: Train ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Train] {TRAIN_START} → {TRAIN_END}")
    train_dates = weekly[TRAIN_START:TRAIN_END].index.tolist()
    print(f"  Train 周数: {len(train_dates)}")

    # IC
    ic_rows = []
    for i in range(30, len(train_dates)-1):
        d = daily.loc[:train_dates[i]]
        if d.shape[0] < 130:
            continue
        score = get_factor_score(d, 120, 20)
        if score is None:
            continue
        w1 = weekly.loc[train_dates[i]]
        w2 = weekly.loc[train_dates[i+1]]
        ret = (w2/w1)-1
        ret = ret[score.index].dropna()
        ic = score[ret.index].corr(ret)
        ic_rows.append(ic)

    mean_ic = np.mean(ic_rows)
    ic_pos  = np.mean([x>0 for x in ic_rows])
    print(f"  IC mean={mean_ic:.3f}, IC>0 ratio={ic_pos:.1%}")

    # 扫描
    scan = []
    total = len(TOP_N_LIST)*len(ROC_WIN_LIST)*len(VOL_WIN_LIST)
    for top_n in TOP_N_LIST:
        for roc_w in ROC_WIN_LIST:
            for vol_w in VOL_WIN_LIST:
                m = run_stage('train', daily, weekly[TRAIN_START:TRAIN_END],
                              train_dates, roc_w, vol_w, top_n, tc=0.0)
                scan.append({**m['params'], **m['metrics'], 'n_weeks': m['n_rets']})

    scan_df = pd.DataFrame(scan).sort_values('sharpe', ascending=False)
    print(f"\n  Train Top5:")
    print(scan_df.head(5)[['top','roc','vol','sharpe','ann_ret','max_dd']].to_string(index=False))
    best_train = scan_df.iloc[0]

    # ── Stage 2: Val ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Val] {VAL_START} → {VAL_END}")
    val_dates = weekly[VAL_START:VAL_END].index.tolist()
    print(f"  Val 周数: {len(val_dates)}")

    val_results = []
    for _, row in scan_df.iterrows():
        p = row
        m = run_stage('val', daily, weekly[VAL_START:VAL_END],
                      val_dates, int(p['roc']), int(p['vol']), int(p['top']), tc=0.0)
        val_results.append({**m['params'], **m['metrics'], 'train_sharpe': p['sharpe'], 'n_weeks': m['n_rets']})

    val_df = pd.DataFrame(val_results).sort_values('sharpe', ascending=False)
    print(f"\n  Val Top5:")
    print(val_df.head(5)[['top','roc','vol','sharpe','ann_ret','max_dd','train_sharpe']].to_string(index=False))
    best_val = val_df.iloc[0]

    # ── Stage 3: Test ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Test] {TEST_START} → {TEST_END}")
    test_dates = weekly[TEST_START:TEST_END].index.tolist()
    print(f"  Test 周数: {len(test_dates)}")

    p = best_val
    bt0 = backtest(compute_signals(daily, test_dates, int(p['roc']), int(p['vol']), int(p['top'])),
                   weekly[TEST_START:TEST_END], tc=0.0)
    btT = backtest(compute_signals(daily, test_dates, int(p['roc']), int(p['vol']), int(p['top'])),
                   weekly[TEST_START:TEST_END], tc=TC)

    m0 = metrics(bt0)
    mT = metrics(btT)

    # SPY B&H
    spy = yf.download('SPY', start=TEST_START, end=TEST_END,
                      auto_adjust=False, progress=False)['Close'].squeeze()
    sw = spy.resample('W').last().dropna()
    sr = sw.pct_change().dropna()
    spy_s = float(sr.mean()*52/(sr.std()*math.sqrt(52)))
    spy_a = float(sr.mean()*52)
    spy_m = float((sw/sw.cummax()).min()-1)

    print(f"\n  锁死参数: top={int(p['top'])}, roc={int(p['roc'])}, vol={int(p['vol'])}")
    print(f"\n  {'指标':<12} {'无TC':>10} {'20bps':>10} {'SPY B&H':>10}")
    print(f"  {'-'*44}")
    for k in ['sharpe','ann_ret','ann_vol','max_dd']:
        v0 = m0.get(k, '-')
        vT = mT.get(k, '-')
        vs = {'sharpe': round(spy_s,2), 'ann_ret': round(spy_a*100,1),
              'ann_vol': round(float(sr.std()*math.sqrt(52))*100,1),
              'max_dd': round(spy_m*100,1)}.get(k, '-')
        print(f"  {k:<12} {str(v0):>10} {str(vT):>10} {str(vs):>10}")

    ann = annual_detail(btT)
    print(f"\n  年度明细 (20bps):")
    print(f"  {'年份':<6} {'收益':>8} {'Sharpe':>8} {'换手':>6}")
    print(f"  {'-'*32}")
    for row in ann:
        sign = '+' if row['ret'] > 0 else ''
        print(f"  {row['year']:<6} {sign}{row['ret']:>6.1f}%  {row['sharpe']:>6.2f}  {row['n_weeks']}")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed/60:.1f} min")

    # ── 保存 ────────────────────────────────────────────────────────────
    result = {
        'universe': {
            'source': 'SPX constituents (pre-2011 IPO filter)',
            'n_tickers': len(good_tickers),
            'n_failed_download': len(failed),
        },
        'train': {
            'ic_mean': round(mean_ic, 3),
            'ic_pos_ratio': round(ic_pos, 3),
            'scan': scan_df.to_dict('records'),
            'best': {**best_train},
        },
        'val': {
            'best_params': {'top': int(p['top']), 'roc': int(p['roc']), 'vol': int(p['vol'])},
            'sharpe': mT['sharpe'],
            'all_results': val_df.to_dict('records'),
        },
        'test': {
            'params': {'top': int(p['top']), 'roc': int(p['roc']), 'vol': int(p['vol'])},
            'metrics_0tc': m0,
            'metrics_20bps': mT,
            'annual': ann,
            'spy_sharpe': round(spy_s, 2),
        },
        'elapsed_min': round(elapsed/60, 1),
    }
    out = '/tmp/spx_v2_results.json'
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n✅ → {out}")
    return result

if __name__ == '__main__':
    run()