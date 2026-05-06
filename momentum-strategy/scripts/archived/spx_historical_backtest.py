#!/usr/bin/env python3
"""
S&P 500 Historical Constituents Momentum Backtest
==================================================
Universe: Wikipedia S&P 500 snapshot @ 2011-01-01 (497 stocks)
          — 真正的 point-in-time constituents，不依赖"幸存者"
          
三段式: Train 2011-2016 | Val 2016-2021 | Test 2021-2026
严格无 info leak：参数在 Val 锁死，不在 Test 做选择
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

DATA_START = '2009-01-01'
DATA_END   = '2026-02-10'
TOP_N = 10
ROC_PERIOD = 60   # 60 trading days = ~12 weeks
VOL_PERIOD = 60   # 60-day volatility
TC = 0.002        # 20bps one-way

# ------------------------------------------------------------------
# 1. 加载 2011-01-01 S&P 500 成分股（Wikipedia 历史快照）
# ------------------------------------------------------------------
def get_spx_2011_tickers():
    """从 Wikipedia 2011-01-01 修订版获取 S&P 500 成分股列表"""
    import urllib.request, json, re
    
    url = ('https://en.wikipedia.org/w/api.php?action=query'
           '&prop=revisions'
           '&titles=List_of_S%26P_500_companies'
           '&rvlimit=1'
           '&rvstart=2011-01-01T00:00:00Z'
           '&rvdir=older'
           '&rvprop=content'
           '&format=json')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research)'})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    
    pages = data.get('query', {}).get('pages', {})
    for pid, page in pages.items():
        revs = page.get('revisions', [])
        if revs:
            content = revs[0]['*']
            nyse = re.findall(r'\{\{NyseSymbol\|([A-Z]+)\}\}', content)
            nasdaq = re.findall(r'\{\{NasdaqSymbol\|([A-Z]+)\}\}', content)
            tickers = sorted(set(nyse + nasdaq))
            return tickers
    return []

# ------------------------------------------------------------------
# 2. 下载数据（批量）
# ------------------------------------------------------------------
def download_data(tickers, start, end):
    """下载日线 + 周线，处理 yfinance 假数据问题"""
    print(f"  下载 {len(tickers)} 只股票日线...")
    
    # 分批下载，每批 50 只
    BATCH = 50
    all_daily = {}
    failed = []
    
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        try:
            df = yf.download(batch, start=start, end=end, 
                           auto_adjust=False, progress=False)
            if df.empty:
                continue
            closes = df['Close'].dropna(how='all')
            for t in closes.columns:
                s = closes[t].dropna()
                if len(s) > 100:  # 基本数据完整性检查
                    all_daily[t] = s
        except Exception as e:
            failed.extend(batch)
    
    if not all_daily:
        return pd.DataFrame(), pd.DataFrame()
    
    daily = pd.DataFrame(all_daily)
    
    # 检查 yfinance 假数据：某股票在 2011-01-01 之前不应该有数据
    # 如果一只股票 IPO 于 2011 后，但它在 yf 里显示有 2011 前数据 → 假
    fake_tickers = []
    for t in daily.columns:
        pre_2011 = daily.loc[:'2011-01-01', t].dropna()
        if len(pre_2011) > 200:  # 2011 前有 200+ 交易日 = 可疑
            # 额外验证：用 ticker.info 的 IPO 日期
            try:
                info = yf.Ticker(t).info
                ipo_date = info.get('firstTradeDateEpochGregorian')
                if ipo_date:
                    from datetime import datetime
                    ipo_ts = datetime.utcfromtimestamp(ipo_date)
                    if ipo_ts.year >= 2012:  # IPO 在 2012 年或之后 = 假数据
                        fake_tickers.append(t)
            except:
                pass
    
    if fake_tickers:
        print(f"  ⚠️  剔除 {len(fake_tickers)} 只 yfinance 假数据: {fake_tickers[:10]}")
        daily = daily.drop(columns=[t for t in fake_tickers if t in daily.columns])
    
    print(f"  最终 universe: {len(daily.columns)} 只（剔除 {len(fake_tickers)} 只假数据）")
    
    # 周线
    weekly = daily.resample('W').last()
    
    return daily, weekly

# ------------------------------------------------------------------
# 3. 选股 & 回测核心
# ------------------------------------------------------------------
def select_top_momentum(weekly, roc_period, vol_period, top_n, date, universe_mask=None):
    """给定截面日期，从 universe 里选动量最好的 top_n 只
    
    动态 warmup：自动扩展到足够计算 vol_period 的历史数据
    单位：vol_period 以"周"计，需要足够多的周数据点
    """
    date_ts = pd.Timestamp(date)
    # weekly 可能 tz-aware，统一强转 naive
    weekly_idx = weekly.index
    if hasattr(weekly_idx, 'tz') and weekly_idx.tz is not None:
        if date_ts.tz is not None:
            date_ts = date_ts.tz_convert(weekly_idx.tz).tz_localize(None)
        weekly = weekly.tz_localize(None)
        weekly_idx = weekly.index
    else:
        date_ts = date_ts.tz_localize(None)
    
    end_dt = date_ts
    # 单位都是"周"：roc_period 和 vol_period 表示所需周数
    # lookback 以天计：1周≈7天，需要 max(roc, vol) 周 × 7天 × 2(安全系数)
    min_weeks = max(roc_period, vol_period)
    start_dt = end_dt - pd.Timedelta(days=min_weeks * 7 * 2)
    
    hist = weekly.loc[start_dt:end_dt].dropna(how='all', axis=1)
    
    if universe_mask is not None:
        hist = hist[universe_mask & hist.columns]
    
    if len(hist.columns) < top_n:
        return [], 0.0
    
    # ROC: 过去 roc_period 的收益率
    roc = (hist.iloc[-1] / hist.iloc[0]) - 1
    
    # 波动率
    rets = hist.pct_change().dropna()
    if len(rets) < vol_period or rets.empty:
        return [], 0.0
    vol = rets.rolling(vol_period).std().iloc[-1]
    
    # 过滤波动率为 0 或 NaN 的
    valid = (vol > 0) & (~vol.isna())
    roc = roc[valid]
    vol = vol[valid]
    
    if len(roc) < top_n:
        return [], 0.0
    
    # 动量分数 = ROC / vol（信息率）
    score = roc / vol
    top = score.nlargest(top_n).index.tolist()
    
    # 如果 top_n 只不够，就用所有可用的（避免空仓）
    if len(top) < top_n:
        top = score.nlargest(len(score)).index.tolist()
    
    return top, score[top].mean() if top else 0.0

def backtest_period(daily, weekly, start, end, roc_period, vol_period, top_n, tc):
    """在 daily 数据上回测一周再平衡"""
    # 统一为 tz-naive，避免比较 tz-aware vs tz-naive 报错
    start_dt = pd.Timestamp(start).tz_localize(None)
    end_dt   = pd.Timestamp(end).tz_localize(None)
    
    # weekly index 如果是 tz-aware，转为 naive
    weekly_idx = weekly.index
    if hasattr(weekly_idx, 'tz') and weekly_idx.tz is not None:
        weekly = weekly.tz_localize(None)
        weekly_idx = weekly.index

    # daily 如果是 tz-aware，也统一为 naive
    daily_idx = daily.index
    if hasattr(daily_idx, 'tz') and daily_idx.tz is not None:
        daily = daily.tz_localize(None)
        daily_idx = daily.index

    equity = [1.0]
    dates  = []
    
    # 每周第一天再平衡
    ws_slice = weekly.loc[start_dt:end_dt]
    week_starts = ws_slice.index[::1]
    
    current_holdings = {}
    last_rebal_date   = None
    
    for reb_date in week_starts:
        # 强制转 naive，避免跨函数时区混乱
        reb_date = pd.Timestamp(reb_date)
        if reb_date.tz is not None:
            reb_date = reb_date.tz_localize(None)
        
        # 在 reb_date 选股
        top, score = select_top_momentum(weekly, roc_period, vol_period, top_n, reb_date)
        
        if not top:
            holdings = {}  # 极端情况：没有任何可用股票
        else:
            holdings = {t: 1.0 / len(top) for t in top}
        
        # 计算从 last_rebal 到 reb_date 的收益
        if last_rebal_date is not None and last_rebal_date < reb_date:
            ret_start = last_rebal_date
            ret_end   = reb_date
            
            if ret_end <= daily_idx[-1] and ret_start < ret_end:
                period_daily = daily.loc[ret_start:ret_end]
                if len(period_daily) > 1:
                    chg = period_daily.pct_change().dropna()
                    
                    # 持仓收益（空仓时 port_ret = 0）
                    port_ret = 0.0
                    for t, w in current_holdings.items():
                        if t in chg.columns:
                            w_ret = (1 + chg[t]).prod() - 1
                            port_ret += w * w_ret
                    
                    # 换手成本
                    new_set = set(holdings.keys())
                    old_set = set(current_holdings.keys())
                    turnover = sum(abs(holdings.get(t, 0) - current_holdings.get(t, 0)) 
                                   for t in set(list(old_set) + list(new_set)))
                    tc_cost  = turnover * tc
                    
                    period_ret = port_ret - tc_cost
                    equity.append(equity[-1] * (1 + period_ret))
                    dates.append(ret_end)
        
        current_holdings = holdings
        last_rebal_date  = reb_date
    
    if len(equity) < 2:
        return {
            'sharpe': 0.0, 'ann_ret': 0.0, 'ann_vol': 0.0,
            'max_dd': 0.0, 'trades': 0, 'returns': [],
            'equity_curve': equity
        }
    
    rets = pd.Series(equity[1:]) / pd.Series(equity[:-1]).values - 1
    ann_ret = (1 + rets.mean()) ** 52 - 1
    ann_vol = rets.std() * np.sqrt(52)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum      = (1 + rets).cumprod()
    max_dd   = (cum / cum.cummax() - 1).min()
    
    return {
        'sharpe':    round(sharpe, 2),
        'ann_ret':   round(ann_ret * 100, 1),
        'ann_vol':   round(ann_vol * 100, 1),
        'max_dd':    round(max_dd   * 100, 1),
        'trades':    int(len(rets)),
        'returns':   list(rets.values),
        'equity_curve': equity
    }

# ------------------------------------------------------------------
# 4. Grid Search（Train → Val）
# ------------------------------------------------------------------
def grid_search(daily, weekly, train_start, train_end, val_start, val_end):
    """在 Train 上选最优参数，Val 上验证"""
    param_grid = []
    for top   in [5, 10]:
        for roc  in [30, 60, 120]:
            for vol  in [20, 60]:
                param_grid.append({'top': top, 'roc': roc, 'vol': vol})
    
    print(f"\nGrid search: {len(param_grid)} 参数组合")
    print(f"Train: {train_start} → {train_end}")
    print(f"Val:   {val_start}   → {val_end}")
    
    train_results = []
    for p in param_grid:
        r = backtest_period(daily, weekly, train_start, train_end,
                           p['roc'], p['vol'], p['top'], TC)
        r.update(p)
        train_results.append(r)
    
    val_results = []
    for p in param_grid:
        r = backtest_period(daily, weekly, val_start, val_end,
                           p['roc'], p['vol'], p['top'], TC)
        r.update(p)
        val_results.append(r)
    
    # Val Sharpe 选最优
    val_results.sort(key=lambda x: x['sharpe'], reverse=True)
    best = val_results[0]
    
    print(f"\nVal Top5 (按 Sharpe):")
    print(f"  {'top':>3}  {'roc':>4}  {'vol':>4}  {'sharpe':>6}  {'ann_ret':>7}  {'max_dd':>7}  {'train_sh':>9}")
    for r in val_results[:5]:
        print(f"  {r['top']:>3}  {r['roc']:>4}  {r['vol']:>4}  {r['sharpe']:>6.2f}  {r['ann_ret']:>7.1f}%  {r['max_dd']:>7.1f}%  {r['sharpe']:>9.2f}")
    
    return best, train_results, val_results, param_grid

# ------------------------------------------------------------------
# 5. 主流程
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("S&P 500 Historical Constituents Momentum Backtest")
    print("Universe: Wikipedia @ 2011-01-01 (point-in-time)")
    print("=" * 60)
    
    # 1. 获取 universe
    spx_2011 = get_spx_2011_tickers()
    print(f"\nS&P 500 constituents @ 2011-01-01: {len(spx_2011)} 只")
    
    # 2. 下载数据
    daily, weekly = download_data(spx_2011, DATA_START, DATA_END)
    
    if daily.empty:
        print("ERROR: No data downloaded!")
        return
    
    print(f"日线: {daily.shape}, 周线: {weekly.shape}")
    
    # 3. Grid search
    # 注意：Train 从 2012-01-01 开始（给 vol_period=60 足够的 warmup history）
    best, train_results, val_results, param_grid = grid_search(
        daily, weekly,
        train_start='2012-01-01', train_end='2016-01-01',
        val_start ='2016-01-01',  val_end  ='2021-01-01'
    )
    
    # 4. Test（锁死 best 参数）
    print(f"\n锁死参数: top={best['top']}, roc={best['roc']}, vol={best['vol']}")
    
    test_tc0 = backtest_period(daily, weekly, '2021-01-01', '2026-02-01',
                              best['roc'], best['vol'], best['top'], 0.0)
    test_tc20 = backtest_period(daily, weekly, '2021-01-01', '2026-02-01',
                               best['roc'], best['vol'], best['top'], TC)
    
    # Benchmark: SPY
    try:
        spy = yf.download('SPY', start='2009-01-01', end='2026-02-10',
                         auto_adjust=False, progress=False)['Close'].dropna()
        # 强制 tz-naive
        if spy.index.tz is not None:
            spy.index = spy.index.tz_localize(None)
        spy_w = spy.resample('W').last()
        spy_ret = spy_w.pct_change().dropna()
        spy_equity = (1 + spy_ret).cumprod()
        # 取 Test 区间
        spy_test = spy_equity['2021-01-01':'2026-02-01'].dropna()
        if len(spy_test) > 2:
            spy_rets = spy_test.pct_change().dropna()
            m = spy_rets.mean()
            s = spy_rets.std()
            spy_ann_ret = float((1 + m) ** 52 - 1)
            spy_ann_vol = float(s * np.sqrt(52))
            spy_sharpe  = float(spy_ann_ret / spy_ann_vol) if spy_ann_vol > 0 else 0.0
            spy_max_dd  = float((spy_test / spy_test.cummax() - 1).min())
        else:
            spy_sharpe = spy_ann_ret = spy_ann_vol = spy_max_dd = 0.0
    except Exception as e:
        print(f"  SPY download error: {e}")
        spy_sharpe = spy_ann_ret = spy_ann_vol = spy_max_dd = 0.0
    
    # 5. 年度明细
    print(f"\n{'='*50}")
    print(f"Test 锁死参数: top={best['top']}, roc={best['roc']}, vol={best['vol']}")
    print(f"\n{'指标':<12}  {'无TC':>8}  {'20bps':>8}  {'SPY B&H':>8}")
    print(f"{'-'*50}")
    print(f"{'Sharpe':<12}  {test_tc0['sharpe']:>8.2f}  {test_tc20['sharpe']:>8.2f}  {spy_sharpe:>8.2f}")
    print(f"{'Ann Ret':<12}  {test_tc0['ann_ret']:>7.1f}%  {test_tc20['ann_ret']:>7.1f}%  {spy_ann_ret*100:>7.1f}%")
    print(f"{'Ann Vol':<12}  {test_tc0['ann_vol']:>7.1f}%  {test_tc20['ann_vol']:>7.1f}%  {spy_ann_vol*100:>7.1f}%")
    print(f"{'Max DD':<12}  {test_tc0['max_dd']:>7.1f}%  {test_tc20['max_dd']:>7.1f}%  {spy_max_dd*100:>7.1f}%")
    
    # 年度明细
    rets = pd.Series(test_tc20['returns'])
    dates = pd.date_range('2021-01-01', periods=len(rets), freq='W')
    rets.index = dates
    yearly = rets.groupby(rets.index.year).apply(lambda x: (1+x).prod()-1)
    
    print(f"\n年度明细 (20bps):")
    print(f"  {'年份':>6}  {'收益':>8}  {'Sharpe':>6}")
    for yr, ret in yearly.items():
        yr_rets = rets[rets.index.year == yr]
        yr_sharpe = (yr_rets.mean() * 52) / (yr_rets.std() * np.sqrt(52)) if yr_rets.std() > 0 else 0
        print(f"  {yr:>6}  {ret*100:>+7.1f}%  {yr_sharpe:>6.2f}")
    
    # 保存
    result = {
        'universe': 'Wikipedia S&P 500 @ 2011-01-01 (point-in-time)',
        'n_tickers': len(spx_2011),
        'best_params': {'top': best['top'], 'roc': best['roc'], 'vol': best['vol']},
        'train_sharpe': {f"top{p['top']}_roc{p['roc']}_vol{p['vol']}": r['sharpe'] 
                        for p, r in zip(param_grid, train_results)},
        'val_sharpe': {f"top{p['top']}_roc{p['roc']}_vol{p['vol']}": r['sharpe']
                      for p, r in zip(val_results[0].keys(), val_results)},
        'test_no_tc':   {k: v for k, v in test_tc0.items() if k != 'equity_curve'},
        'test_20bps':   {k: v for k, v in test_tc20.items() if k != 'equity_curve'},
        'yearly_returns': {str(k): round(v*100, 1) for k, v in yearly.items()},
        'spy_benchmark': {'sharpe': round(spy_sharpe, 2), 'ann_ret': round(spy_ann_ret*100, 1),
                         'ann_vol': round(spy_ann_vol*100, 1), 'max_dd': round(spy_max_dd*100, 1)}
    }
    
    with open('/tmp/spx_hist_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n✅ → /tmp/spx_hist_results.json")

if __name__ == '__main__':
    main()