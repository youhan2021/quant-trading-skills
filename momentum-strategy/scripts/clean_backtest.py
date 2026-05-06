#!/usr/bin/env python3
"""
Clean momentum backtest — SPX 2011 historical constituents
验证版：用已知可靠的数据流，避免 yfinance MultiIndex 陷阱
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json, re, urllib.request, warnings
warnings.filterwarnings('ignore')

DATA_START = '2009-01-01'
DATA_END   = '2026-02-10'
TC = 0.002

# ------------------------------------------------------------------
# 1. 获取 Wikipedia 2011-01-01 S&P 500 成分股
# ------------------------------------------------------------------
def get_spx_2011():
    url = ('https://en.wikipedia.org/w/api.php?action=query'
           '&prop=revisions'
           '&titles=List_of_S%26P_500_companies'
           '&rvlimit=1'
           '&rvstart=2011-01-01T00:00:00Z'
           '&rvdir=older'
           '&rvprop=content'
           '&format=json')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    for page in data.get('query', {}).get('pages', {}).values():
        for rev in page.get('revisions', []):
            c = rev['*']
            nyse  = re.findall(r'\{\{NyseSymbol\|([A-Z]+)\}\}', c)
            nasdaq = re.findall(r'\{\{NasdaqSymbol\|([A-Z]+)\}\}', c)
            return sorted(set(nyse + nasdaq))
    return []

# ------------------------------------------------------------------
# 2. 清理 yfinance 返回：MultiIndex → plain DataFrame/Series
# ------------------------------------------------------------------
def yf_close(tickers, start, end):
    """下载收盘价，返回 {ticker: Series} dict"""
    df = yf.download(tickers, start=start, end=end,
                     auto_adjust=False, progress=False)
    if df.empty:
        return {}
    # MultiIndex: ('Close', 'AAPL') 等 → 用 df['Close'] 取收盘价单层 DataFrame
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']  # DataFrame, columns = ticker names, no MultiIndex
    # 强制 tz-naive
    if df.index.tz is not None:
        df = df.tz_localize(None)
    result = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 100:
            result[str(col)] = s
    return result

# ------------------------------------------------------------------
# 3. 选股
# ------------------------------------------------------------------
def select_top(week_df, roc_wks, vol_wks, top_n, date):
    """从周线 DataFrame 选动量 top_n"""
    dt = pd.Timestamp(date)
    lookback = max(roc_wks, vol_wks) * 10  # days
    start = dt - pd.Timedelta(days=lookback)
    sub = week_df[start:dt].dropna(how='all', axis=1)
    if sub.shape[0] < 2:
        return []
    roc = (sub.iloc[-1] / sub.iloc[0]) - 1
    ret = sub.pct_change().dropna()
    if len(ret) < vol_wks:
        return []
    vol = ret.rolling(vol_wks).std().iloc[-1]
    valid = (vol > 0) & (~vol.isna())
    roc = roc[valid]
    vol = vol[valid]
    if len(roc) < 1:
        return []
    score = roc / vol
    top = score.nlargest(top_n).index.tolist()
    if len(top) < top_n:
        top = score.nlargest(min(len(score), top_n)).index.tolist()
    return top

# ------------------------------------------------------------------
# 4. 回测
# ------------------------------------------------------------------
def backtest(tickers_dict, train_start, train_end, val_start, val_end,
              roc_wks=60, vol_wks=60, top_n=10, tc=0.002,
              test_start=None, test_end=None):
    """三段式回测（可选 Test 期）
    
    tickers_dict: {ticker: Series} — 日线收盘价
    """
    # 拼日线 DataFrame
    prices = pd.DataFrame(tickers_dict)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    week = prices.resample('W').last()
    
    periods = [('Train', train_start, train_end),
               ('Val',   val_start,   val_end)]
    if test_start and test_end:
        periods.append(('Test', test_start, test_end))
    
    results = {}
    for label, start, end in periods:
        # 每周再平衡
        ws = week[start:end].index
        equity = [1.0]
        holdings = {}
        last_dt = None
        
        for reb in ws:
            reb = pd.Timestamp(reb)
            top = select_top(week, roc_wks, vol_wks, top_n, reb)
            new_h = {t: 1.0/len(top) for t in top} if top else {}
            
            if last_dt and last_dt < reb:
                slice_p = prices[last_dt:reb]
                if len(slice_p) > 1:
                    chg = slice_p.pct_change().dropna()
                    # 持仓收益
                    port_r = sum(w * ((1+chg[t]).prod()-1)
                                for t, w in holdings.items() if t in chg.columns)
                    # 换手成本
                    turnover = sum(abs(new_h.get(t,0) - holdings.get(t,0))
                                   for t in set(list(holdings)+list(new_h)))
                    equity.append(equity[-1] * (1 + port_r - turnover * tc))
            
            holdings = new_h
            last_dt  = reb
        
        rets = pd.Series(equity[1:]) / pd.Series(equity[:-1]).values - 1
        if len(rets) < 2:
            sh = ar = vd = mx = 0.0
        else:
            ar = float((1+rets.mean())**52 - 1)
            vd = float(rets.std() * np.sqrt(52))
            sh = float(ar / vd) if vd > 0 else 0.0
            cum = (1+rets).cumprod()
            mx = float((cum/cum.cummax()-1).min())
        
        results[label] = {'sharpe': round(sh,2), 'ann_ret': round(ar*100,1),
                          'ann_vol': round(vd*100,1), 'max_dd': round(mx*100,1),
                          'trades': len(rets), 'equity': equity}
    
    return results

# ------------------------------------------------------------------
# 5. 主流程
# ------------------------------------------------------------------
def main():
    print("=" * 55)
    print("S&P 500 Historical Constituents Momentum — Clean Backtest")
    print("=" * 55)
    
    # 1. Universe
    spx = get_spx_2011()
    print(f"\nUniverse: {len(spx)} stocks @ 2011-01-01")
    
    # 2. 下载（分批，每批30只，避免超时）
    print("Downloading...")
    all_prices = {}
    BATCH = 30
    for i in range(0, len(spx), BATCH):
        batch = spx[i:i+BATCH]
        d = yf_close(batch, DATA_START, DATA_END)
        all_prices.update(d)
        print(f"  {i+len(batch)}/{len(spx)} done, {len(all_prices)} valid")
    
    print(f"Valid tickers: {len(all_prices)}")
    
    # 3. Grid search
    best = None
    best_sharpe = -999
    all_results = []
    
    for top_n in [5, 10]:
        for roc in [30, 60, 120]:
            for vol in [20, 60]:
                r = backtest(all_prices,
                            '2012-01-01', '2016-01-01',
                            '2016-01-01', '2021-01-01',
                            roc, vol, top_n, TC)
                r['params'] = {'top': top_n, 'roc': roc, 'vol': vol}
                all_results.append(r)
                print(f"  top={top_n} roc={roc} vol={vol} → "
                      f"Train sh={r['Train']['sharpe']:.2f} "
                      f"Val sh={r['Val']['sharpe']:.2f}")
                if r['Val']['sharpe'] > best_sharpe:
                    best_sharpe = r['Val']['sharpe']
                    best = r['params']
    
    print(f"\nBest Val params: top={best['top']} roc={best['roc']} vol={best['vol']}")
    print(f"Best Val Sharpe: {best_sharpe:.2f}")
    
    # 4. Test
    test_r = backtest(all_prices,
                      '2012-01-01', '2016-01-01',
                      '2016-01-01', '2021-01-01',
                      best['roc'], best['vol'], best['top'], TC,
                      test_start='2021-01-01', test_end='2026-02-01')
    
    # SPY benchmark
    spy_prices = yf_close(['SPY'], '2021-01-01', '2026-02-01')
    if spy_prices:
        spy_s = list(spy_prices.values())[0]
        spy_ret = spy_s.resample('W').last().pct_change().dropna()
        sr = float((1+spy_ret.mean())**52-1) / float(spy_ret.std()*np.sqrt(52))
        print(f"\nSPY B&H Sharpe: {sr:.2f}")
    
    print(f"\nTest (locked params: top={best['top']} roc={best['roc']} vol={best['vol']})")
    tr = test_r.get('Test', test_r)
    print(f"  Sharpe: {tr.get('sharpe', 'N/A')}")
    print(f"  Ann Ret: {tr.get('ann_ret', 'N/A')}%")
    print(f"  Ann Vol: {tr.get('ann_vol', 'N/A')}%")
    print(f"  Max DD:  {tr.get('max_dd', 'N/A')}%")
    print(f"  Trades:  {tr.get('trades', 'N/A')}")

if __name__ == '__main__':
    main()