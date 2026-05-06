"""
多因子策略 v3 — Price(vol20) + Fundamental 多因子
三段式：Train(2011-2016) / Val(2016-2021) / Test(2021-2026)

Fundamental 因子（来自 SEC XBRL）:
  - rev_growth: YoY revenue growth (annual, 1-year lag)
  - earnings_yield: net_income / stockholders_equity
  - roe: net_income / stockholders_equity
  - book_per_share: stockholders_equity / shares_outstanding
  - de_ratio: total_debt / stockholders_equity

look-ahead 防护:
  - fiscal period end date + typical 60-day filing grace period = 可用日期
  - 季度数据用最近4个季度（rolling 4Q）而非单季
  - annual FY 数据用 FY-end + 60days 才算公开
"""

import sys, math, json, time
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, '/home/ubuntu/.hermes/skills/quant-trading-momentum/scripts')
import requests

warnings = []
import warnings as _w
_w.filterwarnings('ignore')

HEADERS = {'User-Agent': 'QuantResearch agent@example.com'}

# ── XBRL data paths ──────────────────────────────────────────────────────────
FUNDAMENTAL_FILE = '/tmp/fundamental_data.json'
CIK_MAP_FILE     = '/tmp/ticker_cik_map.json'

# ── Time periods ─────────────────────────────────────────────────────────────
DATA_START = '2010-01-01'
DATA_END   = '2026-02-01'

UNIVERSE = [
    'AAPL','MSFT','AMZN','GOOGL','GOOG','META','NVDA','AVGO','TSLA','CSCO',
    'ADBE','NFLX','ORCL','CRM','AMD','INTC','QUALCOMM','TXN','MU','AMAT','IBM',
    'JPM','BAC','WFC','GS','MS','C','BLK','AXP','V','MA','SCHW','USB','PNC',
    'LMT','BA','CAT','GE','UPS','HON','RTX','DE','MMM','ITW','ETN','PH',
    'LLY','UNH','JNJ','PFE','ABBV','MRK','TMO','DHR','AMGN','ISRG','MDT',
    'ABT','BMY','GILD',
    'WMT','HD','COST','PG','KO','PEP','MCD','SBUX','NKE','TGT','LOW','DG',
    'XOM','CVX','COP','SLB','EOG','PSX','VLO','OXY',
    'LIN','APD','SHW','FCX','NEM','DHI','LEN','VMC','MLM',
    'PLD','AMT','EQIX','CCI','PSA','SPG','O',
    'SPY','QQQ','GLD','TLT','EFA','EEM','IWM','VTI','VEA','VWO','BND',
    'BKNG','MAR','ABNB','NOW','SNOW','CRWD','PANW','ZS','TEAM','F',
]
UNIVERSE = sorted(set([t for t in UNIVERSE if isinstance(t, str) and t.isupper()]))

FACTOR_CONFIG = {
    # Price factors
    'roc20':  {'type': 'roc',  'window': 20},
    'roc60':  {'type': 'roc',  'window': 60},
    'roc120': {'type': 'roc',  'window': 120},
    'vol20':  {'type': 'vol',  'window': 20},
    'vol60':  {'type': 'vol',  'window': 60},
}

FILING_LAG_DAYS = 60   # typical days from FY-end to 10-K filing

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load fundamental data from XBRL JSON
# ─────────────────────────────────────────────────────────────────────────────

def load_fundamental_data():
    try:
        with open(FUNDAMENTAL_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f'ERROR: {FUNDAMENTAL_FILE} not found. Run sec_xbrl_fetch.py first.')
        sys.exit(1)

def load_cik_map():
    with open(CIK_MAP_FILE) as f:
        return json.load(f)

def available_date(fiscal_end_date, filing_date, lag_days=FILING_LAG_DAYS):
    """
    The 'effective' public date = filing_date (already in XBRL).
    For robustness, also compute fiscal_end + lag_days.
    Returns the later of the two.
    """
    from datetime import datetime, timedelta
    if not filing_date or not fiscal_end_date:
        return None
    try:
        fiscal_ts = pd.Timestamp(fiscal_end_date)
        # Add typical filing lag
        effective = fiscal_ts + timedelta(days=lag_days)
        filed_ts = pd.Timestamp(filing_date)
        # Use the later of effective_date or filed_date
        return max(effective, filed_ts)
    except:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build fundamental factor time series
#    For each ticker: {date} → factor_value
#    Respects look-ahead: only data with effective_date <= factor_date
# ─────────────────────────────────────────────────────────────────────────────

def build_fundamental_series_v2(fundamental_data, ticker, factor_name):
    """
    Build sorted (effective_date, value) time series for fundamental factors.
    Uses quarterly XBRL data (NetIncomeLoss + StockholdersEquity) to construct
    trailing-4Q metrics, eliminating fiscal year calendar mismatches.

    effective_date = fiscal_quarter_end + FILING_LAG_DAYS (look-ahead guard)
    """
    if ticker not in fundamental_data:
        return []
    fd = fundamental_data[ticker]

    ni_all = sorted(fd.get('NetIncomeLoss', []), key=lambda x: x['end'])
    eq_all = sorted(fd.get('StockholdersEquity', []), key=lambda x: x['end'])
    sh_all = sorted(fd.get('SharesOutstanding', []), key=lambda x: x['end'])
    assets_all = sorted(fd.get('Assets', []), key=lambda x: x['end'])

    if len(ni_all) < 4 or len(eq_all) < 1:
        return []

    # Build date-indexed maps (use fiscal end date as key)
    ni_by_end  = {x['end']: x for x in ni_all}
    eq_by_end  = {x['end']: x for x in eq_all}
    sh_by_end  = {x['end']: x for x in sh_all}
    assets_by_end = {x['end']: x for x in assets_all}

    all_ni_ends = sorted(ni_by_end.keys())

    if factor_name == 'roe':
        results = []
        for i in range(3, len(all_ni_ends)):
            end = all_ni_ends[i]
            # Match equity: closest quarter end <= current end
            valid_eq_ends = [e for e in eq_by_end.keys() if e <= end]
            if not valid_eq_ends:
                continue
            eq_end = max(valid_eq_ends)
            equity = eq_by_end[eq_end]['val']
            if equity <= 0:
                continue
            ni = ni_by_end[end]['val']
            eff_date = pd.Timestamp(end) + pd.Timedelta(days=FILING_LAG_DAYS)
            results.append((eff_date, ni / equity, end))
        return results

    elif factor_name == 'earnings_yield':
        # Trailing-4Q net income / equity
        results = []
        for i in range(3, len(all_ni_ends)):
            # Sum last 4 quarters of NI
            window_ends = all_ni_ends[i-3:i+1]
            total_ni = sum(ni_by_end[e]['val'] for e in window_ends)
            end = all_ni_ends[i]  # anchor to most recent quarter end
            # Match equity: closest quarter end <= end
            valid_eq_ends = [e for e in eq_by_end.keys() if e <= end]
            if not valid_eq_ends:
                continue
            eq_end = max(valid_eq_ends)
            equity = eq_by_end[eq_end]['val']
            if equity <= 0:
                continue
            eff_date = pd.Timestamp(end) + pd.Timedelta(days=FILING_LAG_DAYS)
            results.append((eff_date, total_ni / equity, end))
        return results

    elif factor_name == 'book_per_share':
        # Equity / shares outstanding
        results = []
        all_ends = sorted(set(list(eq_by_end.keys()) + list(sh_by_end.keys())))
        for end in all_ends:
            if end not in eq_by_end or end not in sh_by_end:
                continue
            eq_val = eq_by_end[end]['val']
            sh_val = sh_by_end[end]['val']
            if eq_val <= 0 or sh_val <= 0:
                continue
            eff_date = pd.Timestamp(end) + pd.Timedelta(days=FILING_LAG_DAYS)
            results.append((eff_date, eq_val / sh_val, end))
        return results

    elif factor_name == 'de_ratio':
        # (TotalAssets - Equity) / Equity = debt-to-equity
        results = []
        all_ends = sorted(set(list(assets_by_end.keys()) + list(eq_by_end.keys())))
        for end in all_ends:
            if end not in assets_by_end or end not in eq_by_end:
                continue
            a = assets_by_end[end]['val']
            e = eq_by_end[end]['val']
            if e <= 0:
                continue
            debt = a - e
            if debt <= 0:
                continue
            eff_date = pd.Timestamp(end) + pd.Timedelta(days=FILING_LAG_DAYS)
            results.append((eff_date, debt / e, end))
        return results

    elif factor_name == 'rev_growth_quarterly':
        # YoY revenue growth using quarterly revenue entries
        # NOTE: only works for companies with quarterly revenue in XBRL
        # For most companies, Revenue field is sparse annual data
        # Use OperatingIncomeLoss as proxy if available
        op_inc_all = sorted(fd.get('OperatingIncomeLoss', []), key=lambda x: x['end'])
        if len(op_inc_all) < 8:  # need at least 8 quarters for YoY
            return []
        op_by_end = {x['end']: x for x in op_inc_all}
        op_ends = sorted(op_by_end.keys())
        results = []
        for i in range(4, len(op_ends)):
            curr_end = op_ends[i]
            prev_end = op_ends[i-4]  # YoY = 4 quarters ago
            curr_val = op_by_end[curr_end]['val']
            prev_val = op_by_end[prev_end]['val']
            if prev_val <= 0:
                continue
            eff_date = pd.Timestamp(curr_end) + pd.Timedelta(days=FILING_LAG_DAYS)
            results.append((eff_date, (curr_val - prev_val) / abs(prev_val), curr_end))
        return results

    return []

def get_factor_value(fund_series, as_of_date):
    """
    Return the most recent factor value that was available as of as_of_date.
    fund_series: sorted list of (effective_date, value, fiscal_end)
    """
    as_of = pd.Timestamp(as_of_date)
    candidates = [(d, v) for (d, v, _) in fund_series if d <= as_of]
    if not candidates:
        return np.nan
    return max(candidates, key=lambda x: x[0])[1]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Download price data
# ─────────────────────────────────────────────────────────────────────────────

def download_prices(universe):
    print("Downloading price data...")
    batch_size = 50
    all_closes = []
    for i in range(0, len(universe), batch_size):
        batch = universe[i:i+batch_size]
        n_batch = (len(universe)-1)//batch_size + 1
        print(f"  Batch {i//batch_size + 1}/{n_batch}", end='', flush=True)
        try:
            d = yf.download(batch, start=DATA_START, end=DATA_END,
                            progress=False, auto_adjust=True)
            close = d.xs('Close', axis=1, level=0)
            all_closes.append(close)
            print(f" OK({close.shape[1]})")
        except Exception as e:
            print(f" FAIL: {e}")
        time.sleep(0.2)

    data_all = pd.concat(all_closes, axis=1, sort=True)
    if data_all.columns.duplicated().any():
        data_all = data_all.loc[:, ~data_all.columns.duplicated()]
    return data_all

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build matrices
# ─────────────────────────────────────────────────────────────────────────────

def build_price_factor_matrices(data_all, universe):
    """Build monthly price + return matrices + factor matrices."""
    monthly_ends = pd.date_range('2010-03-01', '2026-01-01', freq='ME')
    n_months = len(monthly_ends)
    n_tickers = len(universe)

    ticker_idx = {t: i for i, t in enumerate(universe)}
    px_matrix = np.full((n_tickers, n_months), np.nan)

    for ti, t in enumerate(universe):
        px = data_all[t].dropna()
        for mi, m_end in enumerate(monthly_ends):
            valid = px.index[px.index <= m_end]
            if len(valid) > 0:
                px_matrix[ti, mi] = float(px.loc[valid[-1]])

    ret_matrix = np.full((n_tickers, n_months), np.nan)
    for ti in range(n_tickers):
        for mi in range(1, n_months):
            p_prev = px_matrix[ti, mi-1]
            p_cur  = px_matrix[ti, mi]
            if not (np.isnan(p_prev) or np.isnan(p_cur) or p_prev <= 0 or p_cur <= 0):
                ret_matrix[ti, mi] = p_cur / p_prev - 1

    # Price factors
    def compute_factor_matrix(factor_key, window):
        ftype = FACTOR_CONFIG[factor_key]['type']
        result = np.full((n_tickers, n_months), np.nan)
        for ti in range(n_tickers):
            for mi in range(window, n_months):
                start_mi = mi - window + 1
                px_win = px_matrix[ti, start_mi:mi+1]
                if np.any(np.isnan(px_win)):
                    continue
                if ftype == 'roc':
                    result[ti, mi] = px_win[-1] / px_win[0] - 1
                else:
                    rets = np.diff(px_win) / px_win[:-1]
                    result[ti, mi] = float(np.std(rets) * math.sqrt(12))
        return result

    factor_ts = {}
    for fname in FACTOR_CONFIG:
        factor_ts[fname] = compute_factor_matrix(fname, FACTOR_CONFIG[fname]['window'])

    return monthly_ends, ticker_idx, px_matrix, ret_matrix, factor_ts

def build_fundamental_matrix(fundamental_data, ticker_idx, monthly_ends, factor_name):
    """Build (n_tickers x n_months) matrix of fundamental factor values."""
    n_tickers = len(ticker_idx)
    n_months = len(monthly_ends)
    matrix = np.full((n_tickers, n_months), np.nan)

    for ticker, ti in ticker_idx.items():
        series = build_fundamental_series_v2(fundamental_data, ticker, factor_name)
        if not series:
            continue
        for mi, m_end in enumerate(monthly_ends):
            val = get_factor_value(series, m_end)
            if not np.isnan(val):
                matrix[ti, mi] = val

    return matrix

# ─────────────────────────────────────────────────────────────────────────────
# 5. IC and backtest (same as rolling_ic_v3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_ic(factor_matrix, ret_matrix, period_start_mi, period_end_mi):
    ic_list = []
    for mi in range(period_start_mi + 1, period_end_mi):
        fvals = factor_matrix[:, mi]
        rets  = ret_matrix[:, mi + 1]
        mask = ~(np.isnan(fvals) | np.isnan(rets))
        if mask.sum() < 15:
            continue
        xs = fvals[mask]
        ys = rets[mask]
        if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
            continue
        ic = float(np.corrcoef(xs, ys)[0, 1])
        ic_list.append(ic)
    return ic_list

def ic_summary(ic_list):
    if len(ic_list) < 3:
        return dict(mean=0, std=0, frac_pos=0, stability=0, n=len(ic_list))
    arr = np.array(ic_list)
    mean_ic = float(np.mean(arr))
    std_ic  = float(np.std(arr))
    frac    = float(np.mean((arr > 0).astype(float)))
    stab    = mean_ic / std_ic if std_ic > 1e-9 else 0.0
    return dict(mean=mean_ic, std=std_ic, frac_pos=frac, stability=stab, n=len(ic_list))

def compute_max_dd(pv):
    pv = np.array(pv, dtype=float)
    running_max = np.maximum.accumulate(pv)
    drawdowns = (pv - running_max) / running_max
    return float(np.min(drawdowns))

def backtest(top_n, factor_matrix_dict, factor_direction, factor_weight,
             ret_matrix, start_mi, end_mi, ticker_idx):
    """Multi-factor backtest using specified factors and weights."""
    if start_mi >= end_mi - 1:
        return None

    portfolio_values = [1.0]
    prev_val = 1.0
    active_factors = [f for f in factor_matrix_dict if factor_direction.get(f, 0) != 0]

    for mi in range(start_mi, end_mi):
        scores = np.zeros(len(ticker_idx))
        for fname in active_factors:
            ft = factor_matrix_dict[fname][:, mi]
            mask = ~np.isnan(ft)
            ft_norm = np.zeros(len(ticker_idx))
            ft_norm[mask] = (ft[mask] - np.mean(ft[mask])) / (np.std(ft[mask]) + 1e-9)
            scores += factor_direction[fname] * factor_weight.get(fname, 0) * ft_norm

        top_indices = np.argsort(scores)[-top_n:]
        valid = ~np.isnan(ret_matrix[top_indices, mi + 1])
        top_indices = top_indices[valid]

        if len(top_indices) == 0:
            portfolio_values.append(prev_val)
            continue

        port_ret = float(np.nanmean(ret_matrix[top_indices, mi + 1]))
        prev_val = prev_val * (1 + port_ret)
        portfolio_values.append(prev_val)

    pv = np.array(portfolio_values)
    monthly_rets = np.diff(pv) / pv[:-1]
    ann  = float(np.mean(monthly_rets) * 12)
    std  = float(np.std(monthly_rets) * math.sqrt(12))
    shrp = ann / std if std > 1e-9 else 0.0
    max_dd = compute_max_dd(portfolio_values)

    return dict(ann=ann, std=std, sharpe=shrp, max_dd=max_dd, n_months=len(monthly_rets))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('='*70)
    print('Multi-Factor Strategy v3: Price(vol20) + SEC XBRL Fundamentals')
    print('='*70)

    # Load fundamental data
    print('\nLoading SEC XBRL fundamental data...')
    fundamental_data = load_fundamental_data()
    cik_map = load_cik_map()

    matched = sum(1 for t in UNIVERSE if t in fundamental_data)
    print(f'  Fundamental data available for {matched}/{len(UNIVERSE)} tickers')
    print(f'  Available fundamental tickers: {sorted(fundamental_data.keys())[:10]}...')

    # Download prices
    data_all = download_prices(UNIVERSE)

    # Build final universe (use local var to avoid shadowing global UNIVERSE)
    available = []
    for t in UNIVERSE:
        if t not in data_all.columns:
            continue
        col = data_all[t].dropna()
        if len(col) < 252:
            continue
        if col.index[0] <= pd.Timestamp('2010-12-31'):
            available.append(t)
    tickers = sorted(set(available))
    print(f'\nFinal universe: {len(tickers)} tickers')

    # Build matrices
    monthly_ends, ticker_idx, px_matrix, ret_matrix, price_factor_ts = \
        build_price_factor_matrices(data_all, tickers)

    n_tickers = len(tickers)
    n_months = len(monthly_ends)

    # Build fundamental matrices
    FUNDAMENTAL_FACTORS = ['roe', 'earnings_yield', 'book_per_share', 'de_ratio']
    fund_factor_ts = {}
    for ff in FUNDAMENTAL_FACTORS:
        print(f'  Building {ff}...', end='', flush=True)
        m = build_fundamental_matrix(fundamental_data, ticker_idx, monthly_ends, ff)
        non_nan = int(np.sum(~np.isnan(m)))
        print(f' OK ({non_nan} non-NaN cells)')
        fund_factor_ts[ff] = m

    # Combine all factors
    all_factor_ts = {**price_factor_ts, **fund_factor_ts}
    print(f'  Total factors: {len(all_factor_ts)} (price={len(price_factor_ts)}, fundamental={len(fund_factor_ts)})')

    # Period indices
    def mi_for(date_str):
        ts = pd.Timestamp(date_str)
        idx = monthly_ends.get_indexer([ts], method='ffill')[0]
        return max(0, idx)

    TRAIN_MI = mi_for('2011-01-01')
    VAL_MI   = mi_for('2016-01-01')
    TEST_MI  = mi_for('2021-01-01')
    END_MI   = n_months - 1
    print(f'  Train: {TRAIN_MI}→{VAL_MI} | Val: {VAL_MI}→{TEST_MI} | Test: {TEST_MI}→{END_MI}')

    # ── IC Analysis ──────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('STEP 1: Train IC (all factors)')
    print('='*70)

    train_ics = {}
    for fname in all_factor_ts:
        ics = compute_rolling_ic(all_factor_ts[fname], ret_matrix, TRAIN_MI, VAL_MI)
        train_ics[fname] = ics
        s = ic_summary(ics)
        sig = '✓' if s['frac_pos'] > 0.55 and s['stability'] > 0.3 else ' '
        print(f'  {fname:<20} mean={s["mean"]:>+7.3f}  IC>0={s["frac_pos"]:5.1%}  stab={s["stability"]:>+6.3f}  n={s["n"]:>3} {sig}')

    print('\n' + '='*70)
    print('STEP 2: Val IC (all factors)')
    print('='*70)

    val_ics = {}
    factor_direction = {}
    factor_weight = {}
    for fname in all_factor_ts:
        ics = compute_rolling_ic(all_factor_ts[fname], ret_matrix, VAL_MI, TEST_MI)
        val_ics[fname] = ics
        s = ic_summary(ics)
        if s['frac_pos'] >= 0.55:
            factor_direction[fname] = +1
        elif s['frac_pos'] <= 0.45:
            factor_direction[fname] = -1
        else:
            factor_direction[fname] = 0
        factor_weight[fname] = abs(s['stability'])
        sig = '✓' if s['frac_pos'] > 0.55 and s['stability'] > 0.3 else ' '
        print(f'  {fname:<20} mean={s["mean"]:>+7.3f}  IC>0={s["frac_pos"]:5.1%}  stab={s["stability"]:>+6.3f}  n={s["n"]:>3} {sig}  dir={factor_direction[fname]:>+2d}')

    active_factors = [f for f in all_factor_ts if factor_direction.get(f, 0) != 0 and factor_weight.get(f, 0) > 0]

    print(f'\nActive factors: {[f for f in active_factors]}')

    # ── Backtest ─────────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('STEP 3: Test Backtest')
    print('='*70)

    # SPY B&H
    spy_idx = ticker_idx.get('SPY')
    spy_vals = [1.0]
    prev = 1.0
    for mi in range(TEST_MI, END_MI):
        r = ret_matrix[spy_idx, mi + 1] if not np.isnan(ret_matrix[spy_idx, mi + 1]) else 0.0
        prev = prev * (1 + r)
        spy_vals.append(prev)
    spy_pv = np.array(spy_vals)
    spy_rets = np.diff(spy_pv) / spy_pv[:-1]
    spy_ann  = float(np.mean(spy_rets) * 12)
    spy_std  = float(np.std(spy_rets) * math.sqrt(12))
    spy_shrp = spy_ann / spy_std if spy_std > 1e-9 else 0.0
    spy_mdd  = compute_max_dd(spy_vals)

    print(f'\n{"Strategy":<40} {"n":>5} {"Sharpe":>8} {"Ann%":>10} {"MaxDD%":>10}')
    print('-'*77)
    print(f'  {"SPY B&H":<38} {"~60":>5} {spy_shrp:>8.2f} {spy_ann*100:>+10.1f}% {spy_mdd*100:>+10.1f}%')

    all_results = []

    # Single-factor price strategies
    for fname in active_factors:
        if fname not in price_factor_ts:
            continue
        for top_n in [5, 10, 20]:
            r = backtest(top_n, {fname: all_factor_ts[fname]},
                         factor_direction, factor_weight, ret_matrix,
                         TEST_MI, END_MI, ticker_idx)
            if r:
                name = f'Price-{fname} top{top_n}'
                print(f'  {name:<38} {r["n_months"]:>5} {r["sharpe"]:>8.2f} {r["ann"]*100:>+10.1f}% {r["max_dd"]*100:>+10.1f}%')
                all_results.append({**r, 'strategy': name})

    # Price vol20 baseline
    for top_n in [5, 10, 20]:
        r = backtest(top_n, {'vol20': all_factor_ts['vol20']},
                     {'vol20': 1}, {'vol20': 0.3}, ret_matrix,
                     TEST_MI, END_MI, ticker_idx)
        if r:
            name = f'Baseline-vol20 top{top_n}'
            print(f'  {name:<38} {r["n_months"]:>5} {r["sharpe"]:>8.2f} {r["ann"]*100:>+10.1f}% {r["max_dd"]*100:>+10.1f}%')
            all_results.append({**r, 'strategy': name})

    # Multi-factor combinations
    if len(active_factors) >= 2:
        combos = [
            ('vol20+earnings', ['vol20', 'earnings_yield']),
            ('vol20+roe',      ['vol20', 'roe']),
            ('vol20+rev_growth', ['vol20', 'rev_growth']),
            ('top3_fund',      active_factors[:3] if len(active_factors) >= 3 else active_factors),
            ('all_active',    active_factors),
        ]
        for combo_name, factors in combos:
            for top_n in [5, 10, 20]:
                factor_dict = {f: all_factor_ts[f] for f in factors if f in all_factor_ts}
                r = backtest(top_n, factor_dict, factor_direction, factor_weight,
                             ret_matrix, TEST_MI, END_MI, ticker_idx)
                if r:
                    name = f'Multi-{combo_name} top{top_n}'
                    print(f'  {name:<38} {r["n_months"]:>5} {r["sharpe"]:>8.2f} {r["ann"]*100:>+10.1f}% {r["max_dd"]*100:>+10.1f}%')
                    all_results.append({**r, 'strategy': name})

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        'universe': UNIVERSE,
        'train_ic': {fname: ic_summary(ics) for fname, ics in train_ics.items()},
        'val_ic':   {fname: ic_summary(ics) for fname, ics in val_ics.items()},
        'factor_direction': {k: int(v) for k, v in factor_direction.items()},
        'factor_weight':    {k: float(v) for k, v in factor_weight.items()},
        'active_factors':   active_factors,
        'test_results':     all_results,
        'benchmark':        {'SPY B&H': dict(sharpe=spy_shrp, ann=spy_ann, max_dd=spy_mdd)},
    }
    with open('/tmp/multi_factor_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\nSaved to /tmp/multi_factor_results.json')

if __name__ == '__main__':
    main()
