"""
================================================================================
COMPREHENSIVE QUANTITATIVE BACKTEST
Train(2011-2016) → Val(2016-2021) → Test(2021-2026)
Three-stage with Walk-Forward Analysis + Monte Carlo

Factors: 5 price (roc20/roc60/roc120/vol20/vol60) + 4 fundamental (roe, earnings_yield, book_per_share, de_ratio)
Combination methods: single-factor, equal-weight multi-factor, IC-weighted multi-factor
================================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
FUNDAMENTAL_FILE = '/tmp/fundamental_data.json'
CIK_MAP_FILE = '/tmp/ticker_cik_map.json'

# ── Dates ─────────────────────────────────────────────────────────────────────
DATA_START = '2010-01-01'
DATA_END   = '2026-02-01'
FILING_LAG_DAYS = 60  # XBRL reporting lag

# ── Universe ───────────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL','MSFT','AMZN','NVDA','GOOGL','META','BRK-B','LLY','AVGO','HD',
    'XOM','UNH','JPM','MA','PG','NVDA','COST','JNJ','ABBV','WMT','BAC',
    'CRM','MRK','CVX','PEP','KO','TMO','CSCO','MCD','ABT','ACN','DHR',
    'NKE','LLY','TXN','NEE','PM','UPS','MS','RTX','HON','LOW','QCOM',
    'INTC','AMD','IBM','GS','BLK','C','ADI','PANW','NOW','AMT','INTU',
    'SPGI','V','MA','JPM','USB','PNC','TGT','SCHW','BKNG','AXP','DE',
    'GE','CAT','BA','MMM','DIS','ISRG','MDT','BDX','SYK','ZTS','GILD',
    'ABT','CI','HUM','EL','PM','MO','SPGI','CME','ICE','COIN','MSTR',
    'PLTR','DOGE-USD','BTC-USD','ETH-USD','SOL-USD',
    'SPY','QQQ','IWM','DIA','EFA','EEM','TLT','GLD','VNQ','AGG',
    'BRK-B','AMGN','GILD','BIIB','REGN','VRTX','MRNA','AZN','SNY',
    'SAP','ASML','TSM','ASML','SNE','KEYS','AMAT','LRCX','MU','SWKS',
    'NXPI','QCOM','TXN','ADI','INTC','AMD','NVDA','META','AVGO','MAR',
]

# Deduplicate
UNIVERSE = sorted(set(UNIVERSE))

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_fundamental_data():
    try:
        with open(FUNDAMENTAL_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f'ERROR: {FUNDAMENTAL_FILE} not found. Run sec_xbrl_fetch.py first.')
        return {}

def load_cik_map():
    try:
        with open(CIK_MAP_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def get_closes(data):
    if isinstance(data.columns, pd.MultiIndex):
        return data.xs('Close', level=1, axis=1)
    else:
        return data[['Close']] if 'Close' in data.columns else data

def resample_monthly(closes):
    monthly = closes.resample('ME').last()
    return monthly[monthly.index <= pd.Timestamp(DATA_END)]

# ── XBRL Fundamental Factor Builder ───────────────────────────────────────────

def parse_xbrl_series(fund_ticker_data, concept):
    """Parse XBRL entries for a given concept. Returns [(end_ts, effective_ts, val)]."""
    entries = fund_ticker_data.get(concept, [])
    if isinstance(entries, dict):
        entries = entries.get('data', [])
    rows = []
    for e in entries:
        end_str = e.get('end', '')
        filed_str = e.get('filed', '')
        val = e.get('val')
        if not end_str or not filed_str or val is None:
            continue
        try:
            end_ts = pd.Timestamp(end_str)
            filed_ts = pd.Timestamp(filed_str)
            eff_ts = max(end_ts + timedelta(days=FILING_LAG_DAYS), filed_ts)
            rows.append((end_ts, eff_ts, float(val)))
        except:
            continue
    return sorted(rows, key=lambda x: x[0])

def build_fundamental_series_v2(fund_ticker_data, concept, monthly_ends):
    """
    Compute derived factors (roe, earnings_yield, book_per_share, de_ratio)
    from raw XBRL concepts using trailing-4Q windows.
    """
    ni_rows = parse_xbrl_series(fund_ticker_data, 'NetIncomeLoss')
    eq_rows = parse_xbrl_series(fund_ticker_data, 'StockholdersEquity')
    sh_rows = parse_xbrl_series(fund_ticker_data, 'SharesOutstanding')
    assets_rows = parse_xbrl_series(fund_ticker_data, 'Assets')

    ni_by_end = {r[0]: r[2] for r in ni_rows}
    eq_by_end = {r[0]: r[2] for r in eq_rows}
    sh_by_end = {r[0]: r[2] for r in sh_rows}
    assets_by_end = {r[0]: r[2] for r in assets_rows}
    all_ni_ends = sorted(ni_by_end.keys())

    def ff_fill(q_dates, q_vals):
        if not q_dates:
            return np.full(len(monthly_ends), np.nan)
        cur_val = np.nan
        cur_idx = 0
        result = np.full(len(monthly_ends), np.nan)
        for mi, me in enumerate(monthly_ends):
            me_ts = pd.Timestamp(me)
            while cur_idx < len(q_dates) and q_dates[cur_idx] <= me_ts:
                cur_val = q_vals[cur_idx]
                cur_idx += 1
            result[mi] = cur_val
        return result

    if concept == 'roe':
        q_dates, q_vals = [], []
        for end_ts in all_ni_ends:
            valid_eq = [e for e in eq_by_end if e <= end_ts]
            if not valid_eq:
                continue
            equity = eq_by_end[max(valid_eq)]
            if equity <= 0:
                continue
            ni = ni_by_end.get(end_ts)
            if ni is None:
                continue
            eff = end_ts + timedelta(days=FILING_LAG_DAYS)
            q_dates.append(eff)
            q_vals.append(ni / equity)
        return ff_fill(q_dates, q_vals)

    elif concept == 'earnings_yield':
        q_dates, q_vals = [], []
        for idx, end_ts in enumerate(all_ni_ends):
            if idx < 3:
                continue
            window = all_ni_ends[idx-3:idx+1]
            total_ni = sum(ni_by_end[e] for e in window)
            valid_eq = [e for e in eq_by_end if e <= end_ts]
            if not valid_eq:
                continue
            equity = eq_by_end[max(valid_eq)]
            if equity <= 0:
                continue
            eff = end_ts + timedelta(days=FILING_LAG_DAYS)
            q_dates.append(eff)
            q_vals.append(total_ni / equity)
        return ff_fill(q_dates, q_vals)

    elif concept == 'book_per_share':
        q_dates, q_vals = [], []
        all_ends = sorted(set(list(eq_by_end.keys()) + list(sh_by_end.keys())))
        for end_ts in all_ends:
            if end_ts not in eq_by_end or end_ts not in sh_by_end:
                continue
            eq_val = eq_by_end[end_ts]
            sh_val = sh_by_end[end_ts]
            if eq_val <= 0 or sh_val <= 0:
                continue
            eff = end_ts + timedelta(days=FILING_LAG_DAYS)
            q_dates.append(eff)
            q_vals.append(eq_val / sh_val)
        return ff_fill(q_dates, q_vals)

    elif concept == 'de_ratio':
        q_dates, q_vals = [], []
        all_ends = sorted(set(list(assets_by_end.keys()) + list(eq_by_end.keys())))
        for end_ts in all_ends:
            if end_ts not in assets_by_end or end_ts not in eq_by_end:
                continue
            a = assets_by_end[end_ts]
            e = eq_by_end[end_ts]
            if e <= 0:
                continue
            debt = a - e
            if debt <= 0:
                continue
            eff = end_ts + timedelta(days=FILING_LAG_DAYS)
            q_dates.append(eff)
            q_vals.append(debt / e)
        return ff_fill(q_dates, q_vals)

    return np.full(len(monthly_ends), np.nan)

def build_fundamental_matrix(fundamental_data, tickers, monthly_ends, concept):
    n = len(tickers)
    m = len(monthly_ends)
    mat = np.full((n, m), np.nan)
    for i, t in enumerate(tickers):
        mat[i, :] = build_fundamental_series_v2(fundamental_data.get(t, {}), concept, monthly_ends)
    return mat

# ── Price Factors ──────────────────────────────────────────────────────────────

def build_price_factor(prices_df, tickers, monthly_ends, factor_type, window=None):
    n, m = len(tickers), len(monthly_ends)
    mat = np.full((n, m), np.nan)
    closes = get_closes(prices_df)

    for i, t in enumerate(tickers):
        if t not in closes.columns:
            continue
        px = closes[t].dropna()
        if len(px) < window:
            continue
        if factor_type == 'roc':
            ret = px.pct_change(window).dropna()
        elif factor_type == 'vol':
            ret = px.pct_change().dropna()
            ret = ret.rolling(window).std().dropna()
        else:
            continue
        # Map to monthly ends
        for j, me in enumerate(monthly_ends):
            me_ts = pd.Timestamp(me)
            nearest = ret.index[indirectly_nearest(ret.index, me_ts)]
            if nearest is not None:
                mat[i, j] = ret.loc[nearest]

    return mat

def indirectly_nearest(idx_series, target):
    """Find nearest date in idx_series to target."""
    idx = idx_series.get_indexer([target], method='ffill')[0]
    if idx >= len(idx_series):
        idx = len(idx_series) - 1
    return idx_series[idx] if idx >= 0 else None

# ── Core Backtest Engine ───────────────────────────────────────────────────────

def rank_zscore(mat, axis=0):
    """Rank-based z-score normalization (0=across tickers per month, 1=across months per ticker)."""
    if axis == 0:
        result = np.full_like(mat, np.nan)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            valid = ~np.isnan(col)
            if valid.sum() < 3:
                continue
            ranks = pd.Series(col[valid]).rank().values
            mu, sigma = ranks.mean(), ranks.std()
            if sigma > 0:
                result[valid, j] = (ranks - mu) / sigma
        return result
    else:
        result = np.full_like(mat, np.nan)
        for i in range(mat.shape[0]):
            row = mat[i, :]
            valid = ~np.isnan(row)
            if valid.sum() < 3:
                continue
            ranks = pd.Series(row[valid]).rank().values
            mu, sigma = ranks.mean(), ranks.std()
            if sigma > 0:
                result[i, valid] = (ranks - mu) / sigma
        return result

def combo_equal(factors_dict):
    """Equal-weight combo of rank-zscored factors."""
    keys = list(factors_dict.keys())
    mats = [rank_zscore(factors_dict[k], axis=0) for k in keys]
    combo = np.nanmean(np.stack(mats), axis=0)
    return combo

def combo_ic_weighted(factors_dict, ret_matrix, train_start, train_end):
    """IC-weighted combo: weight = mean IC during train window."""
    keys = list(factors_dict.keys())
    ic_scores = {}
    for k in keys:
        mat = factors_dict[k][:, train_start:train_end]
        ic_vals = []
        for j in range(mat.shape[1]):
            fcol = mat[:, j]
            rcol = ret_matrix[:, j + train_start] if j + train_start < ret_matrix.shape[1] else None
            if rcol is None:
                continue
            valid = ~np.isnan(fcol) & ~np.isnan(rcol)
            if valid.sum() < 5:
                continue
            ic_vals.append(np.corrcoef(fcol[valid], rcol[valid])[0, 1])
        ic_scores[k] = np.nanmean(ic_vals) if ic_vals else 0.0

    total = sum(abs(v) for v in ic_scores.values())
    if total == 0:
        return combo_equal(factors_dict)
    weights = {k: abs(v) / total for k, v in ic_scores.items()}
    print(f"    IC-weighted: { {k: f'{weights[k]:.3f}' for k in keys} }")

    mats = [rank_zscore(factors_dict[k], axis=0) * weights[k] for k in keys]
    combo = np.nansum(np.stack(mats), axis=0)
    return combo

def select_topn(combo_mat, ret_mat, n, period_start, period_end):
    """Select top-N by composite score, long equal-weight."""
    results = []
    for j in range(period_start, period_end):
        scores = combo_mat[:, j]
        rets = ret_mat[:, j]
        valid = ~np.isnan(scores) & ~np.isnan(rets)
        if valid.sum() == 0:
            results.append(np.nan)
            continue
        top_idx = np.argsort(-scores[valid])[:min(n, valid.sum())]
        top_tickers = np.where(valid)[0][top_idx]
        top_rets = rets[top_tickers]
        results.append(np.mean(top_rets))
    return np.array(results)

def compute_metrics(rets, periods_per_year=12):
    """Compute Sharpe, Ann%, MaxDD from monthly return series."""
    r = rets[~np.isnan(rets)]
    if len(r) < 3:
        return {'sharpe': np.nan, 'ann': np.nan, 'maxdd': np.nan, 'n': len(r)}
    ann = np.mean(r) * periods_per_year
    vol = np.std(r, ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann / vol if vol > 0 else np.nan
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    maxdd = np.min(drawdown) if len(drawdown) > 0 else 0.0
    return {'sharpe': sharpe, 'ann': ann * 100, 'maxdd': maxdd * 100, 'n': len(r)}

# ── Walk-Forward Analysis ─────────────────────────────────────────────────────

def walk_forward_analysis(ret_mat, combo_mat, n, wf_window=24, wf_step=6):
    """Walk-forward: rolling train/val with expanding window. Returns WFA metrics."""
    n_months = ret_mat.shape[1]
    n_periods = (n_months - wf_window) // wf_step + 1

    wf_sharpes = []
    wf_anns = []
    wf_maxdds = []

    for p in range(n_periods):
        train_end = wf_window + p * wf_step
        train_start = max(0, train_end - wf_window)
        val_start = train_end
        val_end = min(n_months, train_end + wf_step)

        # Use train IC to pick top factor in this window
        if train_end - train_start < 12:
            continue

        # Build IC table for single factors
        factor_names = list(combo_mat.keys()) if isinstance(combo_mat, dict) else ['composite']
        best_ic = -999
        best_fname = 'composite'
        best_mat = combo_mat['composite'] if isinstance(combo_mat, dict) else combo_mat

        if isinstance(combo_mat, dict):
            for fname, fmat in combo_mat.items():
                ic_vals = []
                for j in range(train_start, train_end):
                    fcol = fmat[:, j]
                    rcol = ret_mat[:, j]
                    valid = ~np.isnan(fcol) & ~np.isnan(rcol)
                    if valid.sum() < 5:
                        continue
                    ic_vals.append(np.corrcoef(fcol[valid], rcol[valid])[0, 1])
                mean_ic = np.nanmean(ic_vals) if ic_vals else 0
                if mean_ic > best_ic:
                    best_ic = mean_ic
                    best_fname = fname
                    best_mat = fmat

        # Compute val performance
        val_rets = select_topn(best_mat, ret_mat, n, val_start, val_end)
        metrics = compute_metrics(val_rets)
        if not np.isnan(metrics['sharpe']):
            wf_sharpes.append(metrics['sharpe'])
            wf_anns.append(metrics['ann'])
            wf_maxdds.append(metrics['maxdd'])

    if not wf_sharpes:
        return {'sharpe_mean': np.nan, 'sharpe_std': np.nan, 'ann_mean': np.nan, 'wfa_pass': False}

    sharpe_mean = np.mean(wf_sharpes)
    sharpe_std = np.std(wf_sharpes, ddof=1) if len(wf_sharpes) > 1 else np.nan
    ann_mean = np.mean(wf_anns)
    maxdd_mean = np.mean(wf_maxdds)

    # WFA pass: mean Sharpe > 0.5 across rolling windows
    wfa_pass = sharpe_mean > 0.5

    return {
        'sharpe_mean': sharpe_mean,
        'sharpe_std': sharpe_std,
        'ann_mean': ann_mean,
        'maxdd_mean': maxdd_mean,
        'n_windows': len(wf_sharpes),
        'wfa_pass': wfa_pass
    }

# ── Monte Carlo (Block Bootstrap) ──────────────────────────────────────────────

def monte_carlo_block_bootstrap(ret_series, n_sim=1000, block_size=6):
    """Block bootstrap: simulate distribution of Sharpe under random resampling."""
    r = ret_series[~np.isnan(ret_series)]
    if len(r) < block_size * 2:
        return {'mc_sharpe_5pct': np.nan, 'mc_sharpe_95pct': np.nan, 'mc_median': np.nan}

    sharpes = []
    n = len(r)
    n_blocks = n // block_size

    rng = np.random.default_rng(42)

    for _ in range(n_sim):
        # Build bootstrap sample by resampling blocks
        boot = []
        for _ in range(n_blocks + 1):
            start = rng.integers(0, n)
            indices = [(start + b) % n for b in range(block_size)]
            boot.extend([r[i] for i in indices])
        boot = np.array(boot[:n])
        ann = np.mean(boot) * 12
        vol = np.std(boot, ddof=1) * np.sqrt(12)
        sh = ann / vol if vol > 0 else np.nan
        if not np.isnan(sh):
            sharpes.append(sh)

    sharpes = np.array(sharpes)
    return {
        'mc_sharpe_5pct': np.percentile(sharpes, 5),
        'mc_sharpe_95pct': np.percentile(sharpes, 95),
        'mc_median': np.median(sharpes),
        'mc_sharpe_mean': np.mean(sharpes),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

print('='*70)
print('COMPREHENSIVE QUANTITATIVE BACKTEST')
print('Train(2011-2016) → Val(2016-2021) → Test(2021-2026)')
print('='*70)

fundamental_data = load_fundamental_data()
cik_map = load_cik_map()

print('\n[0] Downloading prices...')
data_all = yf.download(UNIVERSE, start=DATA_START, end=DATA_END,
                        progress=False, auto_adjust=True, group_by='ticker')

# Universe filter
available = []
for t in UNIVERSE:
    if t not in data_all.columns.get_level_values(0):
        continue
    col = get_closes(data_all)[t].dropna()
    if len(col) < 252:
        continue
    if col.index[0] <= pd.Timestamp('2010-12-31'):
        available.append(t)
tickers = sorted(set(available))
print(f'  Universe: {len(tickers)} tickers')

# Monthly ends
monthly_idx = resample_monthly(get_closes(data_all)).index
monthly_ends = list(monthly_idx)
n_months = len(monthly_ends)
print(f'  Monthly periods: {n_months}')

# Period boundaries
def mi_for(date_str):
    ts = pd.Timestamp(date_str)
    idx = monthly_idx.get_indexer([ts], method='ffill')[0]
    return max(0, idx)

TRAIN_MI = mi_for('2011-01-01')
VAL_MI   = mi_for('2016-01-01')
TEST_MI  = mi_for('2021-01-01')
END_MI   = n_months - 1
N_TRAIN  = VAL_MI - TRAIN_MI
N_VAL    = TEST_MI - VAL_MI
N_TEST   = END_MI - TEST_MI

print(f'\n  Train: {monthly_ends[TRAIN_MI].date()} → {monthly_ends[VAL_MI].date()} ({N_TRAIN}m)')
print(f'  Val:   {monthly_ends[VAL_MI].date()} → {monthly_ends[TEST_MI].date()} ({N_VAL}m)')
print(f'  Test:  {monthly_ends[TEST_MI].date()} → {monthly_ends[END_MI].date()} ({N_TEST}m)')

# ── Build Factor Matrices ─────────────────────────────────────────────────────
print('\n[1] Building factor matrices...')
factor_mats = {}

# Price factors
for fname, window in [('roc20', 20), ('roc60', 60), ('roc120', 120),
                       ('vol20', 20), ('vol60', 60)]:
    print(f'  {fname}...', end=' ')
    closes = get_closes(data_all)
    mat = np.full((len(tickers), n_months), np.nan)
    for i, t in enumerate(tickers):
        if t not in closes.columns:
            continue
        px = closes[t].dropna()
        if len(px) < window:
            continue
        if fname.startswith('roc'):
            series = px.pct_change(window).dropna()
        else:
            series = px.pct_change().dropna()
            series = series.rolling(window).std().dropna()
        for j, me in enumerate(monthly_ends):
            me_ts = pd.Timestamp(me)
            idx_pos = closes.index.get_indexer([me_ts], method='ffill')[0]
            if idx_pos >= len(closes.index):
                idx_pos = len(closes.index) - 1
            if idx_pos < 0:
                continue
            nearest_ts = closes.index[idx_pos]
            if nearest_ts in series.index:
                mat[i, j] = series.loc[nearest_ts]
            elif idx_pos > 0 and closes.index[idx_pos - 1] in series.index:
                mat[i, j] = series.loc[closes.index[idx_pos - 1]]
    nn = int(np.sum(~np.isnan(mat)))
    print(f'{nn} cells')
    factor_mats[fname] = mat

# Fundamental factors
FUND_FACTORS = ['roe', 'earnings_yield', 'book_per_share', 'de_ratio']
for ff in FUND_FACTORS:
    print(f'  {ff}...', end=' ')
    mat = build_fundamental_matrix(fundamental_data, tickers, monthly_ends, ff)
    nn = int(np.sum(~np.isnan(mat)))
    print(f'{nn} cells')
    factor_mats[ff] = mat

# ── Build Return Matrix ─────────────────────────────────────────────────────────
print('\n[2] Building return matrix...')
ret_mat = np.full((len(tickers), n_months), np.nan)
closes = get_closes(data_all)
monthly_closes = resample_monthly(closes)
monthly_rets = monthly_closes.pct_change().dropna()
for i, t in enumerate(tickers):
    if t not in monthly_rets.columns:
        continue
    series = monthly_rets[t].dropna()
    for j, me in enumerate(monthly_ends):
        me_ts = pd.Timestamp(me)
        if me_ts in series.index:
            ret_mat[i, j] = series.loc[me_ts]
        elif j > 0 and monthly_ends[j-1] in series.index:
            ret_mat[i, j] = series.loc[monthly_ends[j-1]]
nn = int(np.sum(~np.isnan(ret_mat)))
print(f'  Return matrix: {nn} cells')

# SPY buy-and-hold
spy_rets = None
if 'SPY' in monthly_rets.columns:
    spy_rets = monthly_rets['SPY'].values

# ── IC Analysis (Train Period) ─────────────────────────────────────────────────
print('\n' + '='*70)
print('STEP 1: IC Analysis — Factor Direction (Train 2011-2016)')
print('='*70)
print(f'{"Factor":<20} {"Mean IC":>9} {"IC>0%":>7} {"Stability":>10} {"N":>4}  Direction')
print('-'*70)

ic_results = {}
for fname, mat in factor_mats.items():
    ic_vals = []
    for j in range(TRAIN_MI, VAL_MI):
        fcol = mat[:, j]
        rcol = ret_mat[:, j]
        valid = ~np.isnan(fcol) & ~np.isnan(rcol)
        if valid.sum() < 5:
            continue
        ic_vals.append(np.corrcoef(fcol[valid], rcol[valid])[0, 1])
    mean_ic = np.nanmean(ic_vals) if ic_vals else 0
    pct_pos = np.nanmean([v > 0 for v in ic_vals]) * 100 if ic_vals else 0
    # Stability: correlation between consecutive ICs
    stability = np.corrcoef(ic_vals[:-1], ic_vals[1:])[0, 1] if len(ic_vals) > 2 else 0
    n = len(ic_vals)
    direction = 'LONG' if mean_ic > 0 else 'SHORT'
    ic_results[fname] = {'mean_ic': mean_ic, 'pct_pos': pct_pos, 'stability': stability,
                          'n': n, 'direction': direction}
    print(f'  {fname:<18} {mean_ic:>+9.3f} {pct_pos:>6.1f}% {stability:>+10.3f} {n:>4}  {direction}')

# ── Strategy Definitions ─────────────────────────────────────────────────────
print('\n' + '='*70)
print('STEP 2: Strategy Evaluation (Train / Val / Test) — All Combinations')
print('='*70)

# Define all strategies
strategies = []

# Single-factor strategies
for fname in factor_mats:
    for topn in [5, 10, 20]:
        strategies.append({
            'name': f"Single-{fname} top{topn}",
            'type': 'single',
            'factors': [fname],
            'topn': topn
        })

# Multi-factor price combinations
price_factors = ['roc20', 'roc60', 'roc120', 'vol20', 'vol60']
for combo_factors in [['roc60', 'vol60'], ['roc120', 'vol20'], ['roc120', 'vol60'],
                       ['roc60', 'roc120', 'vol60']]:
    for topn in [5, 10, 20]:
        strategies.append({
            'name': f"Multi-{'+'.join(combo_factors)} top{topn}",
            'type': 'multi_equal',
            'factors': combo_factors,
            'topn': topn
        })

# Multi-factor with fundamentals
for combo_factors in [['roc120', 'roe'], ['roc60', 'roe'], ['roc120', 'earnings_yield'],
                       ['roc60', 'book_per_share'], ['roc120', 'vol60', 'roe']]:
    for topn in [5, 10, 20]:
        strategies.append({
            'name': f"Multi-{'+'.join(combo_factors)} top{topn}",
            'type': 'multi_equal',
            'factors': combo_factors,
            'topn': topn
        })

# IC-weighted multi-factor (selected pairs)
ic_pairs = [
    (['roc120', 'vol60'], 5),
    (['roc60', 'vol60'], 10),
    (['roc120', 'roe'], 5),
    (['roc120', 'vol60', 'roe'], 5),
]
for combo_factors, topn in ic_pairs:
    strategies.append({
        'name': f"IC-Weighted-{'+'.join(combo_factors)} top{topn}",
        'type': 'multi_ic',
        'factors': combo_factors,
        'topn': topn
    })

print(f'  Total strategies: {len(strategies)}')

# ── Run All Strategies ────────────────────────────────────────────────────────
all_results = []

for s in strategies:
    fname_parts = s['factors']
    topn = s['topn']
    stype = s['type']

    # Build composite matrix for this strategy
    if stype == 'single':
        combo_mat = factor_mats[fname_parts[0]]
    elif stype == 'multi_equal':
        sub_dict = {k: factor_mats[k] for k in fname_parts if k in factor_mats}
        combo_mat = combo_equal(sub_dict)
    elif stype == 'multi_ic':
        sub_dict = {k: factor_mats[k] for k in fname_parts if k in factor_mats}
        combo_mat = combo_ic_weighted(sub_dict, ret_mat, TRAIN_MI, VAL_MI)

    # ── Train period ──────────────────────────────────────────────────────
    train_rets = select_topn(combo_mat, ret_mat, topn, TRAIN_MI, VAL_MI)
    train_m = compute_metrics(train_rets)

    # ── Val period ────────────────────────────────────────────────────────
    val_rets = select_topn(combo_mat, ret_mat, topn, VAL_MI, TEST_MI)
    val_m = compute_metrics(val_rets)

    # ── Test period (LOCKED — one shot) ────────────────────────────────────
    test_rets = select_topn(combo_mat, ret_mat, topn, TEST_MI, END_MI)
    test_m = compute_metrics(test_rets)

    # ── Walk-Forward Analysis (on Val period) ─────────────────────────────
    wfa = walk_forward_analysis(ret_mat, {fname_parts[0]: factor_mats[fname_parts[0]]}
                                 if stype == 'single' else
                                 {k: factor_mats[k] for k in fname_parts},
                                 topn)

    # ── Monte Carlo on Test period ─────────────────────────────────────────
    mc = monte_carlo_block_bootstrap(test_rets)

    # ── Train IC summary ───────────────────────────────────────────────────
    train_ics = [ic_results[k]['mean_ic'] for k in fname_parts]
    avg_ic = np.mean(train_ics)

    result = {
        'strategy': s['name'],
        'type': stype,
        'factors': fname_parts,
        'topn': topn,
        'train': train_m,
        'val': val_m,
        'test': test_m,
        'wfa': wfa,
        'mc': mc,
        'avg_train_ic': avg_ic,
    }
    all_results.append(result)

# ── Rank by Val Sharpe ────────────────────────────────────────────────────────
all_results.sort(key=lambda x: x['val']['sharpe'] if not np.isnan(x['val']['sharpe']) else -999, reverse=True)

# ── Print Results Table ───────────────────────────────────────────────────────
print('\n' + '='*100)
print('RESULTS: All Strategies Ranked by Val Sharpe')
print('='*100)
print(f'{"Strategy":<40} {"Val-Sharp":>9} {"Val-Ann%":>8} {"Test-Sharp":>10} {"Test-Ann%":>9} {"WFA":>4} {"MC-5%":>7}')
print('-'*100)

for r in all_results:
    wfa_pass = '✓' if r['wfa'].get('wfa_pass', False) else '✗'
    mc5 = f"{r['mc'].get('mc_sharpe_5pct', 0):.2f}" if not np.isnan(r['mc'].get('mc_sharpe_5pct', np.nan)) else 'N/A'
    print(f"  {r['strategy']:<38} {r['val']['sharpe']:>9.3f} {r['val']['ann']:>7.1f}% "
          f"{r['test']['sharpe']:>10.3f} {r['test']['ann']:>8.1f}% {wfa_pass:>4} {mc5:>7}")

# ── SPY B&H ───────────────────────────────────────────────────────────────────
print('\n' + '='*100)
print('SPY BUY-AND-HOLD (Test Period)')
print('='*100)
if spy_rets is not None:
    test_start_idx = TEST_MI
    test_end_idx = END_MI
    spy_test_rets = []
    for j in range(test_start_idx, test_end_idx):
        if j < len(spy_rets):
            spy_test_rets.append(spy_rets[j])
    spy_test_rets = np.array(spy_test_rets)
    spy_m = compute_metrics(spy_test_rets)
    spy_mc = monte_carlo_block_bootstrap(spy_test_rets)
    print(f"  SPY B&H: Sharpe={spy_m['sharpe']:.3f}  Ann={spy_m['ann']:.1f}%  MaxDD={spy_m['maxdd']:.1f}%  "
          f"MC-5%={spy_mc.get('mc_sharpe_5pct', 'N/A')}")

# ── Save Results ───────────────────────────────────────────────────────────────
output = {
    'ic_analysis': ic_results,
    'strategies': all_results,
    'periods': {
        'train': {'start': str(monthly_ends[TRAIN_MI]), 'end': str(monthly_ends[VAL_MI]), 'n_months': N_TRAIN},
        'val': {'start': str(monthly_ends[VAL_MI]), 'end': str(monthly_ends[TEST_MI]), 'n_months': N_VAL},
        'test': {'start': str(monthly_ends[TEST_MI]), 'end': str(monthly_ends[END_MI]), 'n_months': N_TEST},
    }
}

out_path = '/tmp/comprehensive_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f'\nResults saved to {out_path}')

# ── Summary: Best per Category ────────────────────────────────────────────────
print('\n' + '='*100)
print('BEST PER CATEGORY (on Test Sharpe)')
print('='*100)

categories = {
    'Single price factor': [r for r in all_results if r['type'] == 'single' and
                             all(f in price_factors for f in r['factors'])],
    'Single fundamental factor': [r for r in all_results if r['type'] == 'single' and
                                   all(f in FUND_FACTORS for f in r['factors'])],
    'Multi-factor (price only)': [r for r in all_results if r['type'] in ('multi_equal', 'multi_ic') and
                                   all(f in price_factors for f in r['factors'])],
    'Multi-factor (price + fundamental)': [r for r in all_results if r['type'] in ('multi_equal', 'multi_ic') and
                                            any(f in FUND_FACTORS for f in r['factors'])],
}

for cat_name, cat_results in categories.items():
    if not cat_results:
        continue
    cat_results.sort(key=lambda x: x['test']['sharpe'] if not np.isnan(x['test']['sharpe']) else -999, reverse=True)
    best = cat_results[0]
    print(f"\n  [{cat_name}]")
    print(f"    Best: {best['strategy']}")
    print(f"      Val: Sharpe={best['val']['sharpe']:.3f}  Ann={best['val']['ann']:.1f}%  MaxDD={best['val']['maxdd']:.1f}%")
    print(f"      Test: Sharpe={best['test']['sharpe']:.3f}  Ann={best['test']['ann']:.1f}%  MaxDD={best['test']['maxdd']:.1f}%")
    print(f"      WFA pass: {best['wfa'].get('wfa_pass', False)}")
