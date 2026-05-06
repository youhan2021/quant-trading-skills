"""
================================================================================
COMPREHENSIVE BACKTEST v2 — Full Grid + Walk-Forward + Monte Carlo
Train(2011-2016) → Val(2016-2021) → Test(2021-2026)

Extends three_stage_backtest.py with:
- Full factor combination grid (price×price, price×fundamental, triples)
- Walk-Forward Analysis
- Monte Carlo Block Bootstrap
- IC-weighted combo method
- Train / Val / Test per-period metrics for ALL strategies
================================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
FUNDAMENTAL_FILE = '/tmp/fundamental_data.json'
CIK_MAP_FILE     = '/tmp/ticker_cik_map.json'
DATA_START = '2010-01-01'
DATA_END   = '2026-02-01'
FILING_LAG_DAYS = 60

# ── XBRL Fundamental Factor Builder ────────────────────────────────────────────

def parse_xbrl_series(fund_ticker_data, concept):
    entries = fund_ticker_data.get(concept, [])
    if isinstance(entries, dict):
        entries = entries.get('data', [])
    rows = []
    for e in entries:
        es, fs, v = e.get('end', ''), e.get('filed', ''), e.get('val')
        if not es or not fs or v is None:
            continue
        try:
            et = pd.Timestamp(es); ft = pd.Timestamp(fs)
            rows.append((et, max(et + timedelta(days=FILING_LAG_DAYS), ft), float(v)))
        except:
            continue
    return sorted(rows, key=lambda x: x[0])

def build_fundamental_series_v2(fund_ticker_data, concept, monthly_ends_arr):
    ni_r = parse_xbrl_series(fund_ticker_data, 'NetIncomeLoss')
    eq_r = parse_xbrl_series(fund_ticker_data, 'StockholdersEquity')
    sh_r = parse_xbrl_series(fund_ticker_data, 'SharesOutstanding')
    ast_r= parse_xbrl_series(fund_ticker_data, 'Assets')
    ni_map  = {r[0]: r[2] for r in ni_r}
    eq_map  = {r[0]: r[2] for r in eq_r}
    sh_map  = {r[0]: r[2] for r in sh_r}
    ast_map = {r[0]: r[2] for r in ast_r}
    all_ni  = sorted(ni_map.keys())

    def ff_fill(q_dates, q_vals):
        if not q_dates:
            return np.full(len(monthly_ends_arr), np.nan)
        cv, ci, res = np.nan, 0, np.full(len(monthly_ends_arr), np.nan)
        for mi, me in enumerate(monthly_ends_arr):
            mt = pd.Timestamp(me)
            while ci < len(q_dates) and q_dates[ci] <= mt:
                cv = q_vals[ci]; ci += 1
            res[mi] = cv
        return res

    if concept == 'roe':
        qd, qv = [], []
        for et in all_ni:
            ve = [e for e in eq_map if e <= et]
            if not ve: continue
            eq = eq_map[max(ve)]; ni = ni_map.get(et)
            if eq <= 0 or ni is None: continue
            qd.append(et + timedelta(days=FILING_LAG_DAYS)); qv.append(ni / eq)
        return ff_fill(qd, qv)

    elif concept == 'earnings_yield':
        qd, qv = [], []
        for i, et in enumerate(all_ni):
            if i < 3: continue
            w = all_ni[i-3:i+1]; tn = sum(ni_map[e] for e in w)
            ve = [e for e in eq_map if e <= et]
            if not ve: continue
            eq = eq_map[max(ve)]; 
            if eq <= 0: continue
            qd.append(et + timedelta(days=FILING_LAG_DAYS)); qv.append(tn / eq)
        return ff_fill(qd, qv)

    elif concept == 'book_per_share':
        qd, qv = [], []
        all_ends = sorted(set(list(eq_map) + list(sh_map)))
        for et in all_ends:
            if et not in eq_map or et not in sh_map: continue
            ev, sv = eq_map[et], sh_map[et]
            if ev <= 0 or sv <= 0: continue
            qd.append(et + timedelta(days=FILING_LAG_DAYS)); qv.append(ev / sv)
        return ff_fill(qd, qv)

    elif concept == 'de_ratio':
        qd, qv = [], []
        all_ends = sorted(set(list(ast_map) + list(eq_map)))
        for et in all_ends:
            if et not in ast_map or et not in eq_map: continue
            a, e = ast_map[et], eq_map[et]
            if e <= 0: continue
            d = a - e
            if d <= 0: continue
            qd.append(et + timedelta(days=FILING_LAG_DAYS)); qv.append(d / e)
        return ff_fill(qd, qv)

    return np.full(len(monthly_ends_arr), np.nan)

def build_fundamental_matrix(fundamental_data, tickers, monthly_ends_arr, concept):
    n, m = len(tickers), len(monthly_ends_arr)
    mat = np.full((n, m), np.nan)
    for i, t in enumerate(tickers):
        mat[i, :] = build_fundamental_series_v2(fundamental_data.get(t, {}), concept, monthly_ends_arr)
    return mat

# ── Data Loading ───────────────────────────────────────────────────────────────

def get_closes(data):
    if isinstance(data.columns, pd.MultiIndex):
        return data.xs('Close', level=1, axis=1)
    return data[['Close']] if 'Close' in data.columns else data

def resample_monthly(closes):
    monthly = closes.resample('ME').last()
    return monthly[monthly.index <= pd.Timestamp(DATA_END)]

def load_fundamental_data():
    try:
        with open(FUNDAMENTAL_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f'ERROR: {FUNDAMENTAL_FILE} not found')
        return {}

# ── Core Backtest ───────────────────────────────────────────────────────────────

def rank_zscore(mat, axis=0):
    if axis == 0:
        result = np.full_like(mat, np.nan)
        for j in range(mat.shape[1]):
            col = mat[:, j]; mask = ~np.isnan(col)
            if mask.sum() < 3: continue
            ranks = pd.Series(col[mask]).rank().values
            mu, sigma = ranks.mean(), ranks.std()
            if sigma > 0: result[mask, j] = (ranks - mu) / sigma
        return result
    else:
        result = np.full_like(mat, np.nan)
        for i in range(mat.shape[0]):
            row = mat[i, :]; mask = ~np.isnan(row)
            if mask.sum() < 3: continue
            ranks = pd.Series(row[mask]).rank().values
            mu, sigma = ranks.mean(), ranks.std()
            if sigma > 0: result[i, mask] = (ranks - mu) / sigma
        return result

def combo_equal(factor_dict):
    mats = [rank_zscore(factor_dict[k], axis=0) for k in factor_dict]
    return np.nanmean(np.stack(mats), axis=0)

def combo_ic_weighted(factor_dict, ret_mat, train_start, train_end):
    ic_scores = {}
    for k, mat in factor_dict.items():
        ic_vals = []
        for j in range(train_start, train_end):
            fc, rc = mat[:, j], ret_mat[:, j]
            mask = ~np.isnan(fc) & ~np.isnan(rc)
            if mask.sum() < 5: continue
            ic_vals.append(np.corrcoef(fc[mask], rc[mask])[0, 1])
        ic_scores[k] = np.nanmean(ic_vals) if ic_vals else 0.0
    total = sum(abs(v) for v in ic_scores.values())
    if total == 0: return combo_equal(factor_dict)
    weights = {k: abs(v) / total for k, v in ic_scores.items()}
    mats = [rank_zscore(factor_dict[k], axis=0) * weights[k] for k in factor_dict]
    return np.nansum(np.stack(mats), axis=0)

def select_topn(combo_mat, ret_mat, n, period_start, period_end):
    results = []
    for j in range(period_start, period_end):
        scores, rets = combo_mat[:, j], ret_mat[:, j]
        mask = ~np.isnan(scores) & ~np.isnan(rets)
        if mask.sum() == 0:
            results.append(np.nan); continue
        top = np.argsort(-scores[mask])[:min(n, mask.sum())]
        tickers_in_top = np.where(mask)[0][top]
        results.append(np.mean(rets[tickers_in_top]))
    return np.array(results)

def compute_metrics(rets, periods_per_year=12):
    r = rets[~np.isnan(rets)]
    if len(r) < 3:
        return {'sharpe': np.nan, 'ann': np.nan, 'maxdd': np.nan, 'n': len(r)}
    ann  = np.mean(r) * periods_per_year
    vol  = np.std(r, ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann / vol if vol > 0 else np.nan
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    maxdd = np.min(drawdown) if len(drawdown) > 0 else 0.0
    return {'sharpe': sharpe, 'ann': ann * 100, 'maxdd': maxdd * 100, 'n': len(r)}

# ── Walk-Forward Analysis ───────────────────────────────────────────────────────

def walk_forward(ret_mat, combo_mat, n, wf_window=24, wf_step=6):
    """Walk-forward on Val period. Returns WFA metrics dict."""
    n_months = ret_mat.shape[1]
    wf_sharpes = []
    for p in range(0, n_months - wf_window, wf_step):
        train_end = p + wf_window
        train_start = max(0, train_end - wf_window)
        if train_end - train_start < 12: continue

        # Pick best single factor by train IC
        best_ic, best_fname, best_mat = -999, None, combo_mat
        if isinstance(combo_mat, dict):
            for fname, fmat in combo_mat.items():
                ic_vals = []
                for j in range(train_start, train_end):
                    fc, rc = fmat[:, j], ret_mat[:, j]
                    mask = ~np.isnan(fc) & ~np.isnan(rc)
                    if mask.sum() < 5: continue
                    ic_vals.append(np.corrcoef(fc[mask], rc[mask])[0, 1])
                mean_ic = np.nanmean(ic_vals) if ic_vals else 0
                if mean_ic > best_ic:
                    best_ic = mean_ic; best_fname = fname; best_mat = fmat

        val_rets = select_topn(best_mat, ret_mat, n, train_end, min(train_end + wf_step, n_months))
        m = compute_metrics(val_rets)
        if not np.isnan(m['sharpe']):
            wf_sharpes.append(m['sharpe'])

    if not wf_sharpes:
        return {'sharpe_mean': np.nan, 'sharpe_std': np.nan, 'ann_mean': np.nan, 'wfa_pass': False, 'n': 0}
    sharpe_mean = np.mean(wf_sharpes)
    return {
        'sharpe_mean': sharpe_mean,
        'sharpe_std':  np.std(wf_sharpes, ddof=1) if len(wf_sharpes) > 1 else np.nan,
        'ann_mean':    np.mean(wf_sharpes) * 12 * 100,
        'wfa_pass':    sharpe_mean > 0.5,
        'n':           len(wf_sharpes)
    }

# ── Monte Carlo Block Bootstrap ─────────────────────────────────────────────────

def monte_carlo(ret_series, n_sim=1000, block_size=6, seed=42):
    r = ret_series[~np.isnan(ret_series)]
    if len(r) < block_size * 2:
        return {'p5': np.nan, 'p95': np.nan, 'median': np.nan}
    rng = np.random.default_rng(seed)
    sharpes = []
    n = len(r)
    for _ in range(n_sim):
        boot = []
        for _ in range(n // block_size + 1):
            start = rng.integers(0, n)
            for b in range(block_size):
                boot.append(r[(start + b) % n])
        boot = np.array(boot[:n])
        ann = np.mean(boot) * 12
        vol = np.std(boot, ddof=1) * np.sqrt(12)
        sh = ann / vol if vol > 0 else np.nan
        if not np.isnan(sh): sharpes.append(sh)
    if not sharpes: return {'p5': np.nan, 'p95': np.nan, 'median': np.nan}
    sharpes = np.array(sharpes)
    return {
        'p5':     np.percentile(sharpes, 5),
        'p95':    np.percentile(sharpes, 95),
        'median': np.median(sharpes),
        'mean':   np.mean(sharpes),
    }

# ── Main ───────────────────────────────────────────────────────────────────────
print('='*72)
print('COMPREHENSIVE BACKTEST v2 — Full Grid + WFA + Monte Carlo')
print('Train(2011-2016) → Val(2016-2021) → Test(2021-2026)')
print('='*72)

fundamental_data = load_fundamental_data()

print('\n[0] Downloading prices...')
data_all = yf.download(['AAPL','MSFT','AMZN','NVDA','GOOGL','META','BRK-B','LLY','AVGO',
    'HD','XOM','UNH','JPM','MA','PG','COST','JNJ','ABBV','WMT','BAC','CRM','MRK','CVX',
    'PEP','KO','TMO','CSCO','MCD','ABT','ACN','DHR','NKE','TXN','NEE','PM','UPS','MS',
    'RTX','HON','LOW','QCOM','INTC','AMD','IBM','GS','BLK','C','ADI','PANW','NOW','AMT',
    'INTU','SPGI','V','USB','PNC','TGT','SCHW','BKNG','AXP','DE','GE','CAT','BA','MMM',
    'DIS','ISRG','MDT','BDX','SYK','ZTS','GILD','PLTR'],
    start=DATA_START, end=DATA_END, progress=False, auto_adjust=True, group_by='ticker')

available = []
for t in data_all.columns.get_level_values(0).unique():
    if t not in data_all.columns.get_level_values(0): continue
    col = get_closes(data_all)[t].dropna()
    if len(col) >= 252 and col.index[0] <= pd.Timestamp('2010-12-31'):
        available.append(t)
tickers = sorted(set(available))
print(f'  Universe: {len(tickers)} tickers')

closes = get_closes(data_all)
monthly_idx = resample_monthly(closes).index
monthly_ends = list(monthly_idx)
n_months = len(monthly_ends)

def mi_for(s):
    ts = pd.Timestamp(s)
    return max(0, monthly_idx.get_indexer([ts], method='ffill')[0])

TRAIN_MI = mi_for('2011-01-01')
VAL_MI   = mi_for('2016-01-01')
TEST_MI  = mi_for('2021-01-01')
END_MI   = n_months - 1

print(f'\n  Train: {monthly_ends[TRAIN_MI].date()} → {monthly_ends[VAL_MI].date()} ({VAL_MI-TRAIN_MI}m)')
print(f'  Val:   {monthly_ends[VAL_MI].date()} → {monthly_ends[TEST_MI].date()} ({TEST_MI-VAL_MI}m)')
print(f'  Test:  {monthly_ends[TEST_MI].date()} → {monthly_ends[END_MI].date()} ({END_MI-TEST_MI}m)')

# ── Build Factor Matrices ───────────────────────────────────────────────────────
print('\n[1] Building factor matrices...')
factor_mats = {}

for fname, window, ftype in [
    ('roc20', 20, 'roc'), ('roc60', 60, 'roc'), ('roc120', 120, 'roc'),
    ('vol20', 20, 'vol'), ('vol60', 60, 'vol')]:
    print(f'  {fname}...', end=' ', flush=True)
    mat = np.full((len(tickers), n_months), np.nan)
    for i, t in enumerate(tickers):
        if t not in closes.columns: continue
        px = closes[t].dropna()
        if len(px) < window: continue
        series = px.pct_change(window).dropna() if ftype == 'roc' else px.pct_change().dropna().rolling(window).std().dropna()
        for j, me in enumerate(monthly_ends):
            mt = pd.Timestamp(me)
            pos = closes.index.get_indexer([mt], method='ffill')[0]
            if pos >= len(closes.index): pos = len(closes.index) - 1
            if pos < 0: continue
            nt = closes.index[pos]
            if nt in series.index: mat[i, j] = series.loc[nt]
            elif pos > 0 and closes.index[pos-1] in series.index: mat[i, j] = series.loc[closes.index[pos-1]]
    print(f'{int(np.sum(~np.isnan(mat)))} cells')
    factor_mats[fname] = mat

for ff in ['roe', 'earnings_yield', 'book_per_share', 'de_ratio']:
    print(f'  {ff}...', end=' ', flush=True)
    mat = build_fundamental_matrix(fundamental_data, tickers, monthly_ends, ff)
    print(f'{int(np.sum(~np.isnan(mat)))} cells')
    factor_mats[ff] = mat

# ── Return Matrix ──────────────────────────────────────────────────────────────
print('\n[2] Building return matrix...')
monthly_closes = resample_monthly(closes)
monthly_rets = monthly_closes.pct_change()  # NO dropna()
ret_mat = np.full((len(tickers), n_months), np.nan)
for i, t in enumerate(tickers):
    if t not in monthly_rets.columns: continue
    series = monthly_rets[t]
    for j, me in enumerate(monthly_ends):
        mt = pd.Timestamp(me)
        # FIX: score at j → return FROM j TO j+1 (next month)
        # Use j+1 return (forward-looking), fallback to j return if last month
        if j + 1 < n_months and monthly_ends[j + 1] in series.index:
            ret_mat[i, j] = series.loc[monthly_ends[j + 1]]
        elif mt in series.index:
            ret_mat[i, j] = series.loc[mt]
# SPY
spy_rets = monthly_rets['SPY'].values if 'SPY' in monthly_rets.columns else None

# ── IC Analysis (Train) ───────────────────────────────────────────────────────
print('\n' + '='*72)
print('STEP 1: Train IC Analysis — Factor Direction')
print('='*72)
print(f'  {"Factor":<18} {"Mean IC":>9} {"IC>0%":>7} {"Stability":>10} {"N":>4}  Dir')
print('  ' + '-'*60)

ic_results = {}
for fname, mat in factor_mats.items():
    ic_vals = []
    for j in range(TRAIN_MI, VAL_MI):
        fc, rc = mat[:, j], ret_mat[:, j]
        mask = ~np.isnan(fc) & ~np.isnan(rc)
        if mask.sum() < 5: continue
        ic_vals.append(np.corrcoef(fc[mask], rc[mask])[0, 1])
    if not ic_vals: continue
    mean_ic = np.nanmean(ic_vals)
    pct_pos = np.nanmean([v > 0 for v in ic_vals]) * 100
    stab = np.corrcoef(ic_vals[:-1], ic_vals[1:])[0, 1] if len(ic_vals) > 2 else 0
    ic_results[fname] = {'mean_ic': mean_ic, 'pct_pos': pct_pos, 'stability': stab,
                          'n': len(ic_vals), 'direction': 'LONG' if mean_ic > 0 else 'SHORT'}
    print(f'  {fname:<18} {mean_ic:>+9.3f} {pct_pos:>6.1f}% {stab:>+10.3f} {len(ic_vals):>4}  {ic_results[fname]["direction"]}')

# ── Strategy Grid ─────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('STEP 2: Evaluating ALL Strategies on Train / Val / Test')
print('='*72)

PRICE_FACTORS = ['roc20', 'roc60', 'roc120', 'vol20', 'vol60']
FUND_FACTORS  = ['roe', 'earnings_yield', 'book_per_share', 'de_ratio']
TOP_N_OPTIONS = [5, 10, 20]

strategies = []

# Single-factor
for pf in PRICE_FACTORS + FUND_FACTORS:
    for n in TOP_N_OPTIONS:
        strategies.append({'name': f'Single-{pf} top{n}', 'type': 'single', 'factors': [pf], 'topn': n})

# Price × price combos
for pf1 in ['roc60', 'roc120']:
    for pf2 in ['vol20', 'vol60']:
        for n in TOP_N_OPTIONS:
            strategies.append({'name': f'Price-{pf1}+{pf2} top{n}', 'type': 'multi_eq', 'factors': [pf1, pf2], 'topn': n})

# Price × fundamental combos
for pf in ['roc60', 'roc120']:
    for ff in FUND_FACTORS:
        for n in TOP_N_OPTIONS:
            strategies.append({'name': f'Multi-{pf}+{ff} top{n}', 'type': 'multi_eq', 'factors': [pf, ff], 'topn': n})

# Triple combos (price + price + fund)
for pf1, pf2 in [('roc120','vol60'), ('roc60','vol60')]:
    for ff in ['roe', 'earnings_yield']:
        for n in [5, 10]:
            strategies.append({'name': f'Triple-{pf1}+{pf2}+{ff} top{n}', 'type': 'multi_eq', 'factors': [pf1, pf2, ff], 'topn': n})

# IC-weighted pairs
ic_pairs = [
    (['roc120', 'vol60'], 5),
    (['roc60', 'vol60'], 10),
    (['roc120', 'roe'], 5),
    (['roc120', 'vol60', 'roe'], 5),
]
for combo, n in ic_pairs:
    strategies.append({'name': f'ICWt-{"+".join(combo)} top{n}', 'type': 'multi_ic', 'factors': combo, 'topn': n})

print(f'  Total: {len(strategies)} strategies')

# ── Run All ────────────────────────────────────────────────────────────────────
all_results = []

for s in strategies:
    fnames = s['factors']
    topn = s['topn']
    stype = s['type']

    if stype == 'single':
        combo_mat = factor_mats[fnames[0]]
    elif stype == 'multi_eq':
        combo_mat = combo_equal({k: factor_mats[k] for k in fnames})
    elif stype == 'multi_ic':
        combo_mat = combo_ic_weighted({k: factor_mats[k] for k in fnames}, ret_mat, TRAIN_MI, VAL_MI)

    # Period returns
    train_rets = select_topn(combo_mat, ret_mat, topn, TRAIN_MI, VAL_MI)
    val_rets   = select_topn(combo_mat, ret_mat, topn, VAL_MI, TEST_MI)
    test_rets  = select_topn(combo_mat, ret_mat, topn, TEST_MI, END_MI)

    train_m = compute_metrics(train_rets)
    val_m   = compute_metrics(val_rets)
    test_m  = compute_metrics(test_rets)

    # Walk-forward on Val
    combo_for_wfa = {fnames[0]: factor_mats[fnames[0]]} if stype == 'single' else {k: factor_mats[k] for k in fnames}
    wfa = walk_forward(ret_mat, combo_for_wfa, topn)

    # Monte Carlo on Test
    mc = monte_carlo(test_rets)

    # Avg train IC
    avg_ic = np.mean([ic_results.get(f, {}).get('mean_ic', 0) for f in fnames])

    all_results.append({
        'strategy': s['name'],
        'type': stype,
        'factors': fnames,
        'topn': topn,
        'train': train_m,
        'val': val_m,
        'test': test_m,
        'wfa': wfa,
        'mc': mc,
        'avg_train_ic': avg_ic,
    })

# Sort by Val Sharpe
all_results.sort(key=lambda x: x['val']['sharpe'] if not np.isnan(x['val']['sharpe']) else -999, reverse=True)

# ── Results Table ──────────────────────────────────────────────────────────────
print(f'\n{"Strategy":<42} {"Val-Shp":>8} {"Val-Ann":>7} {"Test-Shp":>9} {"Test-Ann":>7} {"WFA":>4} {"MC-p5":>7}')
print('  ' + '-'*90)
for r in all_results:
    wf = '✓' if r['wfa'].get('wfa_pass') else '✗'
    p5 = f"{r['mc'].get('p5', 0):.2f}" if not np.isnan(r['mc'].get('p5', np.nan)) else 'N/A'
    vs = f"{r['val']['sharpe']:.3f}" if not np.isnan(r['val']['sharpe']) else 'N/A'
    va = f"{r['val']['ann']:.1f}" if not np.isnan(r['val']['ann']) else 'N/A'
    ts = f"{r['test']['sharpe']:.3f}" if not np.isnan(r['test']['sharpe']) else 'N/A'
    ta = f"{r['test']['ann']:.1f}" if not np.isnan(r['test']['ann']) else 'N/A'
    print(f'  {r["strategy"]:<40} {vs:>8} {va:>6}% {ts:>9} {ta:>6}% {wf:>4} {p5:>7}')

# ── SPY B&H ────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('SPY BUY-AND-HOLD (Test Period)')
if spy_rets is not None:
    spy_test = []
    for j in range(TEST_MI, END_MI):
        if j < len(spy_rets): spy_test.append(spy_rets[j])
    spy_test = np.array(spy_test)
    spy_m = compute_metrics(spy_test)
    spy_mc = monte_carlo(spy_test)
    print(f'  Sharpe={spy_m["sharpe"]:.3f}  Ann={spy_m["ann"]:.1f}%  MaxDD={spy_m["maxdd"]:.1f}%  '
          f'MC-p5={spy_mc.get("p5", "N/A"):.2f}')

# ── Best Per Category ─────────────────────────────────────────────────────────
print('\n' + '='*72)
print('BEST PER CATEGORY (by Test Sharpe)')
print('='*72)
cats = {
    'Single price':      [r for r in all_results if r['type']=='single' and all(f in PRICE_FACTORS for f in r['factors'])],
    'Single fund':       [r for r in all_results if r['type']=='single' and all(f in FUND_FACTORS for f in r['factors'])],
    'Price×price':       [r for r in all_results if r['type']=='multi_eq' and len(r['factors'])==2 and
                          all(f in PRICE_FACTORS for f in r['factors'])],
    'Price×fund':        [r for r in all_results if r['type']=='multi_eq' and len(r['factors'])==2 and
                          any(f in PRICE_FACTORS for f in r['factors']) and any(f in FUND_FACTORS for f in r['factors'])],
    'Triple':            [r for r in all_results if r['type']=='multi_eq' and len(r['factors'])==3],
    'IC-weighted':       [r for r in all_results if r['type']=='multi_ic'],
}
for cat_name, cat_results in cats.items():
    if not cat_results: continue
    cat_results.sort(key=lambda x: x['test']['sharpe'] if not np.isnan(x['test']['sharpe']) else -999, reverse=True)
    best = cat_results[0]
    print(f'\n  [{cat_name}]')
    print(f'    Strategy: {best["strategy"]}')
    print(f'    Train: Sharpe={best["train"]["sharpe"]:.3f}  Ann={best["train"]["ann"]:.1f}%')
    print(f'    Val:   Sharpe={best["val"]["sharpe"]:.3f}  Ann={best["val"]["ann"]:.1f}%  MaxDD={best["val"]["maxdd"]:.1f}%')
    print(f'    Test:  Sharpe={best["test"]["sharpe"]:.3f}  Ann={best["test"]["ann"]:.1f}%  MaxDD={best["test"]["maxdd"]:.1f}%')
    print(f'    WFA pass: {best["wfa"].get("wfa_pass", False)}  MC-p5: {best["mc"].get("p5", "N/A"):.2f}')

# ── Save ───────────────────────────────────────────────────────────────────────
output = {
    'ic_analysis': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for kk, vv in v.items()} for k, v in ic_results.items()},
    'strategies': all_results,
    'periods': {
        'train': {'start': str(monthly_ends[TRAIN_MI]), 'end': str(monthly_ends[VAL_MI]), 'n': VAL_MI-TRAIN_MI},
        'val':   {'start': str(monthly_ends[VAL_MI]), 'end': str(monthly_ends[TEST_MI]), 'n': TEST_MI-VAL_MI},
        'test':  {'start': str(monthly_ends[TEST_MI]), 'end': str(monthly_ends[END_MI]), 'n': END_MI-TEST_MI},
    }
}
with open('/tmp/comprehensive_v2_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f'\nSaved to /tmp/comprehensive_v2_results.json')
