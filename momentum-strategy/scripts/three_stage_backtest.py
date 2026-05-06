"""
三段式严格回测 — 绝对干净的验证流程
==========================================

目标：消除一切 selection bias，汇报每个阶段真实 performance

三阶段：
  Train (2011-01 ~ 2016-01): IC 分析 → 定因子方向（direction）
  Val   (2016-01 ~ 2021-01): 遍历所有因子组合 → 选 Val Sharpe 最高者
  Test  (2021-01 ~ 2026-01): 锁定选择，运行一次 → 汇报 Test 指标

严格规则：
  - 因子方向由 Train IC 决定（不能用 Val/Test 的数据）
  - 策略选择只看 Val Sharpe（不能用 Test 优化）
  - Test 只跑一次，不调参，不重选
  - 每个阶段分别汇报 metrics，不混用

Key metrics per period:
  - Sharpe ratio (annualized)
  - Annualized return
  - Max drawdown
  - Win rate (% months positive)
"""

import sys, math, json
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, '/home/ubuntu/.hermes/skills/quant-trading-momentum/scripts')
import requests

import warnings
warnings.filterwarnings('ignore')

# ── Data paths ───────────────────────────────────────────────────────────────
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

# ── XBRL ─────────────────────────────────────────────────────────────────────
FILING_LAG_DAYS = 60

def load_fundamental_data():
    try:
        with open(FUNDAMENTAL_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f'ERROR: {FUNDAMENTAL_FILE} not found.')
        sys.exit(1)

def load_cik_map():
    with open(CIK_MAP_FILE) as f:
        return json.load(f)

def build_fundamental_series_v2(fund_ticker_data, concept, monthly_ends, filing_lag_days=60):
    """
    Build a TxN matrix of a fundamental factor for one ticker.
    Computes derived factors (roe, earnings_yield, book_per_share, de_ratio)
    from raw XBRL concepts (NetIncomeLoss, StockholdersEquity, SharesOutstanding, Assets).
    Uses XBRL quarterly data: effective_date = fiscal_end + filing_lag.
    
    XBRL flat format: fund_ticker_data[concept_name] = [{end, filed, val}, ...]
    """
    from datetime import timedelta

    def parse(raw, concept_key):
        """Parse XBRL entries for a given concept. Returns [(end_ts, effective_ts, val)]."""
        entries = raw.get(concept_key, [])
        if isinstance(entries, list):
            parsed = entries
        else:
            parsed = entries.get('data', []) if isinstance(entries, dict) else []
        rows = []
        for e in parsed:
            end_str = e.get('end', '')
            filed_str = e.get('filed', '')
            val = e.get('val')
            if not end_str or not filed_str or val is None:
                continue
            try:
                end_ts = pd.Timestamp(end_str)
                filed_ts = pd.Timestamp(filed_str)
                eff_ts = max(end_ts + timedelta(days=filing_lag_days), filed_ts)
                rows.append((end_ts, eff_ts, float(val)))
            except:
                continue
        return sorted(rows, key=lambda x: x[0])

    ni_rows = parse(fund_ticker_data, 'NetIncomeLoss')
    eq_rows = parse(fund_ticker_data, 'StockholdersEquity')
    sh_rows = parse(fund_ticker_data, 'SharesOutstanding')
    assets_rows = parse(fund_ticker_data, 'Assets')

    ni_by_end = {r[0]: r[2] for r in ni_rows}
    eq_by_end = {r[0]: r[2] for r in eq_rows}
    sh_by_end = {r[0]: r[2] for r in sh_rows}
    assets_by_end = {r[0]: r[2] for r in assets_rows}
    all_ni_ends = sorted(ni_by_end.keys())

    def ff_fill(compute_fn):
        """Given a per-quarter compute_fn(end_ts), forward-fill to monthly_ends."""
        # Build quarterly time series
        q_dates = []
        q_vals = []
        for end_ts in all_ni_ends:
            v = compute_fn(end_ts)
            if v is not None:
                eff_ts = end_ts + timedelta(days=filing_lag_days)
                q_dates.append(eff_ts)
                q_vals.append(v)
        if not q_dates:
            return np.full(len(monthly_ends), np.nan)
        # Forward-fill to monthly
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
        def roe_fn(end_ts):
            valid_eq = [e for e in eq_by_end if e <= end_ts]
            if not valid_eq:
                return None
            eq_end = max(valid_eq)
            equity = eq_by_end[eq_end]
            if equity <= 0:
                return None
            ni = ni_by_end.get(end_ts)
            if ni is None:
                return None
            return ni / equity
        return ff_fill(roe_fn)

    elif concept == 'earnings_yield':
        def ey_fn(end_ts):
            idx = all_ni_ends.index(end_ts)
            if idx < 3:
                return None
            window_ends = all_ni_ends[idx-3:idx+1]
            total_ni = sum(ni_by_end[e] for e in window_ends)
            valid_eq = [e for e in eq_by_end if e <= end_ts]
            if not valid_eq:
                return None
            equity = eq_by_end[max(valid_eq)]
            if equity <= 0:
                return None
            return total_ni / equity
        return ff_fill(ey_fn)

    elif concept == 'book_per_share':
        def bps_fn(end_ts):
            if end_ts not in eq_by_end or end_ts not in sh_by_end:
                return None
            eq_val = eq_by_end[end_ts]
            sh_val = sh_by_end[end_ts]
            if eq_val <= 0 or sh_val <= 0:
                return None
            return eq_val / sh_val
        return ff_fill(bps_fn)

    elif concept == 'de_ratio':
        def de_fn(end_ts):
            if end_ts not in assets_by_end or end_ts not in eq_by_end:
                return None
            a = assets_by_end[end_ts]
            e = eq_by_end[end_ts]
            if e <= 0:
                return None
            debt = a - e
            if debt <= 0:
                return None
            return debt / e
        return ff_fill(de_fn)

    return np.full(len(monthly_ends), np.nan)


def build_fundamental_matrix(fundamental_data, ticker_idx, monthly_ends, concept):
    """Build TxN matrix for a fundamental factor across all tickers."""
    n_tickers = len(ticker_idx)
    n_months = len(monthly_ends)
    matrix = np.full((n_tickers, n_months), np.nan)

    for ti, ticker in enumerate(ticker_idx):
        fund_ticker = fundamental_data.get(ticker, {})
        matrix[ti, :] = build_fundamental_series_v2(fund_ticker, concept, monthly_ends)

    return matrix

# ── Price factors ─────────────────────────────────────────────────────────────

def download_prices(tickers):
    print(f"  Downloading {len(tickers)} tickers from yfinance...")
    data = yf.download(tickers, start=DATA_START, end=DATA_END, progress=False,
                       auto_adjust=True, group_by='ticker')
    return data


def get_closes(data):
    """Extract close prices from yfinance multi-level column DataFrame."""
    if isinstance(data.columns, pd.MultiIndex):
        return data.xs('Close', level=1, axis=1)
    else:
        return data[['Close']] if 'Close' in data.columns else data


def resample_monthly(closes):
    """Resample daily closes to monthly (last value each month) per ticker.
    Handles NaN per-column so one ticker's gap doesn't drop all data.
    """
    monthly = closes.resample('ME').last()
    return monthly[monthly.index <= pd.Timestamp(DATA_END)]


def compute_factor_matrix(prices, tickers, monthly_ends, factor_type, window):
    """Build TxN factor matrix (T=months, N=tickers)."""
    n_tickers = len(tickers)
    n_months = len(monthly_ends)
    matrix = np.full((n_tickers, n_months), np.nan)

    for ti, ticker in enumerate(tickers):
        try:
            if ticker not in prices.columns.get_level_values(0):
                continue
            px = get_closes(prices)[ticker].dropna()
            if len(px) < window:
                continue

            if factor_type == 'roc':
                ret = px.pct_change(window).dropna()
            elif factor_type == 'vol':
                ret = px.pct_change().rolling(window).std().dropna()
            else:
                continue

            for mi, me in enumerate(monthly_ends):
                me_ts = pd.Timestamp(me)
                window_end = me_ts
                lookback_days = max(window * 3, 60)
                window_start = window_end - pd.DateOffset(days=lookback_days)
                mask = (ret.index >= window_start) & (ret.index <= window_end)
                if mask.any():
                    matrix[ti, mi] = ret.loc[mask].iloc[-1]
        except Exception:
            continue

    return matrix


# ── Core backtest engine ───────────────────────────────────────────────────────

def compute_max_dd(pv):
    """Compute max drawdown from portfolio value series."""
    pv = np.array(pv, dtype=float)
    running_max = np.maximum.accumulate(pv)
    drawdowns = (pv - running_max) / running_max
    return float(np.min(drawdowns))


def period_metrics(portfolio_values):
    """Compute Sharpe/Ann/MaxDD/win_rate from portfolio value series."""
    if len(portfolio_values) < 3:
        return dict(sharpe=0, ann=0, max_dd=0, win_rate=0, n_months=0)
    pv = np.array(portfolio_values, dtype=float)
    monthly_rets = np.diff(pv) / pv[:-1]
    monthly_rets = monthly_rets[~np.isnan(monthly_rets)]
    if len(monthly_rets) < 3:
        return dict(sharpe=0, ann=0, max_dd=0, win_rate=0, n_months=len(monthly_rets))
    ann = float(np.mean(monthly_rets) * 12)
    std = float(np.std(monthly_rets) * math.sqrt(12))
    sharpe = ann / std if std > 1e-9 else 0.0
    max_dd = compute_max_dd(portfolio_values)
    win_rate = float(np.mean((monthly_rets > 0).astype(float)))
    return dict(sharpe=sharpe, ann=ann, max_dd=max_dd, win_rate=win_rate, n_months=len(monthly_rets))


def run_backtest(top_n, factor_names, factor_direction, ret_matrix, start_mi, end_mi, ticker_idx):
    """
    Run a backtest for a given combination of factors.
    Returns portfolio values and metrics.
    """
    if start_mi >= end_mi - 1:
        return None, None

    portfolio_values = [1.0]
    prev_val = 1.0

    for mi in range(start_mi, end_mi):
        scores = np.zeros(len(ticker_idx))
        for fname in factor_names:
            ft = factor_matrix_dict[fname][:, mi]
            mask = ~np.isnan(ft)
            if mask.sum() == 0:
                continue
            ft_norm = np.zeros(len(ticker_idx))
            ft_norm[mask] = (ft[mask] - np.nanmean(ft[mask])) / (np.nanstd(ft[mask]) + 1e-9)
            scores += factor_direction[fname] * ft_norm

        top_indices = np.argsort(scores)[-top_n:]
        valid = ~np.isnan(ret_matrix[top_indices, mi + 1])
        top_indices = top_indices[valid]

        if len(top_indices) == 0:
            portfolio_values.append(prev_val)
            continue

        port_ret = float(np.nanmean(ret_matrix[top_indices, mi + 1]))
        prev_val = prev_val * (1 + port_ret)
        portfolio_values.append(prev_val)

    m = period_metrics(portfolio_values)
    return portfolio_values, m


def ic_between_factor_and_returns(factor_col, ret_col):
    """Compute IC (Pearson correlation) between a factor column and next-month returns."""
    mask = ~(np.isnan(factor_col) | np.isnan(ret_col))
    if mask.sum() < 10:
        return np.nan
    return float(np.corrcoef(factor_col[mask], ret_col[mask])[0, 1])


def compute_period_ic(factor_matrix, ret_matrix, start_mi, end_mi):
    """Compute IC series for a factor in a given period."""
    ics = []
    for mi in range(start_mi, min(end_mi, factor_matrix.shape[1] - 1)):
        ic = ic_between_factor_and_returns(factor_matrix[:, mi], ret_matrix[:, mi + 1])
        if not np.isnan(ic):
            ics.append(ic)
    return ics


def ic_summary(ic_list):
    if len(ic_list) < 3:
        return dict(mean=0, std=0, frac_pos=0, stability=0, n=len(ic_list))
    arr = np.array(ic_list)
    mean_ic = float(np.mean(arr))
    std_ic = float(np.std(arr))
    frac = float(np.mean((arr > 0).astype(float)))
    stab = mean_ic / std_ic if std_ic > 1e-9 else 0.0
    return dict(mean=mean_ic, std=std_ic, frac_pos=frac, stability=stab, n=len(ic_list))


# ── Walk-Forward Analysis ──────────────────────────────────────────────────────

def walk_forward_analysis(top_n, factor_names, factor_direction,
                          ret_matrix, monthly_ends, n_train_months=60,
                          n_val_months=12):
    """
    Walk-forward: roll through time, each window:
      Train (60mo) → Val (12mo) → Test (1mo)
    Then Test window shifts by 1 month.
    Aggregates Test metrics across all windows.
    """
    results = []
    n_total = ret_matrix.shape[1]

    # Walk forward: start at month n_train_months, step by n_val_months
    wf_start = n_train_months
    while wf_start + n_val_months < n_total - 1:
        train_end = wf_start
        val_end = min(wf_start + n_val_months, n_total - 1)
        test_end = min(val_end + 1, n_total - 1)  # 1-month test

        _, val_m = run_backtest(top_n, factor_names, factor_direction,
                                  ret_matrix, train_end, val_end,
                                  list(range(ret_matrix.shape[0])))
        _, test_m = run_backtest(top_n, factor_names, factor_direction,
                                  ret_matrix, val_end, test_end,
                                  list(range(ret_matrix.shape[0])))

        if val_m and test_m:
            results.append({
                'train_end': str(monthly_ends[train_end]),
                'val_end': str(monthly_ends[val_end]),
                'test_end': str(monthly_ends[test_end]),
                'val_sharpe': val_m['sharpe'],
                'test_sharpe': test_m['sharpe'],
                'test_ann': test_m['ann'],
                'test_max_dd': test_m['max_dd'],
            })

        wf_start += n_val_months

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

print('='*70)
print('THREE-STAGE CLEAN BACKTEST')
print('Train(2011-2016) → Val(2016-2021) → Test(2021-2026)')
print('='*70)

# Load data
fundamental_data = load_fundamental_data()
cik_map = load_cik_map()

print('\n[0] Downloading prices...')
data_all = download_prices(UNIVERSE)

# Build final universe
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

# Build monthly end dates
monthly_idx = resample_monthly(get_closes(data_all)).index  # Keep as DatetimeIndex for get_indexer
monthly_ends = list(monthly_idx)

# Build price factor matrices
print('\n[0b] Building price factor matrices...')
price_factors = {
    'roc20':  ('roc', 20),
    'roc60':  ('roc', 60),
    'roc120': ('roc', 120),
    'vol20':  ('vol', 20),
    'vol60':  ('vol', 60),
}
factor_matrix_dict = {}
for fname, (ftype, win) in price_factors.items():
    m = compute_factor_matrix(data_all, tickers, monthly_ends, ftype, win)
    factor_matrix_dict[fname] = m
    non_nan = int(np.sum(~np.isnan(m)))
    print(f'  {fname}: {non_nan} non-NaN cells')

# Build fundamental factor matrices
FUNDAMENTAL_FACTORS = ['roe', 'earnings_yield', 'book_per_share', 'de_ratio']
for ff in FUNDAMENTAL_FACTORS:
    m = build_fundamental_matrix(fundamental_data, tickers, monthly_ends, ff)
    factor_matrix_dict[ff] = m
    non_nan = int(np.sum(~np.isnan(m)))
    print(f'  {ff}: {non_nan} non-NaN cells')

# Build return matrix
print('\n[0c] Building return matrix...')
n_tickers = len(tickers)
n_months = len(monthly_ends)
ret_matrix = np.full((n_tickers, n_months), np.nan)
for ti, ticker in enumerate(tickers):
    try:
        px = get_closes(data_all)[ticker].dropna().resample('ME').last()
        monthly_rets = px.pct_change().dropna()
        for mi, me in enumerate(monthly_ends):
            me_ts = pd.Timestamp(me)
            if me_ts in monthly_rets.index:
                ret_matrix[ti, mi] = monthly_rets.loc[me_ts]
            elif mi > 0 and monthly_ends[mi-1] in monthly_rets.index:
                ret_matrix[ti, mi] = monthly_rets.loc[monthly_ends[mi-1]]
    except Exception:
        continue

# Period boundaries
def mi_for(date_str):
    ts = pd.Timestamp(date_str)
    idx = monthly_idx.get_indexer([ts], method='ffill')[0]
    return max(0, idx)

TRAIN_MI = mi_for('2011-01-01')
VAL_MI   = mi_for('2016-01-01')
TEST_MI  = mi_for('2021-01-01')
END_MI   = n_months - 1

print(f'\nPeriods: Train={monthly_ends[TRAIN_MI]}→{monthly_ends[VAL_MI]} | '
      f'Val={monthly_ends[VAL_MI]}→{monthly_ends[TEST_MI]} | '
      f'Test={monthly_ends[TEST_MI]}→{monthly_ends[END_MI]}')
print(f'  MI indices: Train {TRAIN_MI}→{VAL_MI} | Val {VAL_MI}→{TEST_MI} | Test {TEST_MI}→{END_MI}')


# ── STEP 1: TRAIN — IC Analysis → Factor Directions ─────────────────────────
print('\n' + '='*70)
print('STEP 1 (Train): IC Analysis — Factor Direction')
print('='*70)

all_factors = list(factor_matrix_dict.keys())
train_ic = {}
factor_direction = {}

print(f'\n{"Factor":<20} {"Mean IC":>9} {"IC>0%":>7} {"Stability":>10} {"N":>4}  Direction')
print('-'*65)

for fname in all_factors:
    ics = compute_period_ic(factor_matrix_dict[fname], ret_matrix, TRAIN_MI, VAL_MI)
    s = ic_summary(ics)
    train_ic[fname] = s

    # Direction: positive IC → long factor (rank by factor DESC), negative → short factor
    direction = 1 if s['mean'] > 0 else -1
    factor_direction[fname] = direction

    sig = '✓' if s['frac_pos'] > 0.55 and abs(s['stability']) > 0.3 else ' '
    dir_str = 'LONG' if direction == 1 else 'SHORT'
    print(f'  {fname:<20} {s["mean"]:>+9.3f} {s["frac_pos"]:>6.1%} {s["stability"]:>+10.3f} {s["n"]:>4}  {dir_str} {sig}')


# ── STEP 2: VAL — Strategy Selection ─────────────────────────────────────────
print('\n' + '='*70)
print('STEP 2 (Val): Strategy Selection — Best Val Sharpe')
print('='*70)

# Define strategy combos: each is (price_factor, [fundamental_factors], top_n)
PRICE_FACTORS = ['vol20', 'vol60', 'roc20', 'roc60', 'roc120']
FUND_FACTORS = ['roe', 'earnings_yield', 'book_per_share', 'de_ratio']
TOP_N_OPTIONS = [5, 10, 20]

strategy_combos = []

# Pure price strategies
for pf in PRICE_FACTORS:
    for top_n in TOP_N_OPTIONS:
        strategy_combos.append({
            'name': f'Price-{pf} top{top_n}',
            'factors': [pf],
            'top_n': top_n,
        })

# Multi-factor combos
for pf in ['vol20', 'vol60']:
    for ff in FUND_FACTORS:
        for top_n in TOP_N_OPTIONS:
            strategy_combos.append({
                'name': f'Multi-{pf}+{ff} top{top_n}',
                'factors': [pf, ff],
                'top_n': top_n,
            })

# Top-3 fundamental combo
for pf in ['vol20', 'vol60']:
    for top_n in TOP_N_OPTIONS:
        strategy_combos.append({
            'name': f'Multi-{pf}+top3fund top{top_n}',
            'factors': [pf] + FUND_FACTORS,
            'top_n': top_n,
        })

print(f'\nEvaluating {len(strategy_combos)} strategy combos on Val period...')

val_results = []
for combo in strategy_combos:
    _, m = run_backtest(
        combo['top_n'], combo['factors'], factor_direction,
        ret_matrix, VAL_MI, TEST_MI, list(range(n_tickers))
    )
    if m:
        val_results.append({
            'name': combo['name'],
            'factors': combo['factors'],
            'top_n': combo['top_n'],
            'val_sharpe': m['sharpe'],
            'val_ann': m['ann'],
            'val_max_dd': m['max_dd'],
            'val_win_rate': m['win_rate'],
        })

# Sort by Val Sharpe
val_results.sort(key=lambda x: x['val_sharpe'], reverse=True)

print(f'\nTop 10 by Val Sharpe:')
print(f'  {"Strategy":<35} {"Val Sharpe":>10} {"Val Ann%":>8} {"Val MaxDD":>9}')
print('  ' + '-'*65)
for r in val_results[:10]:
    print(f'  {r["name"]:<35} {r["val_sharpe"]:>10.3f} {r["val_ann"]*100:>7.1f}% {r["val_max_dd"]:>9.1%}')


# ── STEP 3: TEST — Locked Evaluation ──────────────────────────────────────────
print('\n' + '='*70)
print('STEP 3 (Test): LOCKED — No Tuning, Run Once')
print('='*70)

# Pick the best strategy from Val (TOP 1 ONLY — no peeking at Test)
best = val_results[0]
print(f'\nSelected strategy: {best["name"]}')
print(f'  Factors: {best["factors"]}')
print(f'  Top N: {best["top_n"]}')
print(f'  Val Sharpe: {best["val_sharpe"]:.3f}')
print(f'  Val Ann: {best["val_ann"]*100:.1f}%')

# Also run a few runner-ups so we can show the spread
print(f'\nAll Top-5 Val strategies on Test (for comparison, NOT for selection):')

test_results = []
for r in val_results[:5]:
    _, m = run_backtest(
        r['top_n'], r['factors'], factor_direction,
        ret_matrix, TEST_MI, END_MI, list(range(n_tickers))
    )
    if m:
        test_results.append({
            'name': r['name'],
            'val_sharpe': r['val_sharpe'],
            'test_sharpe': m['sharpe'],
            'test_ann': m['ann'],
            'test_max_dd': m['max_dd'],
            'test_win_rate': m['win_rate'],
            'test_n_months': m['n_months'],
        })

print(f'\n  {"Strategy":<35} {"Val Sharpe":>10} {"Test Sharpe":>12} {"Test Ann%":>9} {"Test MaxDD":>10} {"Test N":>5}')
print('  ' + '-'*85)
for r in test_results:
    marker = ' ★' if r['name'] == best['name'] else ''
    print(f'  {r["name"]:<35} {r["val_sharpe"]:>10.3f} {r["test_sharpe"]:>12.3f} {r["test_ann"]*100:>8.1f}% {r["test_max_dd"]:>10.1%} {r["test_n_months"]:>5}{marker}')

# SPY benchmark
_, spy_m = run_backtest(1, [], {}, ret_matrix, TEST_MI, END_MI,
                          [tickers.index('SPY')] if 'SPY' in tickers else [])
if spy_m:
    print(f'\n  SPY B&H (Test):  Sharpe={spy_m["sharpe"]:.3f}  Ann={spy_m["ann"]*100:.1f}%  MaxDD={spy_m["max_dd"]:.1%}')


# ── Save Results ──────────────────────────────────────────────────────────────
output = {
    'train_ic': {k: train_ic[k] for k in train_ic},
    'factor_direction': factor_direction,
    'val_results': val_results[:20],
    'selected_strategy': {
        'name': best['name'],
        'factors': best['factors'],
        'top_n': best['top_n'],
        'val_sharpe': best['val_sharpe'],
    },
    'test_results': test_results,
    'periods': {
        'train': {'start': str(monthly_ends[TRAIN_MI]), 'end': str(monthly_ends[VAL_MI])},
        'val': {'start': str(monthly_ends[VAL_MI]), 'end': str(monthly_ends[TEST_MI])},
        'test': {'start': str(monthly_ends[TEST_MI]), 'end': str(monthly_ends[END_MI])},
    }
}

out_path = '/tmp/three_stage_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f'\nResults saved to {out_path}')
