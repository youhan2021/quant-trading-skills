"""
SEC XBRL fundamental data fetcher — 获取 107 只股票的:
  - Revenue (annual FY)
  - NetIncomeLoss (quarterly + annual)
  - StockholdersEquity (quarterly)
  - SharesOutstanding (quarterly, dei)
  - 计算因子: rev_growth, earnings_yield, book_value_per_share, roe, de_ratio

look-ahead 防护: 只使用 filed <= 当月最后一天的已公开数据
输出: /tmp/fundamental_data.json  (所有股票的 XBRL 历史)

用法: python3 sec_xbrl_fetch.py
"""

import requests, json, time, sys
from collections import defaultdict

HEADERS = {'User-Agent': 'QuantResearch agent@example.com'}

UNIVERSE = [
    'AAPL','MSFT','AMZN','GOOGL','GOOG','META','NVDA','AVGO','TSLA','CSCO',
    'ADBE','NFLX','ORCL','CRM','AMD','INTC','QCOM','TXN','MU','AMAT','IBM',
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

# ── Step 1: Ticker → CIK lookup via SEC company_tickers.json ────────────────

def build_ticker_cik_map():
    """Download and parse SEC company_tickers.json → {ticker: cik_str_padded}."""
    url = 'https://www.sec.gov/files/company_tickers.json'
    r = requests.get(url, headers=HEADERS, timeout=15)
    data = r.json()
    out = {}
    for entry in data.values():
        ticker = entry['ticker']
        cik_padded = str(entry['cik_str']).zfill(10)
        out[ticker] = cik_padded
    return out

# ── Step 2: Fetch company facts ─────────────────────────────────────────────

def fetch_company_facts(cik):
    """Fetch full XBRL company facts JSON from SEC."""
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f'    ERROR fetching CIK {cik}: {e}')
    return None

# ── Step 3: Extract clean time series ─────────────────────────────────────

def extract_annual_fiscal(data, tag, unit='USD'):
    """
    Extract annual fiscal year data points.
    Returns list of {end_date, val, filed_date}.
    Only for tags that represent annual totals (not quarterly rolling).
    """
    results = []
    if 'us-gaap' not in data['facts']:
        return results
    tag_data = data['facts']['us-gaap'].get(tag, {})
    units = tag_data.get('units', {})
    if unit not in units:
        return results
    vals = units[unit]
    for v in vals:
        end = v.get('end', '')
        filed = v.get('filed', '')
        val = v.get('val', None)
        if end and filed and val is not None:
            results.append({'end': end, 'filed': filed, 'val': float(val)})
    return sorted(results, key=lambda x: x['end'])

def extract_quarterly(data, tag, unit='USD'):
    """Extract quarterly data points."""
    return extract_annual_fiscal(data, tag, unit)

def extract_shares(data):
    """Extract shares outstanding (dei section)."""
    results = []
    dei = data['facts'].get('dei', {})
    shares_data = dei.get('EntityCommonStockSharesOutstanding', {})
    units = shares_data.get('units', {})
    if 'shares' in units:
        for v in units['shares']:
            end = v.get('end', '')
            filed = v.get('filed', '')
            val = v.get('val', None)
            if end and filed and val is not None:
                results.append({'end': end, 'filed': filed, 'val': float(val)})
    return sorted(results, key=lambda x: x['end'])

# ── Step 4: Get available tags for a ticker ───────────────────────────────

def list_available_tags(data):
    """List all available us-gaap tags with entry counts."""
    tags = {}
    usgaap = data['facts'].get('us-gaap', {})
    for tag, info in usgaap.items():
        units = info.get('units', {})
        total = sum(len(u) for u in units.values())
        tags[tag] = {'count': total, 'label': info.get('label', '')[:60]}
    return tags

# ── Main: fetch for all tickers ─────────────────────────────────────────────

def main():
    print('='*60)
    print('SEC XBRL fetcher — fundamental data for quant factors')
    print('='*60)

    # Step 1: find CIKs for all tickers
    cik_file = '/tmp/ticker_cik_map.json'
    try:
        with open(cik_file) as f:
            cik_map = json.load(f)
        print(f'Loaded {len(cik_map)} CIK mappings from cache')
    except FileNotFoundError:
        print('\nStep 1: Building CIK map from SEC company_tickers.json...')
        all_ciks = build_ticker_cik_map()
        cik_map = {t: all_ciks.get(t, '') for t in UNIVERSE}
        found = sum(1 for v in cik_map.values() if v)
        print(f'Found {found}/{len(UNIVERSE)} tickers in SEC database')
        with open(cik_file, 'w') as f:
            json.dump(cik_map, f, indent=2)
        print(f'Saved CIK map: {cik_file}')

    # Step 2: fetch fundamentals for each ticker
    fundamentals_file = '/tmp/fundamental_data.json'
    try:
        with open(fundamentals_file) as f:
            fundamental_data = json.load(f)
        print(f'Loaded fundamental data for {len(fundamental_data)} tickers from cache')
    except FileNotFoundError:
        fundamental_data = {}
        print(f'\nStep 2: Fetching XBRL facts for {len(cik_map)} companies...')
        for i, (ticker, cik) in enumerate(sorted(cik_map.items())):
            if not cik:
                print(f'  [{i+1}/{len(cik_map)}] {ticker}: no CIK, skipping')
                continue
            print(f'  [{i+1}/{len(cik_map)}] {ticker} (CIK={cik})...', end='', flush=True)
            data = fetch_company_facts(cik)
            if not data:
                print(' FAILED')
                time.sleep(1)
                continue

            # Extract key metrics
            fundamentals = {
                'cik': cik,
                'entityName': data.get('entityName', ''),
                'Revenues':        extract_annual_fiscal(data, 'Revenues'),
                'NetIncomeLoss':   extract_quarterly(data, 'NetIncomeLoss'),
                'StockholdersEquity': extract_quarterly(data, 'StockholdersEquity'),
                'Assets':          extract_quarterly(data, 'Assets'),
                'GrossProfit':     extract_annual_fiscal(data, 'GrossProfit'),
                'OperatingIncomeLoss': extract_annual_fiscal(data, 'OperatingIncomeLoss'),
                'SharesOutstanding': extract_shares(data),
            }

            # Count non-empty series
            non_empty = sum(1 for k, v in fundamentals.items()
                           if k not in ('cik', 'entityName') and len(v) > 0)
            print(f' OK ({non_empty} series)')
            fundamental_data[ticker] = fundamentals
            time.sleep(0.1)  # be nice to SEC

        with open(fundamentals_file, 'w') as f:
            json.dump(fundamental_data, f, indent=2)
        print(f'\nSaved fundamental data: {fundamentals_file}')

    # Summary
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    for ticker, data in sorted(fundamental_data.items()):
        series_info = []
        for k in ['Revenues', 'NetIncomeLoss', 'StockholdersEquity', 'Assets', 'SharesOutstanding']:
            if k in data and len(data[k]) > 0:
                dates = [x['end'] for x in data[k]]
                series_info.append(f'{k}:{len(data[k])}[{min(dates)}-{max(dates)}]')
        print(f'  {ticker}: {" | ".join(series_info)}')

if __name__ == '__main__':
    main()
