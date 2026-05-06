"""
SPX Constituent Fetcher
从 Wikipedia 获取当前 S&P 500 成分股列表
"""
import pandas as pd
import requests
from io import StringIO

def get_spx_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]
    tickers = df['Symbol'].tolist()
    # Clean up
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers

if __name__ == '__main__':
    tickers = get_spx_tickers()
    print(f"SPX tickers: {len(tickers)}")
    print(tickers[:20])
    with open('/tmp/spx_tickers.txt', 'w') as f:
        f.write('\n'.join(sorted(tickers)))
    print(f"Saved {len(tickers)} tickers to /tmp/spx_tickers.txt")
