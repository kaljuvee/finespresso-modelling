
import yfinance as yf
import pandas as pd

def get_price_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    ticker = 'AAPL'
    start_date = '2015-01-01'
    end_date = '2025-01-01'
    price_data = get_price_data(ticker, start_date, end_date)
    if price_data is not None:
        print(price_data.head())


