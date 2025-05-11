import pandas as pd
from data import fetch_stock_data_polygon

API_KEY = "rVj_WzgoKW5RMUAd_l3f8GPARg4CMvJc"

def test_polygon_fetch():
    ticker = "AAPL"
    try:
        df = fetch_stock_data_polygon(ticker=ticker, api_key=API_KEY)
        print(f"Fetched {len(df)} rows for ticker {ticker} from polygon.io")
        print(df.head())
    except Exception as e:
        print(f"Error fetching data from polygon.io: {e}")

if __name__ == "__main__":
    test_polygon_fetch()
