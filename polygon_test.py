import pandas as pd
from data import fetch_stock_data_polygon

# API key for polygon.io
API_KEY = "rVj_WzgoKW5RMUAd_l3f8GPARg4CMvJc"

# Test function to fetch stock data from polygon.io
def test_polygon_fetch():
    ticker = "AAPL"
    try:
        # Fetch data for given ticker
        df = fetch_stock_data_polygon(ticker=ticker, api_key=API_KEY)
        print(f"Fetched {len(df)} rows for ticker {ticker} from polygon.io")
        print(df.head())
    except Exception as e:
        print(f"Error fetching data from polygon.io: {e}")

if __name__ == "__main__":
    # Run test function if script is executed directly
    test_polygon_fetch()
