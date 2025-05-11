"""
Data fetching and preprocessing module.
Includes functions to fetch stock data, preprocess it, and dataset class.
"""

import yfinance as yf
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

def fetch_stock_data(ticker="AAPL", period="1y", interval="1d"):
    """
    Fetch historical stock data for the given ticker using yfinance.
    Returns a DataFrame with Open, High, Low, Close, Volume columns.
    """
    print(f"Fetching data for ticker: {ticker}")
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.dropna(inplace=True)
    print(f"Data fetched: {len(data)} rows.")
    return data

def fetch_stock_data_polygon(ticker="AAPL", start_date=None, end_date=None, api_key=None):
    """
    Fetch historical stock data for the given ticker using polygon.io API.
    Returns a DataFrame with Open, High, Low, Close, Volume columns.
    Parameters:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format. If None, defaults to 1 year ago.
        end_date (str): End date in 'YYYY-MM-DD' format. If None, defaults to today.
        api_key (str): Polygon.io API key.
    """
    import datetime
    if api_key is None:
        raise ValueError("API key for polygon.io must be provided.")
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.datetime.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

    print(f"Fetching data for ticker: {ticker} from {start_date} to {end_date} using polygon.io")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Polygon.io API request failed with status code {response.status_code}: {response.text}")

    data_json = response.json()
    if 'results' not in data_json:
        raise Exception(f"No results found in Polygon.io response: {data_json}")

    results = data_json['results']
    df = pd.DataFrame(results)
    # Rename columns to match yfinance format
    df.rename(columns={
        'o': 'Open',
        'h': 'High',
        'l': 'Low',
        'c': 'Close',
        'v': 'Volume',
        't': 'Timestamp'
    }, inplace=True)
    # Convert timestamp to datetime
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    print(f"Data fetched: {len(df)} rows.")
    return df

def preprocess_data(data):
    """
    Standardize each feature in the data using StandardScaler.
    Returns scaled data and the scaler object.
    """
    print("Preprocessing data...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print("Data preprocessing complete.")
    return data_scaled, scaler

class StockDataset(Dataset):
    """
    Custom PyTorch Dataset for stock data sequences.
    Each sample consists of a sequence of features and the next day's closing price.
    """
    def __init__(self, data, seq_length=30):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + self.seq_length, 3]  # Closing price index
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
