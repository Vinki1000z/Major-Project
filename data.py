"""
Data fetching and preprocessing module.
Includes functions to fetch stock data, preprocess it, and dataset class.
"""

import yfinance as yf
import pandas as pd
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
