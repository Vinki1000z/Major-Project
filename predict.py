"""
Prediction and training module.
Contains the run_prediction function to train the model and generate predictions.
"""

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import numpy as np
import datetime
import os
from data import StockDataset, preprocess_data, fetch_stock_data_polygon
from model import TemporalFusionTransformer

# Hyperparameters
INPUT_SIZE = 5
HIDDEN_SIZE = 128
NUM_LAYERS = 3
OUTPUT_SIZE = 1
SEQ_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.001
NUM_WORKERS = 2

def run_prediction(ticker):
    """
    Run the full prediction pipeline:
    - Fetch and preprocess data
    - Train the TemporalFusionTransformer model
    - Generate predictions for plotting and next day price
    Returns initial price, predicted price, and (dates, real_prices, predicted_prices) for trend plotting.
    """
    print(f"Starting prediction pipeline for ticker: {ticker}")

    # Mapping user tickers to Polygon.io tickers with exchange prefix
    ticker_mapping = {
        "TATA": "NSE:TATA",
        "RELIANCE": "NSE:RELIANCE",
        "INFY": "NSE:INFY",
        "TCS": "NSE:TCS",
        "HDFCBANK": "NSE:HDFCBANK",
        # Add more mappings as needed
    }
    polygon_ticker = ticker_mapping.get(ticker, ticker)  # Use mapped ticker or original if not found

    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

    print(f"Using Polygon.io ticker: {polygon_ticker}")

    try:
        raw_data = fetch_stock_data_polygon(ticker=polygon_ticker, start_date=start_date, end_date=end_date, api_key=POLYGON_API_KEY)
        if raw_data.empty:
            print("No data found for ticker on Polygon.io:", polygon_ticker)
            raise Exception("Empty data from Polygon.io")
    except Exception as e:
        print(f"Polygon.io fetch failed: {e}. Falling back to yfinance.")
        from data import fetch_stock_data
        raw_data = fetch_stock_data(ticker=ticker, period="1y", interval="1d")
        if raw_data.empty:
            print("No data found for ticker on yfinance:", ticker)
            return None, None, None
    data_scaled, scaler = preprocess_data(raw_data)

    dataset = StockDataset(data_scaled, seq_length=SEQ_LENGTH)
    if len(dataset) < 100:
        print("Not enough data for training.")
        return None, None, None

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')

    model = TemporalFusionTransformer(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1, enable_checkpointing=True, callbacks=[checkpoint_callback], accelerator="auto")
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training complete.")

    initial_price = raw_data['Close'].iloc[0]

    test_sample = torch.tensor(data_scaled[:SEQ_LENGTH], dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        predicted_scaled = model(test_sample).item()

    close_mean = scaler.mean_[3]
    close_std = scaler.scale_[3]
    predicted_price = predicted_scaled * close_std + close_mean
    print(f"Predicted next closing price: {predicted_price:.2f}")

    real_prices = []
    predicted_prices = []
    print("First 10 raw_data['Close'] values:")
    print(raw_data['Close'].head(10).values)
    for i in range(len(data_scaled) - SEQ_LENGTH):
        sample = torch.tensor(data_scaled[i:i+SEQ_LENGTH], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(sample).item()
        price = pred * close_std + close_mean
        predicted_prices.append(price)
        real_price = data_scaled[i + SEQ_LENGTH, 3] * close_std + close_mean
        real_prices.append(real_price)
    print("First 10 real_prices values:")
    print(real_prices[:10])

    dates = raw_data.index[SEQ_LENGTH:]
    return initial_price, predicted_price, (dates, real_prices, predicted_prices)
