import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import StandardScaler
from multiprocessing import freeze_support

# Set matmul precision to utilize Tensor Cores optimally
torch.set_float32_matmul_precision('high')

# Hyperparameters
INPUT_SIZE = 5   # Number of features: Open, High, Low, Close, Volume
HIDDEN_SIZE = 128
NUM_LAYERS = 3
OUTPUT_SIZE = 1   # Predicting closing price
SEQ_LENGTH = 30  # Use past 30 days to predict the next day's closing price
BATCH_SIZE = 32
EPOCHS = 2       # Reduced epochs for debugging; increase once debugging is complete
LEARNING_RATE = 0.001
NUM_WORKERS = 2

# Fetch real-time stock data using yfinance
def fetch_stock_data(ticker="AAPL", period="1y", interval="1d"):
    print(f"Fetching data for ticker: {ticker}")
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.dropna(inplace=True)
    print(f"Data fetched: {len(data)} rows.")
    return data

# Preprocess data: Standardize each feature separately
def preprocess_data(data):
    print("Preprocessing data...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print("Data preprocessing complete.")
    return data_scaled, scaler

# Custom Dataset for stock data
class StockDataset(Dataset):
    def __init__(self, data, seq_length=SEQ_LENGTH):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        # Predict the closing price (4th column: index 3) of the next day
        y = self.data[index + self.seq_length, 3]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define a TFT-like model using LSTM + Multihead Attention
class TemporalFusionTransformer(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_out[:, -1, :])
        return out.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        if batch_idx % 10 == 0:
            print(f"Train step {batch_idx}, Loss: {loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

# Run prediction and training for a given ticker
def run_prediction(ticker):
    print(f"Starting prediction pipeline for ticker: {ticker}")
    # Fetch and preprocess data
    raw_data = fetch_stock_data(ticker=ticker)
    if raw_data.empty:
        print("No data found for ticker:", ticker)
        return None, None, None
    data_scaled, scaler = preprocess_data(raw_data)
    
    # Create dataset and check if enough data is available
    dataset = StockDataset(data_scaled)
    if len(dataset) < 100:
        print("Not enough data for training.")
        return None, None, None

    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True)

    # Model checkpointing callback
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    
    # Initialize and train the model
    model = TemporalFusionTransformer(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1, enable_checkpointing=True, callbacks=[checkpoint_callback], accelerator="auto")
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training complete.")

    # Retrieve the initial closing price from raw data
    initial_price = raw_data['Close'].iloc[0]
    
    # Predict using the most recent SEQ_LENGTH days from the scaled data
    test_sample = torch.tensor(data_scaled[:SEQ_LENGTH], dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        predicted_scaled = model(test_sample).item()
    
    # Inverse transform using only the 'Close' feature (index 3)
    close_mean = scaler.mean_[3]
    close_std = np.sqrt(scaler.var_[3])
    predicted_price = predicted_scaled * close_std + close_mean
    print(f"Predicted next closing price: {predicted_price:.2f}")

    # Generate predictions over the dataset to plot trend
    real_prices = []
    predicted_prices = []
    for i in range(len(data_scaled) - SEQ_LENGTH):
        sample = torch.tensor(data_scaled[i:i+SEQ_LENGTH], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(sample).item()
        price = pred * close_std + close_mean
        predicted_prices.append(price)
        real_price = data_scaled[i + SEQ_LENGTH, 3] * close_std + close_mean
        real_prices.append(real_price)
    
    return initial_price, predicted_price, (real_prices, predicted_prices)

# Plot the trend: actual vs. predicted closing prices
def plot_trend(real_prices, predicted_prices, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(real_prices, label="Actual Closing Prices", color='blue')
    plt.plot(predicted_prices, label="Predicted Closing Prices", color='red', linestyle='dashed')
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title(f"Stock Price Trend for {ticker}")
    plt.legend()
    plt.show()

# Tkinter GUI for user interaction
if __name__ == "__main__":
    freeze_support()
    pl.seed_everything(42)
    
    print("Launching Tkinter GUI...")
    # Create main Tkinter window
    root = tk.Tk()
    root.title("Stock Market Prediction System")
    
    # Input Frame
    input_frame = ttk.Frame(root, padding="10")
    input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    ttk.Label(input_frame, text="Enter Stock Ticker:").grid(row=0, column=0, padx=5, pady=5)
    ticker_entry = ttk.Entry(input_frame, width=10)
    ticker_entry.grid(row=0, column=1, padx=5, pady=5)
    ticker_entry.insert(0, "AAPL")  # Default ticker
    
    # Output Label
    output_label = ttk.Label(root, text="", font=("Arial", 14))
    output_label.grid(row=2, column=0, padx=10, pady=10)
    
    # Function called when Predict button is pressed
    def predict_and_show():
        ticker = ticker_entry.get().strip().upper()
        if not ticker:
            output_label.config(text="Please enter a valid ticker symbol.")
            return
        output_label.config(text=f"Processing {ticker}...")
        root.update()
        
        initial_price, predicted_price, trends = run_prediction(ticker)
        if initial_price is None:
            output_label.config(text="Error fetching or processing data for ticker: " + ticker)
            return
        
        real_prices, predicted_prices = trends
        output_text = (f"Ticker: {ticker}\n"
                       f"Initial Price: {initial_price:.2f}\n"
                       f"Predicted Next Close Price: {predicted_price:.2f}")
        output_label.config(text=output_text)
        plot_trend(real_prices, predicted_prices, ticker)
    
    # Predict Button
    predict_button = ttk.Button(input_frame, text="Predict", command=predict_and_show)
    predict_button.grid(row=0, column=2, padx=5, pady=5)
    
    print("GUI is running. Please interact with the window.")
    root.mainloop()
