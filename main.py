from data import fetch_stock_data, preprocess_data, StockDataset
from model import TemporalFusionTransformer
from predict import run_prediction
from plot import plot_trend
from news import fetch_news

import tkinter as tk
from tkinter import ttk
from multiprocessing import freeze_support
import numpy as np

if __name__ == "__main__":
    freeze_support()
    import pytorch_lightning as pl
    pl.seed_everything(42)
    
    print("Launching Tkinter GUI...")
    # Create main Tkinter window
    root = tk.Tk()
    root.title("Stock Market Prediction System")
    root.geometry("900x700")
    
    # Input Frame
    input_frame = ttk.Frame(root, padding="10")
    input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    ttk.Label(input_frame, text="Enter Stock Ticker:", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5)
    ticker_entry = ttk.Entry(input_frame, width=15, font=("Arial", 12))
    ticker_entry.grid(row=0, column=1, padx=5, pady=5)
    ticker_entry.insert(0, "AAPL")  # Default ticker
    
    # Predict Button
    predict_button = ttk.Button(input_frame, text="Predict", width=10)
    predict_button.grid(row=0, column=2, padx=5, pady=5)
    
    # Clear Button
    def clear_all():
        ticker_entry.delete(0, tk.END)
        output_label.config(text="")
        additional_info_label.config(text="")
        feature_info_label.config(text="")
        news_text.config(state=tk.NORMAL)
        news_text.delete(1.0, tk.END)
        news_text.config(state=tk.DISABLED)
        for widget in plot_frame.winfo_children():
            widget.destroy()
    clear_button = ttk.Button(input_frame, text="Clear", width=10, command=clear_all)
    clear_button.grid(row=0, column=3, padx=5, pady=5)
    
    # Output Label
    output_label = ttk.Label(root, text="", font=("Arial", 14))
    output_label.grid(row=1, column=0, padx=10, pady=10)
    
    # Frame for plot
    plot_frame = ttk.Frame(root)
    plot_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.grid_rowconfigure(2, weight=1)
    root.grid_columnconfigure(0, weight=1)
    
    # Additional Info Label
    additional_info_label = ttk.Label(root, text="", font=("Arial", 12), foreground="green")
    additional_info_label.grid(row=4, column=0, padx=10, pady=5)

    # Feature Info Label
    feature_info_label = ttk.Label(root, text="", font=("Arial", 12), foreground="blue")
    feature_info_label.grid(row=5, column=0, padx=10, pady=5)

    # News Frame and Text widget
    news_frame = ttk.LabelFrame(root, text="Related News", padding="10")
    news_frame.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
    root.grid_rowconfigure(6, weight=1)
    news_text = tk.Text(news_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
    news_text.pack(fill=tk.BOTH, expand=True)

    # Progress bar
    progress = ttk.Progressbar(root, mode='indeterminate')
    progress.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)

    # Function called when Predict button is pressed
    def predict_and_show():
        import threading

        def update_ui(initial_price, predicted_price, trends, ticker):
            real_prices, predicted_prices = trends
            output_text = (f"Ticker: {ticker}\n"
                           f"Initial Price: {initial_price:.2f}\n"
                           f"Predicted Next Close Price: {predicted_price:.2f}")
            output_label.config(text=output_text)
            plot_trend(real_prices, predicted_prices, ticker, plot_frame)
            
            # Additional feature: Show percentage change prediction
            try:
                percent_change = ((predicted_price - initial_price) / initial_price) * 100
                additional_info_label.config(text=f"Predicted Change: {percent_change:.2f}%")
            except Exception:
                additional_info_label.config(text="")
            
            # New feature: Show simple moving average (SMA) of closing prices
            try:
                close_prices = [price for price in real_prices]
                window_size = 10
                if len(close_prices) >= window_size:
                    sma = np.convolve(close_prices, np.ones(window_size)/window_size, mode='valid')[-1]
                    feature_info_label.config(text=f"10-day SMA: {sma:.2f}")
                else:
                    feature_info_label.config(text="Not enough data for SMA")
            except Exception:
                feature_info_label.config(text="")
            
            # New feature: Show volatility (standard deviation) of closing prices
            try:
                volatility = np.std(real_prices)
                feature_info_label.config(text=feature_info_label.cget("text") + f" | Volatility: {volatility:.2f}")
            except Exception:
                pass
            
            # New feature: Fetch and display related news
            try:
                news_items = fetch_news(ticker)
                news_text.config(state=tk.NORMAL)
                if news_items:
                    news_text.delete(1.0, tk.END)
                    for item in news_items[:5]:  # Show top 5 news items
                        news_text.insert(tk.END, f"• {item.get('title', 'No Title')}\n")
                        news_text.insert(tk.END, f"  {item.get('publisher', '')} - {item.get('publishedAt', '')}\n")
                        news_text.insert(tk.END, f"  {item.get('link', '')}\n\n")
                else:
                    news_text.insert(tk.END, "No news found for this ticker.")
                news_text.config(state=tk.DISABLED)
            except Exception as e:
                news_text.config(state=tk.NORMAL)
                news_text.insert(tk.END, f"Error fetching news: {e}")
                news_text.config(state=tk.DISABLED)

        def task():
            ticker = ticker_entry.get().strip().upper()
            if not ticker:
                output_label.config(text="Please enter a valid ticker symbol.")
                additional_info_label.config(text="")
                feature_info_label.config(text="")
                news_text.config(state=tk.NORMAL)
                news_text.delete(1.0, tk.END)
                news_text.config(state=tk.DISABLED)
                progress.stop()
                return
            output_label.config(text=f"Processing {ticker}...")
            additional_info_label.config(text="")
            feature_info_label.config(text="")
            news_text.config(state=tk.NORMAL)
            news_text.delete(1.0, tk.END)
            news_text.config(state=tk.DISABLED)
            root.update()
            
            # Real-time update: run prediction every 10 seconds
            import time
            for _ in range(6):  # Run 6 times (1 minute) for demo; adjust as needed
                initial_price, predicted_price, trends = run_prediction(ticker)
                if initial_price is None:
                    output_label.config(text="Error fetching or processing data for ticker: " + ticker)
                    additional_info_label.config(text="")
                    feature_info_label.config(text="")
                    news_text.config(state=tk.NORMAL)
                    news_text.delete(1.0, tk.END)
                    news_text.config(state=tk.DISABLED)
                    progress.stop()
                    return
                
                root.after(0, update_ui, initial_price, predicted_price, trends, ticker)
                time.sleep(10)  # Wait 10 seconds before next update
            
            progress.stop()

        progress.start()
        threading.Thread(target=task).start()

    predict_button.config(command=predict_and_show)

    print("GUI is running. Please interact with the window.")
    root.mainloop()
