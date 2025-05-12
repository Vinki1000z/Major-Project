# Main script to launch the Tkinter GUI for stock market prediction

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
    freeze_support()  # Support for multiprocessing on Windows
    import pytorch_lightning as pl
    pl.seed_everything(42)  # Set random seed for reproducibility
    
    print("Launching Tkinter GUI...")
    # Create main Tkinter window
    root = tk.Tk()
    root.title("Stock Market Prediction System")
    root.geometry("900x700")

    # Create a canvas and a vertical scrollbar for scrolling the main content
    main_canvas = tk.Canvas(root, borderwidth=0, background="#f0f0f0")
    vscrollbar = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
    main_canvas.configure(yscrollcommand=vscrollbar.set)

    vscrollbar.grid(row=1, column=1, sticky='ns')
    main_canvas.grid(row=1, column=0, sticky='nsew')

    # Configure grid weights for resizing
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Create a frame inside the canvas which will contain all other widgets
    main_frame = ttk.Frame(main_canvas, padding="10")
    main_canvas.create_window((0, 0), window=main_frame, anchor="nw")

    # Update scrollregion when the frame size changes
    def on_frame_configure(event):
        main_canvas.configure(scrollregion=main_canvas.bbox("all"))

    main_frame.bind("<Configure>", on_frame_configure)

    # Input Frame inside main_frame for user inputs
    input_frame = ttk.Frame(main_frame, padding="10")
    input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

    # Label and entry for stock ticker input
    ttk.Label(input_frame, text="Enter Stock Ticker:", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5)
    ticker_entry = ttk.Entry(input_frame, width=15, font=("Arial", 12))
    ticker_entry.grid(row=0, column=1, padx=5, pady=5)
    ticker_entry.insert(0, "AAPL")  # Default ticker

    # Predict Button to trigger prediction
    predict_button = ttk.Button(input_frame, text="Predict", width=10)
    predict_button.grid(row=0, column=2, padx=5, pady=5)

    # Clear Button to reset inputs and outputs
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

    # Label to display prediction output
    output_label = ttk.Label(main_frame, text="", font=("Arial", 14))
    output_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')

    # Frame to hold the plot
    plot_frame = ttk.Frame(main_frame)
    plot_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    main_frame.grid_rowconfigure(2, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)

    # Label to display additional info like percentage change
    additional_info_label = ttk.Label(main_frame, text="", font=("Arial", 12), foreground="green")
    additional_info_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')

    # Label to display feature info like SMA and volatility
    feature_info_label = ttk.Label(main_frame, text="", font=("Arial", 12), foreground="blue")
    feature_info_label.grid(row=5, column=0, padx=10, pady=5, sticky='w')

    # News Frame and Text widget to show related news articles
    news_frame = ttk.LabelFrame(main_frame, text="Related News", padding="10")
    news_frame.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
    main_frame.grid_rowconfigure(6, weight=1)
    news_text = tk.Text(news_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
    news_text.pack(fill=tk.BOTH, expand=True)

    # Progress bar to indicate ongoing processing
    progress = ttk.Progressbar(main_frame, mode='indeterminate')
    progress.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)

    # Function called when Predict button is pressed
    def predict_and_show():
        import threading

        # Update UI with prediction results and plots
        def update_ui(initial_price, predicted_price, trends, ticker):
            dates, real_prices, predicted_prices = trends
            output_text = (f"Ticker: {ticker}\n"
                           f"Initial Price: {initial_price:.2f}\n"
                           f"Predicted Next Close Price: {predicted_price:.2f}\n\n")
            # Append output text instead of replacing
            current_text = output_label.cget("text")
            output_label.config(text=current_text + output_text)
            plot_trend(dates, real_prices, predicted_prices, ticker, plot_frame)
            
            # Additional feature: Show percentage change prediction
            try:
                percent_change = ((predicted_price - initial_price) / initial_price) * 100
                current_additional = additional_info_label.cget("text")
                additional_info_label.config(text=current_additional + f"{ticker}: {percent_change:.2f}%\n")
            except Exception:
                pass
            
            # New feature: Show simple moving average (SMA) of closing prices
            try:
                close_prices = [price for price in real_prices]
                window_size = 10
                if len(close_prices) >= window_size:
                    sma = np.convolve(close_prices, np.ones(window_size)/window_size, mode='valid')[-1]
                    current_feature = feature_info_label.cget("text")
                    feature_info_label.config(text=current_feature + f"{ticker}: 10-day SMA: {sma:.2f}\n")
                else:
                    current_feature = feature_info_label.cget("text")
                    feature_info_label.config(text=current_feature + f"{ticker}: Not enough data for SMA\n")
            except Exception:
                pass
            
            # New feature: Show volatility (standard deviation) of closing prices
            try:
                volatility = np.std(real_prices)
                current_feature = feature_info_label.cget("text")
                feature_info_label.config(text=current_feature + f"{ticker}: Volatility: {volatility:.2f}\n")
            except Exception:
                pass
            
            # New feature: Fetch and display related news
            try:
                news_items = fetch_news(ticker)
                news_text.config(state=tk.NORMAL)
                if news_items:
                    # Append news instead of replacing
                    for item in news_items[:5]:  # Show top 5 news items
                        news_text.insert(tk.END, f"Ticker: {ticker}\n")
                        news_text.insert(tk.END, f"• {item.get('title', 'No Title')}\n")
                        news_text.insert(tk.END, f"  {item.get('publisher', '')} - {item.get('publishedAt', '')}\n")
                        news_text.insert(tk.END, f"  {item.get('link', '')}\n\n")
                else:
                    news_text.insert(tk.END, f"No news found for ticker: {ticker}\n")
                news_text.config(state=tk.DISABLED)
            except Exception as e:
                news_text.config(state=tk.NORMAL)
                news_text.insert(tk.END, f"Error fetching news for ticker {ticker}: {e}\n")
                news_text.config(state=tk.DISABLED)

        # Background task to run prediction and update UI
        def task():
            ticker_input = ticker_entry.get().strip().upper()
            if not ticker_input:
                output_label.config(text="Please enter at least one valid ticker symbol.")
                additional_info_label.config(text="")
                feature_info_label.config(text="")
                news_text.config(state=tk.NORMAL)
                news_text.delete(1.0, tk.END)
                news_text.config(state=tk.DISABLED)
                progress.stop()
                return

            tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
            if not tickers:
                output_label.config(text="Please enter at least one valid ticker symbol.")
                additional_info_label.config(text="")
                feature_info_label.config(text="")
                news_text.config(state=tk.NORMAL)
                news_text.delete(1.0, tk.END)
                news_text.config(state=tk.DISABLED)
                progress.stop()
                return

            # Clear previous outputs
            output_label.config(text="")
            additional_info_label.config(text="")
            feature_info_label.config(text="")
            news_text.config(state=tk.NORMAL)
            news_text.delete(1.0, tk.END)
            news_text.config(state=tk.DISABLED)
            root.update()

            import time
            # Run prediction multiple times for demo purposes
            for _ in range(6):  # Run 6 times (1 minute) for demo; adjust as needed
                for ticker in tickers:
                    output_label.config(text=output_label.cget("text") + f"Processing {ticker}...\n")
                    initial_price, predicted_price, trends = run_prediction(ticker)
                    if initial_price is None:
                        output_label.config(text=output_label.cget("text") + f"Error fetching or processing data for ticker: {ticker}\n")
                        continue
                    root.after(0, update_ui, initial_price, predicted_price, trends, ticker)
                    time.sleep(10)  # Wait 10 seconds before next update
            progress.stop()

        progress.start()
        threading.Thread(target=task).start()

    # Bind the predict button to the predict_and_show function
    predict_button.config(command=predict_and_show)

    print("GUI is running. Please interact with the window.")
    root.mainloop()
