import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from data.tiingo_data_fetcher import DataFetcher
from data.csv_loader import CSVLoader
from models.model_factory import ModelFactory
from utils.plotting import plot_forecast
from utils.evaluate import evaluate_model
from utils.visuals import print_colored

# ---- Configuration & Environment ---------------------------------------------
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = "data/sets"
ARTIFACTS_DIR = "data/artifacts"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---- Data Loading & Selection -----------------------------------------------
def load_network_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame({
        "date": dates,
        "active_addresses": np.random.randint(1000, 10000, size=len(dates)),
        "transaction_volume": np.random.uniform(1e6, 1e7, size=len(dates))
    })
    return df

def select_data(fetcher, source=None, file_path=None):
    default_end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if source is None:
        print("Select the data source:")
        print("1. Tiingo Stock Data")
        print("2. Tiingo Crypto Data")
        print("3. Load data from CSV file")
        selection = input("Enter your choice (1/2/3): ").strip()
    else:
        selection = {"stock": "1", "crypto": "2", "csv": "3"}.get(source, "3")

    if selection == "1":
        symbol = input("Enter stock symbol (default: AAPL): ").strip() or "AAPL"
        frequency = input("Enter frequency (daily, etc.): ").strip() or "daily"
        start_date = input("Start date (YYYY-MM-DD): ").strip() or "2021-01-01"
        end_date = input(f"End date (YYYY-MM-DD, default: {default_end_date}): ").strip() or default_end_date
        return fetcher.fetch_tiingo_stock_data(symbol, start_date, end_date, frequency)

    elif selection == "2":
        symbol = input("Enter crypto symbol (default: btcusd): ").strip() or "btcusd"
        frequency = input("Enter frequency (1min, 5min, 15min, 1hour, 4hour, 1day): ").strip() or "1day"

        valid_frequencies = ["1min", "5min", "15min", "1hour", "4hour", "1day"]
        if frequency not in valid_frequencies:
            print_colored(f"Invalid frequency '{frequency}', defaulting to '1day'", "warn")
            frequency = "1day"

        start_date = input("Start date (YYYY-MM-DD): ").strip() or "2021-01-01"
        end_date = input(f"End date (YYYY-MM-DD, default: {default_end_date}): ").strip() or default_end_date

        try:
            data = fetcher.fetch_tiingo_crypto_data(symbol, start_date, end_date, frequency)
            if data.empty:
                raise ValueError("No crypto Tiingo data found.")

            file_name = f"{symbol.upper()}_{start_date}_to_{end_date}_{frequency}.csv"
            file_path = os.path.join(DATA_DIR, file_name)
            data.to_csv(file_path, index=False)
            print(f"Saving crypto data to {file_path}...")

            network_data = load_network_data(symbol, start_date, end_date)
            if not network_data.empty:
                data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
                network_data["date"] = pd.to_datetime(network_data["date"])
                data = pd.merge(data, network_data, on="date", how="left")
                print("Merged network data into price dataset.")

            return data
        except Exception as e:
            print_colored(f"Error fetching crypto data: {e}", "error")
            sys.exit(1)

    elif selection == "3":
        if file_path is None:
            file_path = input("Enter CSV file path: ").strip()
        return CSVLoader.load_csv(file_path)

    print_colored("Invalid data selection", "error")
    sys.exit(1)

# ---- Evaluation & Confidence ------------------------------------------------
def calculate_confidence(forecast):
    std_value = forecast.std().mean()
    confidence = max(0.85, min(0.99, 1 - std_value))
    return round(confidence, 2)

# ---- Saving -----------------------------------------------------------------
def save_artifact(model, forecast, data, model_type):
    confidence = calculate_confidence(forecast)
    result = {
        "model": model_type,
        "forecast": forecast.tolist(),
        "confidence": confidence,
        "latest_data": data.iloc[-1].to_dict()
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ARTIFACTS_DIR, f"{model_type}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print_colored(f"Saved forecast and confidence to {path}", "info")

# ---- Main -------------------------------------------------------------------
def main():
    fetcher = DataFetcher()
    data = select_data(fetcher)

    print("Data validation and preprocessing completed successfully.")

    print("Select models to train:")
    print("1. All models")
    print("2. Custom selection")
    choice = input("Enter choice (1/2): ").strip()

    all_models = [
        "regression", "regression_time_series",
        "random_forest", "random_forest_time_series",
        "xgboost", "xgboost_time_series",
        "prophet", "arima", "lstm"
    ]

    selected_models = all_models
    if choice == "2":
        print("Available models:")
        for i, m in enumerate(all_models, 1):
            print(f"{i}. {m}")
        indices = input("Enter numbers (comma-separated): ").strip().split(',')
        selected_models = [all_models[int(i)-1] for i in indices if i.isdigit() and 0 < int(i) <= len(all_models)]

    for model_type in selected_models:
        print(f"Training {model_type}...")
        model = ModelFactory.create(model_type)
        forecast = model.train_and_forecast(data)
        save_artifact(model, forecast, data, model_type)
        plot_forecast(data, forecast, title=model_type)
        evaluate_model(data, forecast)

if __name__ == "__main__":
    main()
