# ---- Imports -----------------------------------------------------------
import os
import pandas as pd
import requests
from dotenv import load_dotenv

# ---- Environment Loading --------------------------------------------------
# Load the .env.local file if it exists, otherwise load .env
env_path = ".env.local" if os.path.exists(".env.local") else ".env"
print(f"Loading {env_path} file...")
load_dotenv(dotenv_path=env_path, override=True)

# ---- API Config -----------------------------------------------------------
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
BASE_APIURL = "https://api.tiingo.com"

# ---- DataFetcher Class ----------------------------------------------------
class DataFetcher:
    """
    A class to fetch and normalize data for stocks and cryptocurrencies from Tiingo.
    """

    def __init__(self, cache_folder="data/sets"):
        self.cache_folder = cache_folder
        os.makedirs(self.cache_folder, exist_ok=True)

    def _generate_filename(self, symbol, start_date, end_date, frequency):
        return os.path.join(
            self.cache_folder, f"{symbol}_{start_date}_to_{end_date}_{frequency}.csv"
        )

    def fetch_tiingo_stock_data(self, symbol, start_date, end_date, frequency="daily", use_cache=True):
        filename = self._generate_filename(symbol, start_date, end_date, frequency)

        if use_cache and os.path.exists(filename):
            print(f"Loading stock data from {filename}...")
            return pd.read_csv(filename)

        url = f"{BASE_APIURL}/tiingo/daily/{symbol}/prices"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {TIINGO_API_KEY}",
        }
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "resampleFreq": frequency,
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching stock data: {e}")
            return pd.DataFrame()

        if not data:
            print(f"No stock data returned for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df.dropna(inplace=True)

        if use_cache:
            print(f"Saving stock data to {filename}...")
            df.to_csv(filename, index=False)

        return df

    def fetch_tiingo_crypto_data(self, symbol, start_date, end_date, frequency="5min", use_cache=True):
        filename = self._generate_filename(symbol, start_date, end_date, frequency)

        if use_cache and os.path.exists(filename):
            print(f"Loading crypto data from {filename}...")
            return pd.read_csv(filename)

        url = f"{BASE_APIURL}/tiingo/crypto/prices"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {TIINGO_API_KEY}",
        }
        params = {
            "tickers": symbol,
            "startDate": start_date,
            "endDate": end_date,
            "resampleFreq": frequency,
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                print(f"No crypto Tiingo data returned for {symbol}")
                return pd.DataFrame()

            records = []
            for asset in data:
                price_data = asset.get("priceData", [])
                for entry in price_data:
                    records.append({
                        "date": entry.get("date"),
                        "open": entry.get("open"),
                        "high": entry.get("high"),
                        "low": entry.get("low"),
                        "close": entry.get("close"),
                        "volume": entry.get("volume"),
                    })

            if not records:
                print(f"No price data found for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df.dropna(inplace=True)

            if use_cache:
                print(f"Saving crypto data to {filename}...")
                df.to_csv(filename, index=False)

            return df

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return pd.DataFrame()
