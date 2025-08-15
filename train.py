#!/usr/bin/env python3
"""
Unified Model Training Framework
- Competition Mode: Production-grade validation for Allora MDK submissions
- Interactive Mode: Flexible experimentation with various models/data sources
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

# ---- Configuration ----------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data" / "sets"
ARTIFACTS_DIR = ROOT / "artifacts"
SCRIPTS_DIR = ROOT / "scripts"

# Competition constraints
FORECAST_DAYS = 30
MIN_HISTORY_YEARS = 3
CONFIDENCE_FLOOR = 0.85
ETH_DATA_FILE = "eth.csv"

# Import project-specific components
sys.path.append(str(SCRIPTS_DIR))
from configs import models
from data.csv_loader import CSVLoader
from data.tiingo_data_fetcher import DataFetcher
from data.utils.data_preprocessing import preprocess_data
from models.model_factory import ModelFactory
from utils.common import print_colored

# ---- Core Validation Functions ----------------------------------------------
def validate_environment() -> None:
    """Ensure required directories and files exist"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory missing: {DATA_DIR}")
    
    if not SCRIPTS_DIR.exists():
        raise RuntimeError(f"Scripts directory missing: {SCRIPTS_DIR}")

def validate_dataframe(df: pd.DataFrame, competition_mode: bool = False) -> Tuple[bool, str]:
    """Robust dataframe validation with competition-level strictness"""
    if df.empty:
        return False, "Empty dataframe"
    
    # Column checks (case-insensitive)
    cols = {col.lower(): col for col in df.columns}
    required_columns = ['close'] if competition_mode else ['close', 'timestamp']
    
    for col in required_columns:
        if col not in cols:
            return False, f"Missing required column: '{col}'"
    
    # Data quality checks
    close_series = df[cols['close']]
    if close_series.isnull().any():
        return False, "NaN values in price data"
    if (close_series <= 0).any():
        return False, "Non-positive prices detected"
    
    # Competition-specific validations
    if competition_mode:
        if 'timestamp' in cols:
            timestamps = pd.to_datetime(df[cols['timestamp']])
            if not timestamps.is_monotonic_increasing:
                return False, "Timestamps are not monotonically increasing"
            if timestamps.duplicated().any():
                return False, "Duplicate timestamps detected"
    
    return True, ""

def calculate_timespan(df: pd.DataFrame) -> float:
    """Calculate timespan in days with multiple fallback methods"""
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            delta = df.index.max() - df.index.min()
        elif 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            delta = timestamps.max() - timestamps.min()
        else:
            # Conservative estimate: assume daily data
            delta = timedelta(days=len(df))
        return delta.total_seconds() / 86400
    except Exception as e:
        warnings.warn(f"Timespan calculation fell back to row count: {str(e)}")
        return float(len(df))

# ---- Confidence & Forecast Utilities ----------------------------------------
def calculate_confidence(predictions: List[float]) -> float:
    """Robust confidence metric with volatility scaling"""
    if len(predictions) < 2:
        return CONFIDENCE_FLOOR
    
    prices = np.asarray(predictions, dtype=np.float64)
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # Annualized volatility (assuming daily data)
    volatility = np.std(log_returns) * np.sqrt(365)
    confidence = np.exp(-volatility)
    
    # Apply bounds and rounding
    return float(np.clip(round(confidence, 2), CONFIDENCE_FLOOR, 0.99))

def normalize_forecast(
    forecast: Union[pd.DataFrame, pd.Series, np.ndarray, List[float]],
    days: int = FORECAST_DAYS
) -> List[float]:
    """Convert any forecast format to standardized list[float]"""
    if isinstance(forecast, pd.DataFrame):
        # Select first numeric column
        num_cols = forecast.select_dtypes(include=[np.number]).columns
        values = forecast[num_cols[0]] if len(num_cols) > 0 else forecast.iloc[:, 0]
        arr = values.to_numpy(dtype=np.float64)
    elif isinstance(forecast, pd.Series):
        arr = forecast.to_numpy(dtype=np.float64)
    elif isinstance(forecast, (np.ndarray, list)):
        arr = np.asarray(forecast, dtype=np.float64)
    else:
        raise TypeError(f"Unsupported forecast type: {type(forecast)}")
    
    # Clip negative values and ensure length
    arr = np.clip(arr, 1e-8, None)
    return [float(x) for x in arr[:days]]

# ---- Data Loading & Selection -----------------------------------------------
def load_competition_data() -> pd.DataFrame:
    """Load and validate ETH data for competition submission"""
    eth_path = DATA_DIR / ETH_DATA_FILE
    if not eth_path.exists():
        raise FileNotFoundError(
            f"ETH price data not found at {eth_path}\n"
            f"Expected CSV with columns: [timestamp, close, ...]"
        )
    
    df = CSVLoader.load_csv(eth_path)
    is_valid, validation_msg = validate_dataframe(df, competition_mode=True)
    
    if not is_valid:
        raise ValueError(f"Invalid ETH data: {validation_msg}")
    
    # Check history length
    span_days = calculate_timespan(df)
    if span_days < MIN_HISTORY_YEARS * 365:
        warnings.warn(
            f"Insufficient history: {span_days:.1f} days "
            f"(< {MIN_HISTORY_YEARS} years)"
        )
    
    return preprocess_data(df)

def select_data(fetcher: DataFetcher, competition_mode: bool = False) -> pd.DataFrame:
    """Data selection with competition mode override"""
    if competition_mode:
        return load_competition_data()
    
    # Interactive data selection
    default_end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print("Select the data source:")
    print("1. Tiingo Stock Data")
    print("2. Tiingo Crypto Data")
    print("3. Load data from CSV file")
    selection = input("Enter your choice (1/2/3): ").strip()

    if selection == "1":
        symbol = input("Enter the stock symbol (default: AAPL): ").strip() or "AAPL"
        frequency = input("Frequency (daily/weekly/monthly, default: daily): ").strip() or "daily"
        start_date = input("Start date (YYYY-MM-DD, default: 2021-01-01): ").strip() or "2021-01-01"
        end_date = input(f"End date (YYYY-MM-DD, default: {default_end_date}): ").strip() or default_end_date
        return fetcher.fetch_tiingo_stock_data(symbol, start_date, end_date, frequency)

    elif selection == "2":
        symbol = input("Enter crypto symbol (default: btcusd): ").strip() or "btcusd"
        frequency = input("Frequency (1min/5min/1hour/1day, default: 1day): ").strip() or "1day"
        start_date = input("Start date (YYYY-MM-DD, default: 2021-01-01): ").strip() or "2021-01-01"
        end_date = input(f"End date (YYYY-MM-DD, default: {default_end_date}): ").strip() or default_end_date
        return fetcher.fetch_tiingo_crypto_data(symbol, start_date, end_date, frequency)

    elif selection == "3":
        file_path = input("Enter CSV file path: ").strip()
        return CSVLoader.load_csv(file_path)

    else:
        print_colored("Invalid choice", "error")
        sys.exit(1)

# ---- Model Selection --------------------------------------------------------
def select_models(competition_mode: bool = False) -> List[str]:
    """Model selection with competition mode override"""
    if competition_mode:
        return ["prophet"]  # Competition requires Prophet
    
    print("Select models to train:")
    print("1. All models")
    print("2. Custom selection")
    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        return models
    elif choice == "2":
        print("Available models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        selections = input("Enter model numbers (comma separated): ").split(",")
        return [models[int(sel.strip())-1] for sel in selections if sel.strip().isdigit()]
    else:
        print_colored("Invalid choice, using all models", "warning")
        return models

# ---- Competition Artifact Packaging -----------------------------------------
def package_competition_artifacts(
    model: Any, 
    forecast: List[float], 
    data: pd.DataFrame
) -> Dict[str, Any]:
    """Create standardized artifact package for competition submission"""
    span_days = calculate_timespan(data)
    confidence = calculate_confidence(forecast)
    
    return {
        "model": model,
        "forecast": forecast,
        "confidence": confidence,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "eth.csv",
            "data_range": {
                "start": data.index.min().isoformat(),
                "end": data.index.max().isoformat()
            },
            "metrics": {
                "training_rows": len(data),
                "history_days": span_days,
                "forecast_days": len(forecast)
            }
        }
    }

def save_artifacts(
    results: Dict[str, Any], 
    competition_mode: bool = False
) -> Path:
    """Save artifacts with mode-appropriate format"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    if competition_mode:
        artifact_path = ARTIFACTS_DIR / f"competition_submission_{timestamp}.pkl"
        joblib.dump(results, artifact_path)
        
        # Verify save was successful
        if not artifact_path.exists():
            raise RuntimeError("Artifact save failed - file not created")
            
        return artifact_path
    
    # Interactive mode - save each model separately
    for model_name, model_data in results.items():
        model_path = ARTIFACTS_DIR / f"{model_name}_{timestamp}.pkl"
        joblib.dump({
            "model": model_data["model"],
            "forecast": model_data["forecast"],
            "metadata": {
                "trained_at": timestamp,
                "model_type": model_name
            }
        }, model_path)
    
    return ARTIFACTS_DIR

# ---- Main Training Pipelines ------------------------------------------------
def run_competition_mode() -> Dict[str, Any]:
    """End-to-end competition submission pipeline"""
    validate_environment()
    
    # Load and validate data
    data = load_competition_data()
    
    # Train model
    factory = ModelFactory()
    model = factory.create_model("prophet")
    model.train(data)
    
    # Generate forecast
    raw_forecast = model.forecast(FORECAST_DAYS)
    forecast = normalize_forecast(raw_forecast)
    
    if len(forecast) != FORECAST_DAYS:
        raise RuntimeError(
            f"Forecast length mismatch: expected {FORECAST_DAYS}, got {len(forecast)}"
        )
    
    # Package and save artifacts
    artifacts = package_competition_artifacts(model, forecast, data)
    save_path = save_artifacts(artifacts, competition_mode=True)
    
    print_colored(
        f"✓ Competition submission package saved to: {save_path}\n"
        f"• Confidence: {artifacts['confidence']:.2f}\n"
        f"• Forecast range: {forecast[0]:.2f} to {forecast[-1]:.2f}",
        "success"
    )
    return artifacts

def run_interactive_mode() -> Dict[str, Dict[str, Any]]:
    """Interactive training pipeline"""
    validate_environment()
    fetcher = DataFetcher()
    results = {}
    
    # Data selection and validation
    data = select_data(fetcher)
    is_valid, msg = validate_dataframe(data)
    if not is_valid:
        raise ValueError(f"Invalid data: {msg}")
    
    data = preprocess_data(data)
    span_days = calculate_timespan(data)
    print_colored(f"✓ Loaded data with {len(data)} rows ({span_days:.1f} days)", "info")
    
    # Model selection
    model_types = select_models()
    factory = ModelFactory()
    
    # Forecast horizon
    forecast_days = int(input("Enter forecast horizon (days): ") or 30)
    
    # Train models and generate forecasts
    for model_type in model_types:
        print_colored(f"\n=== Training {model_type} model ===", "header")
        model = factory.create_model(model_type)
        
        try:
            model.train(data)
            raw_forecast = model.forecast(forecast_days)
            forecast = normalize_forecast(raw_forecast, days=forecast_days)
            
            results[model_type] = {
                "model": model,
                "forecast": forecast,
                "forecast_days": forecast_days
            }
            
            print_colored(
                f"✓ {model_type} forecast: {forecast[0]:.2f} → {forecast[-1]:.2f} "
                f"({len(forecast)} days)",
                "success"
            )
        except Exception as e:
            print_colored(f"✗ {model_type} training failed: {str(e)}", "error")
    
    # Save artifacts
    save_path = save_artifacts(results)
    print_colored(f"\n✓ Models saved to: {save_path}", "success")
    return results

# ---- Main Entry Point -------------------------------------------------------
def main():
    """Entry point with mode detection"""
    competition_mode = "--competition" in sys.argv
    benchmark_mode = "--benchmark" in sys.argv
    
    try:
        if competition_mode:
            print_colored("=== ALLORA MDK COMPETITION MODE ===", "header")
            results = run_competition_mode()
        elif benchmark_mode:
            print_colored("=== BENCHMARK MODE ===", "header")
            # Implement model benchmarking here
        else:
            print_colored("=== INTERACTIVE TRAINING MODE ===", "header")
            results = run_interactive_mode()
            
    except Exception as e:
        print_colored(f"\n✗ Critical error: {str(e)}", "error")
        sys.exit(1)
        
    print_colored("\n✓ Operation completed successfully", "success")

if __name__ == "__main__":
    main()
