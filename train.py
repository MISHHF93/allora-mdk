#!/usr/bin/env python3
"""
Unified Model Training Framework — Production-Ready
- Competition Mode: Strict validation + reproducible pipeline for Allora MDK submissions
- Interactive Mode: Flexible experimentation with non-interactive option for CI

Notes:
- This is a drop-in, hardened replacement for the original script, preserving behavior while
  fixing correctness issues and tightening robustness.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd

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
DEFAULT_SEED = 42

# Import project-specific components
sys.path.append(str(SCRIPTS_DIR))
from configs import models  # type: ignore
from data.csv_loader import CSVLoader  # type: ignore
from data.tiingo_data_fetcher import DataFetcher  # type: ignore
from data.utils.data_preprocessing import preprocess_data  # type: ignore
from models.model_factory import ModelFactory  # type: ignore
from utils.common import print_colored  # type: ignore

# ---- Logging ----------------------------------------------------------------
logger = logging.getLogger("trainer")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)

# ---- Utilities --------------------------------------------------------------
def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _lower_cols_map(df: pd.DataFrame) -> Dict[str, str]:
    return {c.lower(): c for c in df.columns}


# ---- Core Validation Functions ----------------------------------------------
def validate_environment() -> None:
    """Ensure required directories and files exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory missing: {DATA_DIR}")

    if not SCRIPTS_DIR.exists():
        raise RuntimeError(f"Scripts directory missing: {SCRIPTS_DIR}")


def validate_dataframe(df: pd.DataFrame, competition_mode: bool = False) -> Tuple[bool, str]:
    """Robust dataframe validation.

    In competition mode, *both* 'timestamp' and 'close' are required, with strict
    timestamp checks. In interactive mode, 'close' is required; if 'timestamp' is
    present, it is validated as well.
    """
    if df is None or df.empty:
        return False, "Empty dataframe"

    cols = _lower_cols_map(df)

    if competition_mode:
        required = ["timestamp", "close"]
    else:
        required = ["close"]

    for name in required:
        if name not in cols:
            return False, f"Missing required column: '{name}'"

    # Price checks
    close = df[cols["close"]]
    if close.isnull().any():
        return False, "NaN values in price data"
    if (close <= 0).any():
        return False, "Non-positive prices detected"

    # Timestamp checks (strict in competition mode; opportunistic otherwise)
    if competition_mode or ("timestamp" in cols):
        ts = pd.to_datetime(df[cols["timestamp"]], errors="coerce")
        if ts.isnull().any():
            return False, "Unparseable timestamps detected"
        if not ts.is_monotonic_increasing:
            return False, "Timestamps are not monotonically increasing"
        if ts.duplicated().any():
            return False, "Duplicate timestamps detected"

    return True, ""


def calculate_timespan(df: pd.DataFrame) -> float:
    """Calculate timespan in days with multiple fallbacks (case-insensitive)."""
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            delta = df.index.max() - df.index.min()
        else:
            cols = _lower_cols_map(df)
            if "timestamp" in cols:
                ts = pd.to_datetime(df[cols["timestamp"]], errors="coerce").dropna()
                if len(ts) >= 2:
                    delta = ts.max() - ts.min()
                else:
                    delta = timedelta(days=len(df))
            else:
                delta = timedelta(days=len(df))
        return float(delta.total_seconds() / 86400.0)
    except Exception as e:  # pragma: no cover (safety net)
        warnings.warn(f"Timespan calculation fell back to row count: {e}")
        return float(len(df))


# ---- Confidence & Forecast Utilities ----------------------------------------
def calculate_confidence(predictions: List[float]) -> float:
    """Confidence inversely related to annualized volatility with safety nets."""
    if predictions is None or len(predictions) < 2:
        return CONFIDENCE_FLOOR

    prices = np.asarray(predictions, dtype=np.float64)
    prices = np.where(~np.isfinite(prices) | (prices <= 0), np.nan, prices)
    if np.isnan(prices).any():
        return CONFIDENCE_FLOOR

    log_returns = np.log(prices[1:] / prices[:-1])
    log_returns = log_returns[np.isfinite(log_returns)]
    if log_returns.size == 0:
        return CONFIDENCE_FLOOR

    volatility = float(np.std(log_returns) * np.sqrt(365.0))
    if not np.isfinite(volatility):
        return CONFIDENCE_FLOOR

    # Slightly softened curve to avoid saturating at 0.99 for flat series
    confidence = float(np.exp(-0.75 * volatility))
    return float(np.clip(round(confidence, 2), CONFIDENCE_FLOOR, 0.99))


def normalize_forecast(
    forecast: Union[pd.DataFrame, pd.Series, np.ndarray, List[float]],
    days: int = FORECAST_DAYS,
) -> List[float]:
    """Convert any forecast format to standardized list[float] with NaN handling and padding."""
    # Extract numeric array
    if isinstance(forecast, pd.DataFrame):
        num_cols = forecast.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            values = pd.to_numeric(forecast.iloc[:, 0], errors="coerce")
        else:
            values = forecast[num_cols[0]]
        arr = values.to_numpy(dtype=np.float64)
    elif isinstance(forecast, pd.Series):
        arr = pd.to_numeric(forecast, errors="coerce").to_numpy(dtype=np.float64)
    elif isinstance(forecast, (np.ndarray, list)):
        arr = np.asarray(forecast, dtype=np.float64)
    else:
        raise TypeError(f"Unsupported forecast type: {type(forecast)}")

    # Handle invalids: forward-fill; seed first value if needed
    if arr.size == 0 or np.isnan(arr).all():
        arr = np.full(shape=(days,), fill_value=1.0, dtype=np.float64)
    else:
        for i in range(arr.shape[0]):
            if i == 0 and (not np.isfinite(arr[i]) or arr[i] <= 0):
                arr[i] = 1.0
            elif (not np.isfinite(arr[i])) or (arr[i] <= 0):
                arr[i] = arr[i - 1]

    # Pad / truncate to desired horizon
    if arr.shape[0] < days:
        pad_val = arr[-1] if arr.shape[0] > 0 else 1.0
        arr = np.pad(arr, (0, days - arr.shape[0]), constant_values=pad_val)
    elif arr.shape[0] > days:
        arr = arr[:days]

    return [float(x) for x in arr]


# ---- Data Loading & Selection -----------------------------------------------
def load_competition_data() -> pd.DataFrame:
    """Load and validate ETH data for competition submission."""
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

    # Check history length (warning only)
    span_days = calculate_timespan(df)
    if span_days < MIN_HISTORY_YEARS * 365:
        warnings.warn(
            f"Insufficient history: {span_days:.1f} days (< {MIN_HISTORY_YEARS} years)"
        )

    return preprocess_data(df)


def select_data(
    fetcher: DataFetcher,
    competition_mode: bool = False,
    *,
    source: str | None = None,
    symbol: str | None = None,
    frequency: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    csv_path: str | None = None,
    non_interactive: bool = False,
) -> pd.DataFrame:
    """Data selection with competition mode override and non-interactive support."""
    if competition_mode:
        return load_competition_data()

    default_end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if non_interactive:
        # Resolve defaults for CI/headless runs
        src = (source or "csv").lower()
        if src == "stock":
            sym = symbol or "AAPL"
            freq = frequency or "daily"
            sd = start_date or "2021-01-01"
            ed = end_date or default_end_date
            return fetcher.fetch_tiingo_stock_data(sym, sd, ed, freq)
        elif src == "crypto":
            sym = symbol or "btcusd"
            freq = frequency or "1day"
            sd = start_date or "2021-01-01"
            ed = end_date or default_end_date
            return fetcher.fetch_tiingo_crypto_data(sym, sd, ed, freq)
        elif src == "csv":
            if not csv_path:
                raise ValueError("--source=csv requires --csv=<path> in non-interactive mode")
            return CSVLoader.load_csv(csv_path)
        else:
            raise ValueError(f"Unknown source: {source}")

    # Interactive prompts
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

    if selection == "2":
        symbol = input("Enter crypto symbol (default: btcusd): ").strip() or "btcusd"
        frequency = input("Frequency (1min/5min/1hour/1day, default: 1day): ").strip() or "1day"
        start_date = input("Start date (YYYY-MM-DD, default: 2021-01-01): ").strip() or "2021-01-01"
        end_date = input(f"End date (YYYY-MM-DD, default: {default_end_date}): ").strip() or default_end_date
        return fetcher.fetch_tiingo_crypto_data(symbol, start_date, end_date, frequency)

    if selection == "3":
        file_path = input("Enter CSV file path: ").strip()
        return CSVLoader.load_csv(file_path)

    print_colored("Invalid choice", "error")
    sys.exit(1)


# ---- Model Selection --------------------------------------------------------
def select_models(competition_mode: bool = False, *, non_interactive: bool = False, use_all: bool = True) -> List[str]:
    """Model selection with competition mode override and non-interactive support."""
    if competition_mode:
        return ["prophet"]  # Competition requires Prophet

    if non_interactive:
        return models if use_all else models[:1]

    print("Select models to train:")
    print("1. All models")
    print("2. Custom selection")
    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        return models
    if choice == "2":
        print("Available models:")
        for i, model_name in enumerate(models, 1):
            print(f"{i}. {model_name}")
        selections = input("Enter model numbers (comma separated): ").split(",")
        return [models[int(sel.strip()) - 1] for sel in selections if sel.strip().isdigit()]

    print_colored("Invalid choice, using all models", "warning")
    return models


# ---- Competition Artifact Packaging -----------------------------------------
def _safe_range_from_data(data: pd.DataFrame) -> Tuple[str | None, str | None]:
    if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 0:
        return data.index.min().isoformat(), data.index.max().isoformat()
    cols = _lower_cols_map(data)
    if "timestamp" in cols:
        ts = pd.to_datetime(data[cols["timestamp"]], errors="coerce").dropna()
        if len(ts) > 0:
            return ts.min().isoformat(), ts.max().isoformat()
    return None, None


def package_competition_artifacts(
    model: Any,
    forecast: List[float],
    data: pd.DataFrame,
) -> Dict[str, Any]:
    """Create standardized artifact package for competition submission."""
    span_days = calculate_timespan(data)
    confidence = calculate_confidence(forecast)
    start_iso, end_iso = _safe_range_from_data(data)

    return {
        "model": model,
        "forecast": forecast,
        "confidence": confidence,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_source": ETH_DATA_FILE,
            "data_range": {"start": start_iso, "end": end_iso},
            "metrics": {
                "training_rows": int(len(data)),
                "history_days": float(span_days),
                "forecast_days": int(len(forecast)),
            },
        },
    }


def save_artifacts(results: Dict[str, Any], competition_mode: bool = False) -> Path:
    """Save artifacts with mode-appropriate format and shape parity."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if competition_mode:
        artifact_path = ARTIFACTS_DIR / f"competition_submission_{timestamp}.pkl"
        joblib.dump(results, artifact_path)
        if not artifact_path.exists():
            raise RuntimeError("Artifact save failed - file not created")
        return artifact_path

    # Interactive mode: save each model separately, including confidence
    for model_name, model_data in results.items():
        model_path = ARTIFACTS_DIR / f"{model_name}_{timestamp}.pkl"
        joblib.dump(
            {
                "model": model_data["model"],
                "forecast": model_data["forecast"],
                "confidence": calculate_confidence(model_data["forecast"]),
                "metadata": {"trained_at": timestamp, "model_type": model_name},
            },
            model_path,
        )
    return ARTIFACTS_DIR


# ---- Main Training Pipelines ------------------------------------------------
def run_competition_mode() -> Dict[str, Any]:
    """End-to-end competition submission pipeline."""
    set_global_seed()
    validate_environment()

    # Load and validate data
    data = load_competition_data()

    # Train model
    factory = ModelFactory()
    model = factory.create_model("prophet")
    model.train(data)

    # Generate forecast (normalized & padded to exact horizon)
    raw_forecast = model.forecast(FORECAST_DAYS)
    forecast = normalize_forecast(raw_forecast, days=FORECAST_DAYS)

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
        "success",
    )
    return artifacts


def run_interactive_mode(
    *,
    non_interactive: bool = False,
    source: str | None = None,
    symbol: str | None = None,
    frequency: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    csv: str | None = None,
    horizon: int = FORECAST_DAYS,
    use_all_models: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Interactive/CI training pipeline."""
    set_global_seed()
    validate_environment()
    fetcher = DataFetcher()
    results: Dict[str, Dict[str, Any]] = {}

    # Data selection and validation
    data = select_data(
        fetcher,
        competition_mode=False,
        source=source,
        symbol=symbol,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
        csv_path=csv,
        non_interactive=non_interactive,
    )

    is_valid, msg = validate_dataframe(data)
    if not is_valid:
        raise ValueError(f"Invalid data: {msg}")

    data = preprocess_data(data)
    span_days = calculate_timespan(data)
    print_colored(f"✓ Loaded data with {len(data)} rows ({span_days:.1f} days)", "info")

    # Model selection
    model_types = select_models(non_interactive=non_interactive, use_all=use_all_models)
    factory = ModelFactory()

    # Forecast horizon
    forecast_days = int(horizon)

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
                "forecast_days": forecast_days,
            }

            print_colored(
                f"✓ {model_type} forecast: {forecast[0]:.2f} → {forecast[-1]:.2f} "
                f"({len(forecast)} days)",
                "success",
            )
        except Exception as e:
            print_colored(f"✗ {model_type} training failed: {e}", "error")

    # Save artifacts
    save_path = save_artifacts(results)
    print_colored(f"\n✓ Models saved to: {save_path}", "success")
    return results


# ---- CLI --------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Model Training Framework")
    p.add_argument("--competition", action="store_true", help="Run Allora MDK competition pipeline")
    p.add_argument("--benchmark", action="store_true", help="(Reserved) Run benchmarking suite")

    # Non-interactive / CI flags
    p.add_argument("--non-interactive", action="store_true", help="Disable prompts; use flags")
    p.add_argument("--horizon", type=int, default=FORECAST_DAYS, help="Forecast horizon in days")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global RNG seed")

    # Data selection flags
    p.add_argument("--source", choices=["stock", "crypto", "csv"], help="Data source for non-interactive mode")
    p.add_argument("--symbol", help="Ticker/symbol for Tiingo sources")
    p.add_argument("--frequency", help="Data frequency (e.g., daily, 1day)")
    p.add_argument("--start-date", dest="start_date", help="Start date YYYY-MM-DD")
    p.add_argument("--end-date", dest="end_date", help="End date YYYY-MM-DD")
    p.add_argument("--csv", dest="csv_path", help="Path to CSV when --source=csv")

    # Model selection flags
    p.add_argument("--all-models", dest="all_models", action="store_true", help="Train all models")
    p.add_argument("--first-model-only", dest="first_model_only", action="store_true", help="Train only the first model in configs.models")

    return p.parse_args(argv)


# ---- Main Entry Point -------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed)

    try:
        if args.competition:
            print_colored("=== ALLORA MDK COMPETITION MODE ===", "header")
            _ = run_competition_mode()
        elif args.benchmark:
            print_colored("=== BENCHMARK MODE ===", "header")
            # (Reserved) Implement benchmarking pipeline as needed.
            print_colored("Benchmark mode is not yet implemented.", "warning")
        else:
            print_colored("=== INTERACTIVE TRAINING MODE ===", "header")
            _ = run_interactive_mode(
                non_interactive=args.non_interactive,
                source=args.source,
                symbol=args.symbol,
                frequency=args.frequency,
                start_date=args.start_date,
                end_date=args.end_date,
                csv=args.csv_path,
                horizon=args.horizon,
                use_all_models=False if args.first_model_only else True if args.all_models else True,
            )

    except Exception as e:
        print_colored(f"\n✗ Critical error: {e}", "error")
        sys.exit(1)

    print_colored("\n✓ Operation completed successfully", "success")


if __name__ == "__main__":
    main()
