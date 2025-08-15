# /workspaces/NeuroForge/allora-mdk/train.py
# Production-grade with exhaustive validation

from __future__ import annotations
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
import joblib
from data.csv_loader import CSVLoader
import numpy as np
import pandas as pd

# ---- Configuration (Immutable) ----------------------------------------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SCRIPTS = ROOT / "scripts"
DATA_DIR = ROOT / "data" / "sets"
ARTIFACTS_DIR = ROOT / "artifacts"

# Competition constraints
FORECAST_DAYS = 30  # Strict Allora requirement
MIN_HISTORY_YEARS = 3
CONFIDENCE_FLOOR = 0.85
ETH_DATA_FILE = "eth.csv"

# ---- Environment Validation -------------------------------------------------
def _validate_environment() -> None:
    """Pre-flight checks before execution."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not (DATA_DIR / ETH_DATA_FILE).exists():
        raise FileNotFoundError(
            f"ETH price data not found at {DATA_DIR/ETH_DATA_FILE}\n"
            f"Expected CSV with columns: [timestamp, close, ...]"
        )
    
    if not SCRIPTS.exists():
        raise RuntimeError(f"Project scripts directory missing: {SCRIPTS}")

# ---- Core Validation Functions ----------------------------------------------
def _validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Strict dataframe validation for financial time-series."""
    if df.empty:
        return False, "Empty dataframe"
    
    # Column checks (case-insensitive)
    cols = {col.lower(): col for col in df.columns}
    if 'close' not in cols:
        return False, "Missing 'close' price column"
    
    # Data quality checks
    close_series = df[cols['close']]
    if close_series.isnull().any():
        return False, "NaN values in price data"
    if (close_series <= 0).any():
        return False, "Non-positive prices detected"
    
    return True, ""

def _calculate_timespan(df: pd.DataFrame) -> float:
    """Calculate timespan in days with multiple fallback methods."""
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            delta = df.index.max() - df.index.min()
        elif 'timestamp' in df.columns:
            delta = pd.to_datetime(df['timestamp']).max() - pd.to_datetime(df['timestamp']).min()
        else:
            # Conservative estimate: assume daily data
            delta = timedelta(days=len(df))
        return delta.total_seconds() / 86400
    except Exception:
        warnings.warn("Timespan calculation fell back to row count")
        return float(len(df))

# ---- Confidence Calculation -------------------------------------------------
def _calculate_confidence(predictions: List[float]) -> float:
    """Robust confidence metric with volatility scaling."""
    if len(predictions) < 2:
        return CONFIDENCE_FLOOR
    
    prices = np.asarray(predictions, dtype=np.float64)
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # Annualized volatility (assuming daily data)
    volatility = np.std(log_returns) * np.sqrt(365)
    confidence = np.exp(-volatility)
    
    # Apply bounds and rounding
    return float(np.clip(round(confidence, 2), CONFIDENCE_FLOOR, 0.99))

# ---- Forecast Processing ----------------------------------------------------
def _normalize_forecast(
    forecast: Union[pd.DataFrame, pd.Series, np.ndarray, List[float]],
    days: int = FORECAST_DAYS
) -> List[float]:
    """Convert any forecast format to standardized list[float]."""
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

# ---- Main Training Pipeline -------------------------------------------------
def train_competition_model() -> Dict[str, Any]:
    """End-to-end training with production-grade validation."""
    _validate_environment()
    
    # Data Loading
    try:
        df = CSVLoader.load_csv("/workspaces/NeuroForge/allora-mdk/data/sets/eth.csv")
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {e}") from e
    
    # Data Validation
    is_valid, validation_msg = _validate_dataframe(df)
    if not is_valid:
        raise ValueError(f"Invalid data: {validation_msg}")
    
    # Preprocessing
    processed = preprocess_data(df)
    span_days = _calculate_timespan(processed)
    
    if span_days < MIN_HISTORY_YEARS * 365:
        warnings.warn(
            f"Insufficient history: {span_days:.1f} days "
            f"(< {MIN_HISTORY_YEARS} years)"
        )
    
    # Model Training
    factory = ModelFactory()
    model = factory.create_model("prophet")
    model.train(processed)
    
    # Forecasting
    raw_forecast = model.forecast(FORECAST_DAYS)
    forecast = _normalize_forecast(raw_forecast)
    
    if len(forecast) != FORECAST_DAYS:
        raise RuntimeError(
            f"Forecast length mismatch: expected {FORECAST_DAYS}, got {len(forecast)}"
        )
    
    # Artifact Packaging
    artifacts = {
        "model": model,
        "forecast": forecast,
        "confidence": _calculate_confidence(forecast),
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_range": {
                "start": processed.index.min().isoformat(),
                "end": processed.index.max().isoformat()
            },
            "metrics": {
                "training_rows": len(processed),
                "history_days": span_days,
                "forecast_days": len(forecast)
            }
        }
    }
    
    # Save with checksum verification
    artifact_path = ARTIFACTS_DIR / "competition_model.pkl"
    joblib.dump(artifacts, artifact_path)
    
    # Verify save was successful
    if not artifact_path.exists():
        raise RuntimeError("Artifact save failed - file not created")
    
    return artifacts

# ---- Entry Point ------------------------------------------------------------
def main() -> None:
    try:
        if "--competition" in sys.argv:
            results = train_competition_model()
            print_colored(
                "✓ Training successful\n"
                f"• Confidence: {results['confidence']:.2f}\n"
                f"• Forecast range: {results['forecast'][0]:.2f} to {results['forecast'][-1]:.2f}",
                "success"
            )
        else:
            print("[INFO] Run with --competition for Allora submission mode")
    except Exception as e:
        print(f"[ERROR] ✗ Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
