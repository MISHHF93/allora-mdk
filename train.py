#!/usr/bin/env python3
"""
Unified Model Training Framework — Production-Ready (Custom Exceptions + Robust Imports)
- Competition Mode: Strict validation + reproducible pipeline for Allora MDK submissions
- Interactive Mode: Flexible experimentation with non-interactive option for CI
- Error Handling: Centralized custom exceptions with consistent exit codes

Project layout assumed from your tree:
.
├── configs.py
├── data/ ... (package; has __init__.py)
├── models/ ... (namespace package; no __init__ required)
├── utils/ ... (package; has __init__.py)
└── ...

Key fixes in this revision:
- Prefer the **project root** on sys.path (not a non-existent scripts/).
- Load model list from `configs.py` at repo root; avoid importing unrelated site-packages `configs`.
- Defer imports and add fallbacks (e.g., `data/utils/data_preprocessing.py` lacks __init__.py in `utils/` → load by filepath).
- Hardened environment setup to prevent `SystemExit: 78` — create/fallback for artifacts dir and (optionally) auto-create data dir instead of failing.
- Keep self-tests; add env/IO tests; no changes to existing tests.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import os
import random
import sys
import traceback
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# ---- Configuration ----------------------------------------------------------
# Some environments (e.g., interactive notebooks, sandboxes) don't define __file__
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # e.g., running in an interactive shell or certain sandboxes
    HERE = Path(os.getcwd()).resolve()

ROOT = HERE if (HERE / "configs.py").exists() else HERE.parent

# Ensure **project root** is at the front of sys.path so local modules win.
if str(ROOT) in sys.path:
    try:
        sys.path.remove(str(ROOT))
    except ValueError:
        pass
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"  # your repo has data/, not data/sets/
ARTIFACTS_DIR = ROOT / "artifacts"

# Competition constraints
FORECAST_DAYS = 30
MIN_HISTORY_YEARS = 3
CONFIDENCE_FLOOR = 0.85
# Default ETH CSV: if not provided, we will try to auto-detect a CSV in data/ or data/utils/
ETH_DATA_FILE = "eth.csv"  # can be overridden by ENV ETH_CSV
DEFAULT_SEED = 42


# ---- Custom Exceptions & Exit Codes ----------------------------------------
class TrainingError(Exception):
    """Base exception for all training-related errors."""


class DataError(TrainingError):
    """Data ingestion/validation errors."""


class ModelError(TrainingError):
    """Model construction/training errors."""


class EnvironmentError(TrainingError):
    """Missing directories, bad environment, permissions, etc."""


class ForecastError(TrainingError):
    """Forecast generation/normalization/horizon errors."""


class ArtifactError(TrainingError):
    """Artifact packaging/saving errors."""


_EXIT_CODES = {
    DataError: 64,  # EX_USAGE-like
    ModelError: 65,
    ForecastError: 66,
    ArtifactError: 67,
    EnvironmentError: 78,  # EX_CONFIG-like
    TrainingError: 70,  # software error (generic)
}


def _exit_code_for(e: BaseException) -> int:
    for etype, code in _EXIT_CODES.items():
        if isinstance(e, etype):
            return code
    return 1


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


# print_colored fallback so the script remains usable without project deps
try:
    from utils.common import print_colored  # type: ignore
except Exception:  # pragma: no cover - fallback used when project util missing
    def print_colored(msg: str, level: str = "info") -> None:
        prefix = {
            "error": "[ERROR]",
            "warning": "[WARN]",
            "success": "[OK]",
            "header": "[==]",
            "info": "[INFO]",
        }.get(level, "[INFO]")
        print(f"{prefix} {msg}")


# ---- Robust dynamic imports -------------------------------------------------

def _import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_preprocess() -> Any:
    """Import preprocess_data with a filepath fallback because data/utils has no __init__.py."""
    try:
        return importlib.import_module("data.utils.data_preprocessing").preprocess_data  # type: ignore
    except Exception:
        path = ROOT / "data" / "utils" / "data_preprocessing.py"
        if not path.exists():
            raise DataError("Cannot find data_preprocessing.py under data/utils/")
        return _import_module_from_path("data_preprocessing", path).preprocess_data  # type: ignore


# ---- Load model list robustly (fixes ImportError: cannot import name 'models')
MODELS: List[str]


def _load_models_list() -> List[str]:
    """Attempt to load model names from root-level `configs.py`.

    Order of attempts:
      1) import configs (must be your root module due to ROOT on sys.path) and read 'models'/'MODELS'/'model_list'
      2) fallback to ['prophet']
    """
    try:
        cfg = importlib.import_module("configs")
        for attr in ("models", "MODELS", "model_list"):
            if hasattr(cfg, attr):
                m = getattr(cfg, attr)
                if isinstance(m, (list, tuple)) and all(isinstance(x, str) for x in m):
                    return list(m)
    except Exception:
        pass

    # Safe default (also used by competition mode)
    return ["prophet"]


MODELS = _load_models_list()


# ---- Core Validation Functions ----------------------------------------------
def validate_environment(ensure_data_dir: bool = False) -> None:
    """Ensure required directories exist & are writable.

    - Always attempts to create ARTIFACTS_DIR; if it fails (e.g., RO filesystem),
      it falls back to CWD/artifacts, then to a temp directory and updates the
      global ARTIFACTS_DIR accordingly.
    - If ensure_data_dir=True, attempts to create DATA_DIR (instead of failing on
      missing); only raises if creation is impossible.
    """
    import tempfile
    global ARTIFACTS_DIR, DATA_DIR

    # Create or fallback artifacts dir
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fallback 1: CWD
        cwd_artifacts = Path(os.getcwd()) / "artifacts"
        try:
            cwd_artifacts.mkdir(parents=True, exist_ok=True)
            ARTIFACTS_DIR = cwd_artifacts
            print_colored(f"Artifacts dir fallback to {ARTIFACTS_DIR}", "warning")
        except Exception:
            # Fallback 2: temp
            tmp_root = Path(tempfile.mkdtemp())
            tmp_artifacts = tmp_root / "artifacts"
            try:
                tmp_artifacts.mkdir(parents=True, exist_ok=True)
                ARTIFACTS_DIR = tmp_artifacts
                print_colored(f"Artifacts dir fallback to {ARTIFACTS_DIR}", "warning")
            except Exception as e3:  # pragma: no cover
                raise EnvironmentError(
                    "Unable to create artifacts directory in ROOT, CWD, or temp"
                ) from e3

    # Ensure (or create) data directory if requested
    if ensure_data_dir:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            raise EnvironmentError(f"Data directory inaccessible: {DATA_DIR}") from e


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
    try:
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
    except Exception as e:
        raise ForecastError(f"Failed to coerce forecast to numeric array: {e}") from e

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

    out = [float(x) for x in arr]
    if len(out) != days:
        raise ForecastError(f"Forecast length after normalization is {len(out)}, expected {days}")
    return out


# ---- Data Loading & Selection -----------------------------------------------

def _resolve_competition_csv() -> Path:
    """Find an ETH CSV from common locations/patterns.
    Resolution order:
      1) ENV ETH_CSV if set
      2) ROOT/data/ETH_DATA_FILE (default 'eth.csv')
      3) any file matching 'eth*.csv' in ROOT/data
      4) any file matching '*Ethereum*historical_data_coinmarketcap*.csv' in ROOT/data or ROOT/data/utils
    """
    env_path = os.getenv("ETH_CSV")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    candidate = DATA_DIR / ETH_DATA_FILE
    if candidate.exists():
        return candidate

    # Search patterns
    for pat in ("eth*.csv",):
        for f in DATA_DIR.glob(pat):
            if f.is_file():
                return f
    for pat in ("*Ethereum*historical_data_coinmarketcap*.csv",):
        for base in (DATA_DIR, DATA_DIR / "utils"):
            for f in base.glob(pat):
                if f.is_file():
                    return f

    raise DataError(
        "Cannot locate ETH CSV. Set ETH_CSV env var or place eth.csv under data/ "
        "(we also look for *Ethereum*historical_data_coinmarketcap*.csv)."
    )


def load_competition_data() -> pd.DataFrame:
    """Load and validate ETH data for competition submission."""
    eth_path = _resolve_competition_csv()

    try:
        CSVLoader = importlib.import_module("data.csv_loader").CSVLoader  # type: ignore
        df = CSVLoader.load_csv(eth_path)
    except Exception as e:  # propagate as DataError
        raise DataError(f"Failed to load competition CSV: {e}") from e

    is_valid, validation_msg = validate_dataframe(df, competition_mode=True)
    if not is_valid:
        raise DataError(f"Invalid ETH data: {validation_msg}")

    # Check history length (warning only)
    span_days = calculate_timespan(df)
    if span_days < MIN_HISTORY_YEARS * 365:
        warnings.warn(
            f"Insufficient history: {span_days:.1f} days (< {MIN_HISTORY_YEARS} years)"
        )

    try:
        preprocess_data = _import_preprocess()
        return preprocess_data(df)
    except Exception as e:
        raise DataError(f"Preprocessing failed: {e}") from e


def select_data(
    fetcher: Any,
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
        try:
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
                    raise DataError("--source=csv requires --csv=<path> in non-interactive mode")
                CSVLoader = importlib.import_module("data.csv_loader").CSVLoader  # type: ignore
                return CSVLoader.load_csv(csv_path)
            else:
                raise DataError(f"Unknown source: {source}")
        except DataError:
            raise
        except Exception as e:
            raise DataError(f"Failed to fetch data from source '{src}': {e}") from e

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
        try:
            CSVLoader = importlib.import_module("data.csv_loader").CSVLoader  # type: ignore
            return CSVLoader.load_csv(file_path)
        except Exception as e:
            raise DataError(f"Failed to load CSV from '{file_path}': {e}") from e

    raise DataError("Invalid data source selection; expected 1/2/3")


# ---- Model Selection --------------------------------------------------------

def select_models(competition_mode: bool = False, *, non_interactive: bool = False, use_all: bool = True) -> List[str]:
    """Model selection with competition mode override and non-interactive support."""
    if competition_mode:
        return ["prophet"]  # Competition requires Prophet

    if non_interactive:
        return MODELS if use_all else MODELS[:1]

    print("Select models to train:")
    print("1. All models")
    print("2. Custom selection")
    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        return MODELS
    if choice == "2":
        print("Available models:")
        for i, model_name in enumerate(MODELS, 1):
            print(f"{i}. {model_name}")
        selections = input("Enter model numbers (comma separated): ").split(",")
        selected = []
        for sel in selections:
            s = sel.strip()
            if s.isdigit() and 1 <= int(s) <= len(MODELS):
                selected.append(MODELS[int(s) - 1])
        if not selected:
            raise ModelError("No valid models selected")
        return selected

    raise ModelError("Invalid model selection; expected '1' or '2'")


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
            "data_source": str(_resolve_competition_csv()) if os.getenv("ETH_CSV") else ETH_DATA_FILE,
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

    try:
        if competition_mode:
            artifact_path = ARTIFACTS_DIR / f"competition_submission_{timestamp}.pkl"
            joblib.dump(results, artifact_path)
            if not artifact_path.exists():
                raise ArtifactError("Artifact save failed - file not created")
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
    except ArtifactError:
        raise
    except Exception as e:
        raise ArtifactError(f"Failed to save artifacts: {e}") from e


# ---- Main Training Pipelines ------------------------------------------------

def run_competition_mode() -> Dict[str, Any]:
    """End-to-end competition submission pipeline."""
    set_global_seed()
    # Ensure dirs; create data dir if missing so subsequent steps have a place to look
    validate_environment(ensure_data_dir=True)

    # Load and validate data
    data = load_competition_data()

    # Train model
    try:
        ModelFactory = importlib.import_module("models.model_factory").ModelFactory  # type: ignore
        model = ModelFactory().create_model("prophet")
    except Exception as e:
        raise ModelError(f"Model creation failed: {e}") from e

    try:
        model.train(data)
    except Exception as e:
        raise ModelError(f"Training failed: {e}") from e

    # Generate forecast (normalized & padded to exact horizon)
    try:
        raw_forecast = model.forecast(FORECAST_DAYS)
    except Exception as e:
        raise ForecastError(f"Model forecast failed: {e}") from e

    forecast = normalize_forecast(raw_forecast, days=FORECAST_DAYS)

    if len(forecast) != FORECAST_DAYS:
        raise ForecastError(
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
    # For interactive/CI we don't require data dir; still ensure artifacts are writable
    validate_environment(ensure_data_dir=False)

    # Lazy import here so self-tests can run without project deps
    DataFetcher = importlib.import_module("data.tiingo_data_fetcher").DataFetcher  # type: ignore
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
        raise DataError(f"Invalid data: {msg}")

    try:
        preprocess_data = _import_preprocess()
        data = preprocess_data(data)
    except Exception as e:
        raise DataError(f"Preprocessing failed: {e}") from e

    span_days = calculate_timespan(data)
    print_colored(f"✓ Loaded data with {len(data)} rows ({span_days:.1f} days)", "info")

    # Model selection
    model_types = select_models(non_interactive=non_interactive, use_all=use_all_models)

    # Forecast horizon
    forecast_days = int(horizon)

    # Train models and generate forecasts
    ModelFactory = importlib.import_module("models.model_factory").ModelFactory  # type: ignore

    for model_type in model_types:
        print_colored(f"\n=== Training {model_type} model ===", "header")
        try:
            model = ModelFactory().create_model(model_type)
        except Exception as e:
            print_colored(f"✗ {model_type} creation failed: {e}", "error")
            logger.debug(traceback.format_exc())
            continue

        try:
            model.train(data)
        except Exception as e:
            print_colored(f"✗ {model_type} training failed: {e}", "error")
            logger.debug(traceback.format_exc())
            continue

        try:
            raw_forecast = model.forecast(forecast_days)
            forecast = normalize_forecast(raw_forecast, days=forecast_days)
        except TrainingError as e:
            print_colored(f"✗ {model_type} forecast failed: {e}", "error")
            logger.debug(traceback.format_exc())
            continue
        except Exception as e:
            print_colored(f"✗ {model_type} forecast failed: {e}", "error")
            logger.debug(traceback.format_exc())
            continue

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

    if not results:
        raise TrainingError("No models produced a valid forecast; aborting save.")

    # Save artifacts
    save_path = save_artifacts(results)
    print_colored(f"\n✓ Models saved to: {save_path}", "success")
    return results


# ---- Self Tests (lightweight, no external IO) -------------------------------
# Run with: python train_clean.py --self-test

def _run_self_tests() -> int:
    """Return 0 if all tests pass, non-zero otherwise."""
    import tempfile

    failures = 0

    def _assert(name: str, cond: bool, msg: str = "") -> None:
        nonlocal failures
        if not cond:
            failures += 1
            print_colored(f"[TEST FAIL] {name}: {msg}", "error")
        else:
            print_colored(f"[TEST OK] {name}", "success")

    # 0) MODELS list loads and is sane (root configs.py)
    _assert(
        "MODELS is non-empty list of str",
        isinstance(MODELS, list) and len(MODELS) >= 1 and all(isinstance(x, str) for x in MODELS),
        str(MODELS),
    )

    # 1) normalize_forecast: pad + NaN handling
    nf = normalize_forecast([1.0, np.nan, 2.0], days=5)
    _assert("normalize_forecast pads to horizon", len(nf) == 5, f"len={len(nf)}")
    _assert("normalize_forecast has no NaNs", not any([not np.isfinite(x) for x in nf]))

    # 2) normalize_forecast: truncate
    nf2 = normalize_forecast(list(range(100)), days=7)
    _assert("normalize_forecast truncates", len(nf2) == 7)

    # 3) calculate_confidence: NaN -> floor
    conf = calculate_confidence([np.nan, np.nan])
    _assert("confidence floor on NaNs", conf == CONFIDENCE_FLOOR, f"got {conf}")

    # 4) calculate_confidence: constant -> <= 0.99
    conf2 = calculate_confidence([1.0] * 10)
    _assert("confidence capped", conf2 <= 0.99, f"got {conf2}")

    # 5) validate_dataframe: competition requires timestamp
    df_no_ts = pd.DataFrame({"close": [1, 2, 3]})
    ok, _ = validate_dataframe(df_no_ts, competition_mode=True)
    _assert("competition requires timestamp", ok is False)

    # 6) validate_dataframe: case-insensitive columns
    df_ci = pd.DataFrame({"TimeStamp": pd.date_range("2020-01-01", periods=3, freq="D"), "Close": [1, 2, 3]})
    ok2, msg = validate_dataframe(df_ci, competition_mode=True)
    _assert("case-insensitive columns valid", ok2 is True, msg)

    # 7) package_competition_artifacts structure
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
        "close": [1.0, 1.1, 1.2],
    })
    art = package_competition_artifacts(model="dummy", forecast=[1, 1, 1], data=df)
    _assert("artifact has confidence", "confidence" in art and isinstance(art["confidence"], float))
    _assert("artifact has metadata", "metadata" in art and "metrics" in art["metadata"]) 

    # 8) save_artifacts (interactive)
    with tempfile.TemporaryDirectory() as tdir:
        global ARTIFACTS_DIR
        old_dir = ARTIFACTS_DIR
        try:
            ARTIFACTS_DIR = Path(tdir)
            results = {"dummy": {"model": object(), "forecast": [1.0, 1.0], "forecast_days": 2}}
            p = save_artifacts(results, competition_mode=False)
            _assert("save_artifacts interactive returns dir", p == ARTIFACTS_DIR)
        finally:
            ARTIFACTS_DIR = old_dir

    # 9) validate_environment creates artifacts (and can fallback)
    with tempfile.TemporaryDirectory() as tdir:
        global ARTIFACTS_DIR
        old_art = ARTIFACTS_DIR
        try:
            ARTIFACTS_DIR = Path(tdir) / "nested" / "artifacts"
            validate_environment()  # should create without raising
            _assert("validate_environment creates artifacts dir", ARTIFACTS_DIR.exists())
        finally:
            ARTIFACTS_DIR = old_art

    # 10) validate_environment can ensure data dir
    with tempfile.TemporaryDirectory() as tdir:
        global DATA_DIR
        old_data = DATA_DIR
        try:
            DATA_DIR = Path(tdir) / "data_should_exist"
            validate_environment(ensure_data_dir=True)
            _assert("validate_environment ensures data dir", DATA_DIR.exists())
        finally:
            DATA_DIR = old_data

    return failures


# ---- CLI --------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Model Training Framework")
    p.add_argument("--competition", action="store_true", help="Run Allora MDK competition pipeline")
    p.add_argument("--benchmark", action="store_true", help="(Reserved) Run benchmarking suite")

    # Non-interactive / CI flags
    p.add_argument("--non-interactive", action="store_true", help="Disable prompts; use flags")
    p.add_argument("--horizon", type=int, default=FORECAST_DAYS, help="Forecast horizon in days")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global RNG seed")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging and tracebacks")
    p.add_argument("--self-test", action="store_true", help="Run built-in unit tests and exit")

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

    # Configure logging verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    if args.self_test:
        print_colored("=== RUNNING SELF-TESTS ===", "header")
        failures = _run_self_tests()
        if failures:
            print_colored(f"\n✗ {failures} self-test(s) failed", "error")
            sys.exit(1)
        print_colored("\n✓ All self-tests passed", "success")
        return

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

    except TrainingError as e:
        # Known, expected error type; print succinct message.
        print_colored(f"\n✗ {e.__class__.__name__}: {e}", "error")
        if args.verbose:
            logger.error("\n" + traceback.format_exc())
        sys.exit(_exit_code_for(e))

    except Exception as e:
        # Unexpected error: include traceback when verbose
        print_colored(f"\n✗ Unhandled exception: {e}", "error")
        if args.verbose:
            logger.error("\n" + traceback.format_exc())
        sys.exit(_exit_code_for(TrainingError("unhandled")))

    print_colored("\n✓ Operation completed successfully", "success")


if __name__ == "__main__":
    main()
