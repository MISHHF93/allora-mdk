#!/bin/bash

# Path to target file
TRAIN_FILE="/workspaces/allora-mdk/train.py"

# Backup original
BACKUP_FILE="${TRAIN_FILE%.py}_backup_$(date +%Y%m%d_%H%M%S).py"
cp "$TRAIN_FILE" "$BACKUP_FILE"
echo "[✓] Backed up existing train.py to: $BACKUP_FILE"

# Overwrite with refactored version
cat > "$TRAIN_FILE" << 'EOF'
import argparse
import pandas as pd
import os
import pickle
import time
from random import random
from datetime import datetime
from pathlib import Path

from models.model_factory import ModelFactory
from utils.metrics import _in_sample_metrics_if_available

DATA_DIR = Path("data/")

# -------------------------
# CLI ARGUMENTS
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--competition", action="store_true")
parser.add_argument("--model", type=str, default="prophet")
parser.add_argument("--asset", type=str, default="ETH")
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--horizon", type=int, default=30)
parser.add_argument("--retry", type=int, default=2)
args = parser.parse_args()

# -------------------------
# UTILS
# -------------------------
def _resolve_asset_csv(asset: str) -> Path:
    patterns = [
        f"{asset.lower()}.csv",
        f"{asset.lower()}*.csv",
        f"*{asset}*historical_data_coinmarketcap*.csv"
    ]
    for pattern in patterns:
        for path in DATA_DIR.rglob(pattern):
            if path.is_file():
                return path
    raise FileNotFoundError(f"No dataset found for asset: {asset}")

def load_competition_data(asset: str, dataset_path: str = None) -> pd.DataFrame:
    if dataset_path:
        return pd.read_csv(dataset_path)
    else:
        path = _resolve_asset_csv(asset)
        return pd.read_csv(path)

def _forecast_with_retry(model, horizon: int, max_retries: int = 2):
    for attempt in range(max_retries + 1):
        try:
            return model.forecast(horizon)
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"[!] Forecast failed (attempt {attempt + 1}/{max_retries}), retrying...")
            time.sleep(1 + random())

def package_competition_artifacts(model, forecast, df, asset):
    metrics = _in_sample_metrics_if_available(model, df) or {}
    artifact = {
        "model": model,
        "forecast": forecast,
        "confidence": 0.95,
        "metadata": {
            "trained_at": datetime.utcnow().isoformat(),
            "asset": asset,
            "metrics": {
                "training_rmse": metrics.get("rmse"),
                "training_mape": metrics.get("mape"),
                "training_rows": len(df),
                "forecast_days": len(forecast),
            }
        }
    }
    os.makedirs("artifacts", exist_ok=True)
    filename = f"artifacts/competition_submission_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(artifact, f)
    print(f"[✓] Artifact saved to: {filename}")
    return artifact

# -------------------------
# COMPETITION PIPELINE
# -------------------------
def run_competition_mode(
    model_name: str = "prophet",
    asset: str = "ETH",
    dataset: str = None,
    horizon: int = 30,
    retries: int = 2
):
    df = load_competition_data(asset, dataset)
    model = ModelFactory().create_model(model_name)
    model.train(df)
    forecast = _forecast_with_retry(model, horizon, retries)
    return package_competition_artifacts(model, forecast, df, asset)

# -------------------------
# ENTRYPOINT
# -------------------------
if args.competition:
    run_competition_mode(
        model_name=args.model,
        asset=args.asset,
        dataset=args.dataset,
        horizon=args.horizon,
        retries=args.retry
    )
EOF

echo "[✓] train.py has been successfully updated."
