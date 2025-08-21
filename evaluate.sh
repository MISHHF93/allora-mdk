#!/bin/bash

# -----------------------------
# Allora Workers: Evaluation Runner
# -----------------------------

echo "[INFO] Starting model evaluation..."

# === Define directories ===
ARTIFACTS_DIR="data/artifacts"
DATASETS_DIR="data/sets"
EVALUATION_DIR="evaluation"
PLOTS_DIR="$EVALUATION_DIR/plots"
SCORES_FILE="$EVALUATION_DIR/score_table.csv"

# === Create output folders ===
mkdir -p "$ARTIFACTS_DIR"
mkdir -p "$DATASETS_DIR"
mkdir -p "$PLOTS_DIR"

# === Run Python evaluation ===
python3 <<EOF
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from core.utils.evaluation import evaluate_model
from core.utils.plotting import plot_forecast

ARTIFACTS_DIR = "$ARTIFACTS_DIR"
DATA_DIR = "$DATASETS_DIR"
EVALUATION_DIR = "$EVALUATION_DIR"
PLOTS_DIR = "$PLOTS_DIR"
SCORES_FILE = "$SCORES_FILE"

os.makedirs(EVALUATION_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_latest_csv():
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if not files:
            raise FileNotFoundError("No CSV files found in data/sets/")
        latest = max(files, key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)))
        return os.path.join(DATA_DIR, latest)
    except Exception as e:
        print(f"[ERROR] Could not find valid CSV: {e}")
        return None

def main():
    csv_path = get_latest_csv()
    if not csv_path:
        return

    try:
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    except Exception as e:
        print(f"[ERROR] Failed to load historical data: {e}")
        return

    rows = []

    for file in os.listdir(ARTIFACTS_DIR):
        if not file.endswith(".json"):
            continue

        path = os.path.join(ARTIFACTS_DIR, file)
        try:
            with open(path, "r") as f:
                content = json.load(f)

            model_type = content["model"]
            forecast_vals = content["forecast"]
            confidence = content.get("confidence", None)

            forecast = pd.Series(forecast_vals)
            real = df.iloc[-len(forecast):].copy()

            if real.empty or forecast.empty:
                print(f"[WARN] Skipping {file}: empty data")
                continue

            scores = evaluate_model(real, forecast)
            scores["model"] = model_type
            scores["file"] = file
            scores["confidence"] = confidence

            plot_path = os.path.join(PLOTS_DIR, f"{model_type}_{file.replace('.json', '')}.png")
            plot_forecast(real, forecast, title=f"{model_type} Forecast")
            plt.savefig(plot_path)
            plt.close()

            print(f"[OK] Evaluated and plotted {model_type}")
            rows.append(scores)

        except Exception as e:
            print(f"[ERROR] Processing {file} failed: {e}")
            continue

    if rows:
        df_scores = pd.DataFrame(rows)
        df_scores.to_csv(SCORES_FILE, index=False)
        print(f"[DONE] Saved scores to {SCORES_FILE}")
    else:
        print("[WARN] No evaluations completed.")

if __name__ == "__main__":
    main()
EOF

# === Done ===
echo "[INFO] Evaluation complete."
