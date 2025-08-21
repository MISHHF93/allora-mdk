# evaluate_model.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.plotting import plot_forecast

# === Directories ===
ARTIFACTS_DIR = "data/artifacts"
DATA_DIR = "data/sets"
EVALUATION_DIR = "evaluation"
PLOTS_DIR = os.path.join(EVALUATION_DIR, "plots")
SCORES_FILE = os.path.join(EVALUATION_DIR, "score_table.csv")

os.makedirs(EVALUATION_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Locate Latest CSV ===
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

# === Evaluation Metrics ===
def evaluate_model(actual, forecast):
    if isinstance(actual, pd.DataFrame):
        actual = actual["close"]
    mse = mean_squared_error(actual, forecast)
    mae = mean_absolute_error(actual, forecast)
    r2 = r2_score(actual, forecast)
    return {
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "r2_score": round(r2, 4)
    }

# === Evaluation Logic ===
def main():
    csv_path = get_latest_csv()
    if not csv_path:
        return

    # Load real data
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

            # Check alignment
            if real.empty or forecast.empty:
                print(f"[WARN] Skipping {file}: empty data")
                continue

            # Evaluation
            scores = evaluate_model(real, forecast)
            scores["model"] = model_type
            scores["file"] = file
            scores["confidence"] = confidence

            # Save Plot
            plot_path = os.path.join(PLOTS_DIR, f"{model_type}_{file.replace('.json', '')}.png")
            plot_forecast(real, forecast, title=f"{model_type} Forecast")
            plt.savefig(plot_path)
            plt.close()

            print(f"[OK] Evaluated and plotted {model_type}")
            rows.append(scores)

        except Exception as e:
            print(f"[ERROR] Processing {file} failed: {e}")
            continue

    # Save score table
    if rows:
        df_scores = pd.DataFrame(rows)
        df_scores.to_csv(SCORES_FILE, index=False)
        print(f"[DONE] Saved scores to {SCORES_FILE}")
    else:
        print("[WARN] No evaluations completed.")

if __name__ == "__main__":
    main()
