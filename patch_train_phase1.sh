#!/bin/bash
set -euo pipefail

TRAIN_FILE="/workspaces/allora-mdk/train.py"
BACKUP_FILE="${TRAIN_FILE}.bak"

echo "📦 Backing up original train.py to $BACKUP_FILE"
cp "$TRAIN_FILE" "$BACKUP_FILE"

echo "🛠️  Applying Phase 1: Refactored competition pipeline..."

# Use a Python inline script to patch the content in-place
python3 - <<EOF
from pathlib import Path

file_path = Path("$TRAIN_FILE")
code = file_path.read_text()

# === Insert new helper functions ===
injection = """
# === Phase 1 Helpers ===
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
    raise DataError(f"No CSV found for asset: {asset}")

def load_competition_data(asset: str = "ETH", dataset_path: str = None) -> pd.DataFrame:
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
            time.sleep(1 + random())
"""

# Add helpers after package_competition_artifacts
if "def package_competition_artifacts" in code:
    split_index = code.index("def package_competition_artifacts")
    before = code[:split_index]
    after = code[split_index:]
    code = before + injection + "\n\n" + after

# === Replace old run_competition_mode ===
import re

code = re.sub(
    r"def run_competition_mode\(\)[\s\S]+?return artifacts",
    '''
def run_competition_mode(
    model_name: str = "prophet",
    asset: str = "ETH",
    dataset: str = None,
    horizon: int = 30,
    retries: int = 2
) -> Dict[str, Any]:
    set_global_seed()
    validate_environment(ensure_data_dir=True)
    data = load_competition_data(asset, dataset)

    try:
        ModelFactory = importlib.import_module("models.model_factory").ModelFactory  # type: ignore
        model = ModelFactory().create_model(model_name)
        model.train(data)
    except Exception as e:
        raise ModelError(f"Training failed: {e}") from e

    try:
        raw_forecast = _forecast_with_retry(model, horizon, retries)
        forecast = normalize_forecast(raw_forecast, days=horizon)
    except Exception as e:
        raise ForecastError(f"Forecast failed: {e}") from e

    artifacts = package_competition_artifacts(model, forecast, data)
    save_path = save_artifacts(artifacts, competition_mode=True)

    print_colored(
        f"✓ Competition submission saved: {save_path}\\n"
        f"• Confidence: {artifacts['confidence']:.2f}\\n"
        f"• Forecast: {forecast[0]:.2f} → {forecast[-1]:.2f}",
        "success",
    )
    return artifacts
''',
    code,
    flags=re.DOTALL
)

# === Patch main() to forward CLI args ===
code = code.replace(
    '_ = run_competition_mode()',
    '''_ = run_competition_mode(
        model_name=args.model,
        asset=args.asset,
        dataset=args.csv_path,
        horizon=args.horizon,
        retries=args.retry
    )'''
)

# === Add CLI args (if missing) ===
if "--model" not in code:
    code = code.replace(
        'p.add_argument("--competition",',
        '''p.add_argument("--model", type=str, default="prophet", help="Model to train (competition mode)")
    p.add_argument("--asset", type=str, default="ETH", help="Asset symbol to load (BTC, ETH, etc.)")
    p.add_argument("--dataset", dest="csv_path", help="Path to CSV (optional override)")
    p.add_argument("--retry", type=int, default=2, help="Retry count for forecast")
    p.add_argument("--competition",'''
    )

file_path.write_text(code)
print("✅ Phase 1 patch applied successfully to train.py")
EOF

echo -e "\n🎉 Done! You can now run the updated competition script like this:\n"
echo "    python3 /workspaces/allora-mdk/train.py --competition --model xgboost --asset BTC --horizon 15 --retry 3"
echo -e "\n🧪 To verify, check the printed confidence + forecast range and look for saved artifact in artifacts/.\n"
