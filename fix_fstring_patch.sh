#!/bin/bash
set -euo pipefail

TRAIN_FILE="/workspaces/allora-mdk/train.py"
BACKUP_FILE="${TRAIN_FILE}.before_fstring_fix.bak"

echo "📦 Backing up current train.py to: $BACKUP_FILE"
cp "$TRAIN_FILE" "$BACKUP_FILE"

echo "🛠️  Fixing unterminated f-string at line 685..."

# Use Python to do a safe replacement of the broken print_colored block
python3 - <<EOF
from pathlib import Path
import re

file_path = Path("$TRAIN_FILE")
code = file_path.read_text()

fixed_print = '''
    print_colored(
        f"✓ Competition submission saved: {save_path}\\n"
        f"• Confidence: {artifacts['confidence']:.2f}\\n"
        f"• Forecast: {forecast[0]:.2f} → {forecast[-1]:.2f}",
        "success"
    )
'''

# Replace the broken line that starts with print_colored(...) near save_path
code = re.sub(
    r'print_colored\((.|\n)*?save_path(.|\n)*?\)',
    fixed_print,
    code,
    flags=re.MULTILINE
)

file_path.write_text(code)
print("✅ Fixed unterminated f-string.")
EOF

echo -e "\n🎯 Done! Re-run with:\n"
echo "    python3 train.py --competition --model prophet --asset BTC --horizon 15 --retry 2"
