#!/bin/bash
# Staged run: Generate 800 rollouts, select shortest 500 for counterfactuals

set -e

CONFIG="examples/math/config_qwen3_4b_staged.json"
OUTPUT_DIR="outputs/qwen3-4b/math_staged"
ROLLOUTS_FILE="$OUTPUT_DIR/rollouts.jsonl"

# Number of rollouts to generate in Stage 1
NUM_ROLLOUTS=800

# Number of shortest valid rollouts to select for Stage 2
SELECT_COUNT=500

echo "============================================================"
echo "STAGED COUNTERFACTUAL PROBING"
echo "============================================================"
echo "Config: $CONFIG"
echo "Stage 1: Generate $NUM_ROLLOUTS rollouts"
echo "Stage 2: Select shortest $SELECT_COUNT for counterfactuals"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run both stages
cd /root/counterfactual_probing
source .venv/bin/activate

python3 -c "
from counterfactual_probing.run_staged import run_both_stages

stats = run_both_stages(
    config_path='$CONFIG',
    num_rollouts=$NUM_ROLLOUTS,
    select_count=$SELECT_COUNT,
    rollouts_path='$ROLLOUTS_FILE',
    output_dir='$OUTPUT_DIR',
)

print()
print('Final stats:', stats)
"
