#!/bin/bash
# Run counterfactual probing overnight with logging and progress tracking

set -e

# Configuration
CONFIG="${1:-examples/math/config_qwen3_4b_1000.json}"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"
PROGRESS_FILE="${LOG_DIR}/progress_${TIMESTAMP}.txt"

# Create log directory
mkdir -p "$LOG_DIR"

# Print startup info
echo "=============================================="
echo "COUNTERFACTUAL PROBING - OVERNIGHT RUN"
echo "=============================================="
echo "Started at: $(date)"
echo "Config: $CONFIG"
echo "Log file: $LOG_FILE"
echo "Progress file: $PROGRESS_FILE"
echo "=============================================="

# Activate virtual environment
source /root/counterfactual_probing/.venv/bin/activate

# Export progress file location for the Python script
export PROGRESS_FILE="$PROGRESS_FILE"

# Run the pipeline with tee to both log and display
echo "Starting run at $(date)" > "$PROGRESS_FILE"
python -u -c "
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'src')

from counterfactual_probing import run
from counterfactual_probing.config import load_config
from counterfactual_probing.dataset import Dataset

config_path = '$CONFIG'
progress_file = os.environ.get('PROGRESS_FILE', 'progress.txt')

# Load config to get dataset size
config = load_config(config_path)
dataset = Dataset(path=config.dataset.path, prompt_field=config.dataset.prompt_field)
total_prompts = len(list(dataset))

print(f'Total prompts to process: {total_prompts}')
print(f'Output directory: {config.output.dir}')
print()

# Track progress by counting output files
output_dir = Path(config.output.dir)
output_dir.mkdir(parents=True, exist_ok=True)

start_time = time.time()

def update_progress():
    completed = len(list(output_dir.glob('*.json')))
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0
    remaining = (total_prompts - completed) / rate if rate > 0 else 0

    with open(progress_file, 'w') as f:
        f.write(f'Status: RUNNING\n')
        f.write(f'Started: {datetime.fromtimestamp(start_time).isoformat()}\n')
        f.write(f'Completed: {completed}/{total_prompts} ({100*completed/total_prompts:.1f}%)\n')
        f.write(f'Elapsed: {elapsed/3600:.1f}h\n')
        f.write(f'Rate: {rate*3600:.1f} prompts/hour\n')
        f.write(f'ETA: {remaining/3600:.1f}h remaining\n')
        f.write(f'Last update: {datetime.now().isoformat()}\n')

# Start progress monitor in background thread
import threading
stop_monitor = threading.Event()

def monitor_loop():
    while not stop_monitor.is_set():
        try:
            update_progress()
        except:
            pass
        stop_monitor.wait(30)  # Update every 30 seconds

monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
monitor_thread.start()

try:
    # Run the pipeline
    run(config_path)

    # Final update
    with open(progress_file, 'w') as f:
        completed = len(list(output_dir.glob('*.json')))
        elapsed = time.time() - start_time
        f.write(f'Status: COMPLETED\n')
        f.write(f'Started: {datetime.fromtimestamp(start_time).isoformat()}\n')
        f.write(f'Completed: {completed}/{total_prompts} (100%)\n')
        f.write(f'Total time: {elapsed/3600:.2f}h\n')
        f.write(f'Finished: {datetime.now().isoformat()}\n')

    print()
    print('============================================')
    print('RUN COMPLETED SUCCESSFULLY')
    print(f'Total time: {elapsed/3600:.2f} hours')
    print('============================================')

except Exception as e:
    with open(progress_file, 'w') as f:
        f.write(f'Status: FAILED\n')
        f.write(f'Error: {str(e)}\n')
        f.write(f'Time: {datetime.now().isoformat()}\n')
    raise

finally:
    stop_monitor.set()
" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Run finished at $(date)"
echo "Log saved to: $LOG_FILE"
