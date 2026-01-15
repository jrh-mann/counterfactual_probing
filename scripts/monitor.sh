#!/bin/bash
# Monitor the overnight counterfactual probing run

LOG_DIR="logs"

# Find the most recent progress file
PROGRESS_FILE=$(ls -t ${LOG_DIR}/progress_*.txt 2>/dev/null | head -1)
LOG_FILE=$(ls -t ${LOG_DIR}/run_*.log 2>/dev/null | head -1)

if [ -z "$PROGRESS_FILE" ]; then
    echo "No active run found. Start one with:"
    echo "  tmux new -s cfprobe './scripts/run_overnight.sh'"
    exit 1
fi

echo "=============================================="
echo "COUNTERFACTUAL PROBING - MONITOR"
echo "=============================================="
echo "Progress file: $PROGRESS_FILE"
echo "Log file: $LOG_FILE"
echo "=============================================="
echo ""

# Show current progress
echo "=== CURRENT PROGRESS ==="
cat "$PROGRESS_FILE"
echo ""

# Show output file count
CONFIG_OUTPUT=$(grep -o '"dir":[^,}]*' examples/math/config_qwen3_4b_1000.json 2>/dev/null | head -1 | cut -d'"' -f4)
if [ -n "$CONFIG_OUTPUT" ] && [ -d "$CONFIG_OUTPUT" ]; then
    COUNT=$(ls -1 "$CONFIG_OUTPUT"/*.json 2>/dev/null | wc -l)
    echo "=== OUTPUT FILES ==="
    echo "Directory: $CONFIG_OUTPUT"
    echo "Files generated: $COUNT"
    echo ""
fi

# Show last few log lines
if [ -f "$LOG_FILE" ]; then
    echo "=== RECENT LOG (last 20 lines) ==="
    tail -20 "$LOG_FILE"
fi

echo ""
echo "=============================================="
echo "Commands:"
echo "  Watch live:     watch -n 10 ./scripts/monitor.sh"
echo "  Tail log:       tail -f $LOG_FILE"
echo "  Attach tmux:    tmux attach -t cfprobe"
echo "=============================================="
