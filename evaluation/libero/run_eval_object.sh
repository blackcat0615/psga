#!/bin/bash
# =============================================================================
# SimVLA LIBERO Evaluation Script - Single Task (object)
# =============================================================================

set -e

# =============================================================================
# LIBERO Environment Setup
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"

echo "LIBERO Environment:"
echo "   LIBERO_ROOT: $LIBERO_ROOT"
echo "   PYTHONPATH: $PYTHONPATH"
echo ""

# =============================================================================
# Parameter settings (with defaults)
# =============================================================================
PORT=${1:-8102}                 # WebSocket server port
NUM_TRIALS=${2:-10}             # Number of trials per task
OUTPUT_PREFIX=${3:-"eval_simvla_object"}  # Prefix for log files
GPU=${4:-0}                      # GPU device ID to use (default: 0)


# Create output directory
OUTPUT_DIR="./reeval/eval_simvla_exp3_att_add_language_noise_${PORT}"
mkdir -p "$OUTPUT_DIR"

echo "Starting LIBERO evaluation for object task..."
echo "   Server Port: $PORT"
echo "   Num Trials: $NUM_TRIALS"
echo "   Output Prefix: $OUTPUT_PREFIX"
echo "   Output Dir: $OUTPUT_DIR"
echo "   Using GPU: $GPU"
echo ""

# =============================================================================
# Run single object task
# =============================================================================
logfile="${OUTPUT_DIR}/${OUTPUT_PREFIX}_object.txt"
CUDA_VISIBLE_DEVICES=$GPU python -u libero_client.py \
    --host 127.0.0.1 \
    --port $PORT \
    --client_type websocket \
    --task_suite "libero_object" \
    --add_img_noise_ratio 0.0  \
    --num_trials $NUM_TRIALS \
    --video_out "$OUTPUT_DIR" > "$logfile" 2>&1 \
    --add_language_noise \

echo ""
echo "Evaluation completed!"
echo ""

# =============================================================================
# Display results
# =============================================================================
echo "Results:"
echo "=========================================="
if [ -f "$logfile" ]; then
    grep -E "Success Rate|Average" "$logfile" 2>/dev/null || echo "  (no success rate info found in log)"
else
    echo "  Log file not found: $logfile"
fi
echo "=========================================="