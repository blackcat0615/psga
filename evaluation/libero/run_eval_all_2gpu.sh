#!/bin/bash
# =============================================================================
# SimVLA LIBERO Evaluation Script (Adaptive to GPU count)
# =============================================================================

set -e

# =============================================================================
# LIBERO Environment Setup
#==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"

echo "LIBERO Environment:"
echo "   LIBERO_ROOT: $LIBERO_ROOT"
echo "   PYTHONPATH: $PYTHONPATH"
echo ""

# Default arguments
PORT=${1:-8102}
NUM_TRIALS=${2:-10}
OUTPUT_PREFIX=${3:-"eval_simvla_200k"}
GPUS=${4:-"0 1"}  # Default GPUs: 4 5 6 7

# Parse GPU list
read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Validate GPU count
if [ "$NUM_GPUS" -ne 2 ] && [ "$NUM_GPUS" -ne 4 ]; then
    echo "ERROR: This script requires exactly 2 or 4 GPUs. Got $NUM_GPUS."
    echo "   Usage: $0 <port> <num_trials> <output_prefix> \"<gpu1> <gpu2> [gpu3] [gpu4]\""
    exit 1
fi

# Output directory
OUTPUT_DIR="./eval_simvla_${PORT}"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Starting LIBERO evaluation..."
echo "   Server Port: $PORT"
echo "   Num Trials: $NUM_TRIALS"
echo "   Output Prefix: $OUTPUT_PREFIX"
echo "   Output Dir: $OUTPUT_DIR"
echo "   GPUs detected: $NUM_GPUS (${GPU_ARRAY[@]})"
echo ""

# =============================================================================
# Helper function: run a batch of tasks
# =============================================================================
run_batch() {
    local -n gpu_ref=$1     # name of array variable containing GPUs for this batch
    local -n task_ref=$2    # name of array variable containing task names
    local batch_name=$3

    echo "--- Starting batch: $batch_name ---"
    pids=()
    for i in "${!task_ref[@]}"; do
        task="${task_ref[$i]}"
        gpu="${gpu_ref[$i]}"
        logfile="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${task}.txt"
        echo "   Launching $task on GPU $gpu -> $logfile"
        CUDA_VISIBLE_DEVICES=$gpu python -u libero_client.py \
            --host 127.0.0.1 \
            --port $PORT \
            --client_type websocket \
            --task_suite "libero_${task}" \
            --num_trials $NUM_TRIALS \
            --video_out "$OUTPUT_DIR" > "$logfile" 2>&1 &
        pids+=($!)
    done

    # Wait for all tasks in this batch
    for pid in "${pids[@]}"; do
        wait $pid
    done
    echo "--- Batch $batch_name completed ---"
    echo ""
}

# =============================================================================
# Execute tasks based on available GPUs
# =============================================================================
if [ "$NUM_GPUS" -eq 4 ]; then
    # Run all 4 tasks in parallel
    echo "Launching all 4 evaluation tasks in parallel..."
    ALL_TASKS=("spatial" "object" "goal" "10")
    run_batch GPU_ARRAY ALL_TASKS "parallel-4gpu"
else
    # 2 GPUs: run in two batches
    echo "Launching evaluations in two batches (2 tasks per batch)..."

    # Batch 1: spatial & object
    BATCH1_GPUS=("${GPU_ARRAY[0]}" "${GPU_ARRAY[1]}")
    BATCH1_TASKS=("spatial" "object")
    run_batch BATCH1_GPUS BATCH1_TASKS "1-spatial-object"

    # Batch 2: goal & 10
    BATCH2_GPUS=("${GPU_ARRAY[0]}" "${GPU_ARRAY[1]}")
    BATCH2_TASKS=("goal" "10")
    run_batch BATCH2_GPUS BATCH2_TASKS "2-goal-10"
fi

echo ""
echo "All evaluations completed!"
echo ""
echo "Results summary:"
echo "=========================================="
for suite in spatial object goal 10; do
    file="${OUTPUT_PREFIX}_${suite}.txt"
    if [ -f "$file" ]; then
        echo "--- $suite ---"
        grep -E "Success Rate|Average" "$file" 2>/dev/null || echo "  (see $file)"
    fi
done
echo "=========================================="