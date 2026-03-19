#!/bin/bash
# SimVLA Training Script for LIBERO (Large Model)
# 
# Key features:
#   - 384x384 image resolution (SmolVLM requirement)
#   - All views processed together by VLM (no aux_visual_inputs)
#   - Larger action transformer configuration

set -e

# =============================================================================
# Command line arguments (with defaults)
# =============================================================================

BATCH_SIZE=${1:-64}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./runs/simvla_libero_large}
RESUME_CKPT=${4:-""}

echo "Training parameters:"
echo "   batch_size: $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir: $OUTPUT_DIR"
echo "   resume_ckpt: ${RESUME_CKPT:-'None (training from scratch)'}"

# GPU configuration
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Suppress TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=2

# =============================================================================
# Path configuration
# =============================================================================
LIBERO_DATA_DIR="./datasets/metas"
NORM_STATS_PATH="./norm_stats/libero_norm.json"
TRAIN_METAS_PATH="./datasets/metas/libero_train.json"

# SmolVLM backbone (can be local path or HuggingFace repo)
SMOLVLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"

# =============================================================================
# Training hyperparameters
# =============================================================================
LEARNING_RATE=2e-4
NUM_ACTIONS=10          # Action horizon
ITERS=200000
WARMUP_STEPS=0
FREEZE_STEPS=1000
SAVE_INTERVAL=10000
LOG_INTERVAL=20
NUM_WORKERS=4
MAX_GRAD_NORM=1.0

# Model architecture (Large configuration)
HIDDEN_SIZE=1024
DEPTH=24
NUM_HEADS=16
USE_ADALN=false          # DiT-style conditioning

# =============================================================================
# Step 1: Create training metadata (if not exists)
# =============================================================================
if [ ! -f "$TRAIN_METAS_PATH" ]; then
    echo "Creating training metadata..."
    python create_libero_meta.py \
        --data_dir $LIBERO_DATA_DIR \
        --subsets libero_10 libero_goal libero_object libero_spatial libero_90 \
        --output $TRAIN_METAS_PATH
fi

# =============================================================================
# Step 2: Compute normalization statistics (if not exists)
# =============================================================================
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "Computing normalization statistics..."
    python compute_libero_norm_stats.py \
        --data_dir $LIBERO_DATA_DIR \
        --subsets libero_10 libero_goal libero_object libero_spatial libero_90 \
        --output $NORM_STATS_PATH
fi

# =============================================================================
# Step 3: Build training arguments
# =============================================================================
ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode libero_joint \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef ${LEARNING_COEF} \
    --num_actions ${NUM_ACTIONS} \
    --iters ${ITERS} \
    --warmup_steps ${WARMUP_STEPS} \
    --freeze_steps ${FREEZE_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --depth ${DEPTH} \
    --num_heads ${NUM_HEADS} \
    --num_workers ${NUM_WORKERS} \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --image_size 384 \
    --norm_stats_path ${NORM_STATS_PATH} \
    --max_grad_norm ${MAX_GRAD_NORM}"

# Add AdaLN flag if enabled
if [ "${USE_ADALN}" = true ]; then
    ARGS="${ARGS} --use_adaln"
fi

# Add resume checkpoint if specified
if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

# =============================================================================
# Step 4: Start training
# =============================================================================
echo "============================================================"
echo "Starting SimVLA Training on LIBERO (Large Action Transformer)"
echo "============================================================"
echo "SmolVLM backbone: ${SMOLVLM_MODEL}"
echo "Data directory: $LIBERO_DATA_DIR"
echo "Normalization stats: $NORM_STATS_PATH"
echo "Action mode: libero_joint"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Learning coef: ${LEARNING_COEF}"
echo "Num actions: ${NUM_ACTIONS}"
echo "Image size: 384x384"
echo "============================================================"
echo "Action Transformer configuration:"
echo "   Hidden size: ${HIDDEN_SIZE}"
echo "   Depth: ${DEPTH}"
echo "   Num heads: ${NUM_HEADS}"
echo "   Use AdaLN: ${USE_ADALN}"
echo "   Estimated parameters: ~302M"
echo "============================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"

# Multi-GPU training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=4 \
    --main_process_port 29504 \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS}

echo "Training completed!"
