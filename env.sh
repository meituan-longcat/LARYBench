#!/bin/bash
# =============================================================================
# LARY Environment Configuration
# =============================================================================
# Usage:
#   source /path/to/LARY/env.sh
# =============================================================================

# -----------------------------------------------------------------------------
# Project Paths (REQUIRED)
# -----------------------------------------------------------------------------
export LARY_ROOT="${LARY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}" # 项目文件夹地址
export LARY_LOG_DIR="${LARY_LOG_DIR:-<path/to/logs>}"                            # 保存latent action和分类回归实验结果的目录

export DATA_DIR="${DATA_DIR:-<path/to/LARYBench>}"                               # 数据集保存的目录

# Model weights directory
export MODEL_DIR="${MODEL_DIR:-<path/to/pretrained_lam_weights>}"                # 模型保存地址

# Latent action storage root (written by extract, read by downstream tasks)
export LARY_LA_DIR="${LARY_LA_DIR:-<path/to/latent_actions>}"                   # 保存/读取 latent action 的目录

# -----------------------------------------------------------------------------
# WandB Configuration
# -----------------------------------------------------------------------------
export WANDB_API_KEY="${WANDB_API_KEY:-<your_wandb_api_key>}"
export WANDB_PROJECT="${WANDB_PROJECT:-lary}"

# -----------------------------------------------------------------------------
# Conda Environment
# -----------------------------------------------------------------------------
export CONDA_SH_PATH="${CONDA_SH_PATH:-<path/to/miniconda3/etc/profile.d/conda.sh>}"

# -----------------------------------------------------------------------------
# External Project Paths
# -----------------------------------------------------------------------------

# villa-x model repo (needed by get_latent_action/tokenizer.py when model=villa-x)
export VILLA_X_DIR="${VILLA_X_DIR:-${LARY_ROOT}/get_latent_action/models/villa_x}"

# Per-model checkpoint paths (optional: defaults to $MODEL_DIR/{subpath})
# Uncomment and set only those you need.
export UNIVLA_CKPT_PATH="${UNIVLA_CKPT_PATH:-<path/to/univla/lam-stage-2.ckpt>}"
export VILLA_X_CKPT_PATH="${VILLA_X_CKPT_PATH:-<path/to/villa-x/lam>}"
export AE_MODEL_PATH="${AE_MODEL_PATH:-<path/to/FLUX.2-dev/ae.safetensors>}"
export WAN22_VAE_PATH="${WAN22_VAE_PATH:-<path/to/Wan2.2_VAE.pth>}"
export VJEPA2_CKPT_PATH="${VJEPA2_CKPT_PATH:-<path/to/vjepa2/vitl.pt>}"

# -----------------------------------------------------------------------------
# Pretrained Model Paths
# -----------------------------------------------------------------------------
export DINO_V2_PATH="${DINO_V2_PATH:-<path/to/dinov2-large>}"
export DINO_V3_PATH="${DINO_V3_PATH:-<path/to/dinov3-vitl16>}"

# SigLIP2 encoder
export SIGLIP2_PATH="${SIGLIP2_PATH:-<path/to/siglip2-base-patch16-224>}"

# Open-MagViT2 tokenizer
export MAGVIT2_CONFIG_PATH="${MAGVIT2_CONFIG_PATH:-<path/to/Open-MAGVIT2/pretrain_lfqgan_256_262144.yaml>}"
export MAGVIT2_TOKENIZER_PATH="${MAGVIT2_TOKENIZER_PATH:-<path/to/Open-MAGVIT2/pretrain256_262144.ckpt>}"

# -----------------------------------------------------------------------------
# HTTP Proxy (optional)
# -----------------------------------------------------------------------------
# export HTTP_PROXY_ADDR="http://<proxy_host>:<proxy_port>"
# export http_proxy="$HTTP_PROXY_ADDR"
# export https_proxy="$HTTP_PROXY_ADDR"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

# Activate the appropriate conda environment for a model
lary_activate() {
    local model=$1
    source "$CONDA_SH_PATH"

    case "$model" in
        "wan2-2") conda activate wan ;;
        "villa-x")
            if [[ -n "$VILLA_X_DIR" && -f "$VILLA_X_DIR/.venv/bin/activate" ]]; then
                source "$VILLA_X_DIR/.venv/bin/activate"
            fi
            ;;
        "vjepa2") conda activate vjepa2 ;;
        *) conda activate laq ;;
    esac
}

# Print current configuration
lary_config() {
    echo "LARY Configuration:"
    echo "  LARY_ROOT:     $LARY_ROOT"
    echo "  LARY_LOG_DIR:  $LARY_LOG_DIR"
    echo "  LARY_LA_DIR:   $LARY_LA_DIR"
    echo "  DATA_DIR:      $DATA_DIR"
    echo "  MODEL_DIR:     $MODEL_DIR"
    echo "  DINO_V2_PATH:  $DINO_V2_PATH"
    echo "  DINO_V3_PATH:  $DINO_V3_PATH"
    echo "  VILLA_X_DIR:   $VILLA_X_DIR"
}


















