#!/bin/bash
# Setup environment variables for cloud training with local datasets
#
# Usage:
#   source scripts/setup_cloud_env.sh
#   # Then run your training commands

# Clean up any deprecated LEROBOT environment variables
unset LEROBOT_HOME
unset HF_LEROBOT_HOME
unset HF_HUB_OFFLINE

# Set JAX to use 90% of GPU memory (important for full fine-tuning on 80GB H100)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "✓ Environment configured for cloud training"
echo ""
echo "Settings:"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION: $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "  Config uses absolute path for local dataset"
echo ""

# Check if dataset exists (config uses absolute path)
DATASET_PATH="/home/x-jsong13/.cache/huggingface/lerobot/franka_maniskill_pickcube_200"
if [ -d "$DATASET_PATH" ]; then
    echo "✓ Dataset found: $DATASET_PATH"
    # Check key directories
    if [ -d "$DATASET_PATH/data" ] && [ -d "$DATASET_PATH/meta" ]; then
        echo "✓ Dataset structure valid (data/ and meta/ exist)"
    else
        echo "⚠️  WARNING: Dataset structure incomplete"
        echo "   Expected: data/ and meta/ directories"
    fi
else
    echo "✗ ERROR: Dataset not found at $DATASET_PATH"
    echo "  Please run data conversion first:"
    echo "  uv run examples/franka/convert_franka_data_to_lerobot.py --h5_path <path>"
    exit 1
fi

echo ""
echo "Ready to run:"
echo "  uv run scripts/compute_norm_stats.py --config-name pi0_franka_maniskill"
echo "  uv run scripts/train.py pi0_franka_maniskill --exp-name my_experiment"
echo ""
