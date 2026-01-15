#!/bin/bash
# Setup environment variables for cloud training with local datasets
#
# Usage:
#   source scripts/setup_cloud_env.sh
#   # Then run your training commands

# Unset deprecated variable
unset LEROBOT_HOME

# Set JAX to use 90% of GPU memory (important for full fine-tuning on 80GB H100)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Verify settings
echo "✓ Environment configured:"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION: $XLA_PYTHON_CLIENT_MEM_FRACTION"

# Check if dataset exists (config now uses absolute path)
DATASET_PATH="/anvil/scratch/x-jsong13/openpi_maniskill/finetune_pi/franka/maniskill_pickcube_200"
if [ -d "$DATASET_PATH" ]; then
    echo "  ✓ Dataset found: $DATASET_PATH"
    # Check key directories
    if [ -d "$DATASET_PATH/data" ] && [ -d "$DATASET_PATH/meta_data" ]; then
        echo "  ✓ Dataset structure looks good (data/ and meta_data/ exist)"
    else
        echo "  ⚠️  WARNING: Dataset structure incomplete"
    fi
else
    echo "  ✗ ERROR: Dataset not found at $DATASET_PATH"
    echo "     Please run data conversion first"
    exit 1
fi

echo ""
echo "Ready to run:"
echo "  uv run scripts/compute_norm_stats.py --config-name pi0_franka_maniskill"
echo "  uv run scripts/train.py pi0_franka_maniskill --exp-name my_experiment"
