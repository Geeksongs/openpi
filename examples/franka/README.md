# Franka ManiSkill Dataset for OpenPI

This directory contains tools for converting ManiSkill demonstration data (Franka Panda robot) to LeRobot format and training pi0 models on it.

## Overview

- **Robot**: Franka Panda
- **Environment**: ManiSkill PickCube-v1
- **Control Mode**: Joint position control (pd_joint_pos)
- **Data Format**: HDF5 files from ManiSkill
- **Action Space**: 8-dim (7 joints + 1 gripper)
- **Observations**: RGB images (128x128x3), robot state (qpos/qvel)

## Files

### 1. `load_maniskill_h5.py`
**Purpose**: Load and parse ManiSkill HDF5 trajectory files

**Usage**:
```bash
uv run examples/franka/load_maniskill_h5.py --h5_path /path/to/trajectory_200.h5
```

**Output**: Validates data format and prints statistics
- ✓ RGB images: (T, 128, 128, 3) uint8
- ✓ Depth images: (T, 128, 128, 1) int16
- ✓ Robot state (qpos): (T, 9) float32
- ✓ Robot state (qvel): (T, 9) float32
- ✓ Actions: (T, 8) float32
- ✓ TCP pose: (T, 7) float32
- ✓ Success flags: (T,) bool

### 2. `convert_franka_data_to_lerobot.py`
**Purpose**: Convert ManiSkill h5 data to LeRobot dataset format

**Usage**:
```bash
# Basic conversion
uv run examples/franka/convert_franka_data_to_lerobot.py \
    --h5_path /path/to/trajectory_200.h5 \
    --repo_id your_hf_username/maniskill_pickcube

# With custom settings
uv run examples/franka/convert_franka_data_to_lerobot.py \
    --h5_path /path/to/trajectory_200.h5 \
    --repo_id your_hf_username/maniskill_pickcube \
    --task_name "Pick up the cube and place it at the goal position" \
    --fps 10 \
    --push_to_hub
```

**Output**: Creates LeRobot dataset at `~/.cache/huggingface/lerobot/your_repo_id`

**Features**:
- Preserves original 128x128 resolution (no resizing)
- Uses 8-dim state (first 8 dims of qpos to match actions)
- Automatically handles time alignment (obs has T+1 frames, actions have T frames)

### 3. Policy Configuration
**File**: `src/openpi/policies/franka_policy.py`

**Classes**:
- `FrankaInputs`: Transforms LeRobot data → pi0 model input
  - Maps single base camera to 3-camera format (with padding)
  - Handles image format conversion (CHW↔HWC, float32↔uint8)
  - Supports pi0, pi0-FAST, and pi05 models

- `FrankaOutputs`: Transforms pi0 model output → Franka actions
  - Extracts first 8 dims from 32-dim model output

### 4. Testing
**File**: `test_franka_policy.py`

**Usage**:
```bash
uv run examples/franka/test_franka_policy.py
```

**Tests**:
- ✓ FrankaInputs with all model types
- ✓ FrankaOutputs action extraction
- ✓ Image parsing (CHW/HWC, uint8/float32)
- ✓ Example generation

## Data Conversion Details

### LeRobot Dataset Schema
```python
features = {
    "image": {
        "dtype": "image",
        "shape": (128, 128, 3),  # Original resolution preserved
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (8,),  # First 8 dims of qpos (7 joints + 1 gripper)
        "names": ["state"],
    },
    "actions": {
        "dtype": "float32",
        "shape": (8,),  # 7 joint positions + 1 gripper
        "names": ["actions"],
    },
}
```

### Important Design Decisions

1. **No Image Resizing**: We keep the original 128x128 resolution
   - Pi0's image preprocessing will handle resizing automatically during training
   - Different robots use different resolutions (DROID:180x320, LIBERO:256x256, ALOHA:480x640)

2. **8-Dim State**: Use first 8 dimensions of qpos (not all 9)
   - Matches action dimension
   - ManiSkill qpos[8] is the last gripper dimension, less critical

3. **Simple Feature Naming**: Use `image`, `state`, `actions` (like LIBERO)
   - Avoids issues with `/` in feature names
   - Cleaner than DROID's `exterior_image_1_left` style

4. **Single Camera**: Only base_camera RGB
   - No wrist cameras in ManiSkill data
   - Padding with zeros for missing cameras
   - Image mask properly configured for pi0/pi0-FAST

5. **Time Alignment**: Critical handling of observation/action mismatch
   - Observations: T+1 frames (includes terminal state)
   - Actions: T frames
   - Solution: Truncate observations to match actions

## Next Steps

### 1. Convert Your Data
```bash
uv run examples/franka/convert_franka_data_to_lerobot.py \
    --h5_path /home/python/Desktop/fine_tune_openpi/trajectory_200.h5 \
    --repo_id your_hf_username/maniskill_pickcube_200
```

### 2. Create Training Config
Add to `src/openpi/training/config.py`:

```python
from openpi.policies import franka_policy
from openpi.models import pi0_config

# Define config
def franka_pickcube_config() -> TrainConfig:
    model = pi0_config.Pi0Config(
        action_dim=8,  # 7 joints + 1 gripper
        action_horizon=10,
        max_token_len=180,
    )

    return TrainConfig(
        model=model,
        data=LeRobotDataConfig(
            repo_id="your_hf_username/maniskill_pickcube_200",
            transforms=Transforms(
                inputs=[franka_policy.FrankaInputs(model_type=ModelType.PI0)],
                outputs=[franka_policy.FrankaOutputs()],
            ),
        ),
        batch_size=16,
        learning_rate=1e-4,
        num_iterations=20000,
        save_interval=5000,
        weight_loader=WeightLoader(
            checkpoint_dir="gs://openpi-assets/checkpoints/pi0_base",
        ),
    )

# Register config
_CONFIG_MAP["franka_pickcube"] = franka_pickcube_config
```

### 3. Compute Normalization Statistics
```bash
uv run scripts/compute_norm_stats.py --config-name franka_pickcube
```

This will create `norm_stats.json` in your config directory.

### 4. Start Training
```bash
# Enable high GPU memory usage
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py franka_pickcube \
    --exp-name=my_first_experiment \
    --overwrite
```

**Training Tips**:
- Monitor on Weights & Biases (auto-logged)
- Checkpoints saved to `checkpoints/franka_pickcube/my_first_experiment/`
- Use `--overwrite` to replace existing checkpoints

### 5. Run Inference
```bash
# Start policy server
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=franka_pickcube \
    --policy.dir=checkpoints/franka_pickcube/my_first_experiment/20000
```

Then connect your robot client to the server (default port: 8000).

## Comparison with Other Platforms

| Feature | Franka (Ours) | LIBERO | DROID | ALOHA |
|---------|---------------|---------|-------|--------|
| Robot | Panda | Panda | Franka | Trossen X |
| Cameras | 1 (base) | 2 (base+wrist) | 2 (exterior+wrist) | 4 (high+low+2 wrists) |
| Resolution | 128x128 | 256x256 | 180x320 | 480x640 |
| State Dims | 8 | 8 | 8 (7+1) | 14 (dual-arm) |
| Action Dims | 8 | 7 | 8 | 14 |
| Control | Position | Position | Velocity | Position |
| Data Source | ManiSkill h5 | RLDS (tfds) | h5 + MP4 | h5 (compressed) |

## Troubleshooting

### Issue: "Feature names should not contain '/'"
**Solution**: Use simple names (`image`, `state`, `actions`) instead of `observation/image`

### Issue: Shape mismatch between observations and actions
**Solution**: Time alignment is handled in `convert_franka_data_to_lerobot.py` (line 120-128)

### Issue: Out of memory during training
**Solutions**:
1. Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`
2. Reduce batch size
3. Use LoRA fine-tuning instead of full fine-tuning
4. Enable FSDP with `--fsdp-devices <num_gpus>`

### Issue: Training loss diverges
**Solution**: Check `norm_stats.json` for extreme values. Manually adjust if needed.

## References

- [OpenPI Documentation](https://github.com/Physical-Intelligence/openpi)
- [ManiSkill Documentation](https://maniskill.readthedocs.io/)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [Pi0 Paper](https://www.physicalintelligence.company/blog/pi0)

## Dataset Statistics (Example: trajectory_200.h5)

- Total trajectories: 200
- Total frames: 15,620
- Average frames/trajectory: 78.1
- Frame range: 55-98 steps
- Success rate: 100%
- Task: PickCube-v1 (pick up cube and place at goal)

## Contact

For questions or issues related to Franka ManiSkill integration, please file an issue on the OpenPI repository.
