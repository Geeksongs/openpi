"""
Convert ManiSkill Franka h5 data to LeRobot format.

This script converts ManiSkill demonstration data (stored in h5 format) to the
LeRobot dataset format, which can be used for training pi0 models.

Usage:
    uv run examples/franka/convert_franka_data_to_lerobot.py \
        --h5_path /path/to/trajectory_200.h5 \
        --repo_id your_hf_username/maniskill_pickcube

Optional arguments:
    --task_name: Task description (default: "Pick up the cube and place it at the goal position")
    --push_to_hub: Push the dataset to Hugging Face Hub after conversion
    --fps: Frames per second of the robot control loop (default: 10)
"""

import shutil
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
import tyro

# Import our h5 loading function
from load_maniskill_h5 import load_maniskill_trajectory


def convert_to_lerobot(
    h5_path: str,
    repo_id: str = "your_hf_username/maniskill_pickcube",
    task_name: str = "Pick up the cube and place it at the goal position",
    fps: int = 10,
    *,
    push_to_hub: bool = False,
) -> None:
    """
    Convert ManiSkill h5 data to LeRobot format.

    Args:
        h5_path: Path to the ManiSkill h5 file
        repo_id: Repository ID for the LeRobot dataset (format: username/dataset_name)
        task_name: Natural language description of the task
        fps: Frames per second (default: 10)
        push_to_hub: Whether to push the dataset to Hugging Face Hub
    """
    print("="*80)
    print("MANISKILL TO LEROBOT CONVERTER")
    print("="*80)
    print(f"\nInput h5 file:  {h5_path}")
    print(f"Output repo ID: {repo_id}")
    print(f"Task name:      {task_name}")
    print(f"FPS:            {fps}")
    print(f"Push to hub:    {push_to_hub}")

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"\n⚠️  Removing existing dataset at: {output_path}")
        shutil.rmtree(output_path)

    # Load ManiSkill trajectories
    print(f"\n{'─'*80}")
    print("STEP 1: Loading ManiSkill trajectories")
    print("─"*80)
    trajectories = load_maniskill_trajectory(h5_path)
    print(f"✓ Loaded {len(trajectories)} trajectories")

    # Create LeRobot dataset
    print(f"\n{'─'*80}")
    print("STEP 2: Creating LeRobot dataset")
    print("─"*80)

    # Define features schema
    # Note: We keep the original 128x128 resolution (no resizing)
    # Note: We use 8-dim state (first 8 dimensions of qpos) to match action dimension
    # Note: Using simple names (like LIBERO) instead of observation/ prefix
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (128, 128, 3),  # Keep original resolution
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),  # 8-dim: first 8 dimensions of qpos (7 joints + 1 gripper)
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # 8-dim: 7 joint positions + 1 gripper
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    print(f"✓ Created LeRobot dataset with schema:")
    print(f"  - image: (128, 128, 3) uint8")
    print(f"  - state: (8,) float32")
    print(f"  - actions: (8,) float32")

    # Convert and write episodes
    print(f"\n{'─'*80}")
    print("STEP 3: Converting and writing episodes")
    print("─"*80)

    total_frames = 0

    for traj_idx, traj in enumerate(tqdm(trajectories, desc="Converting episodes")):
        num_steps = len(traj['actions'])
        total_frames += num_steps

        # Add each frame to the dataset
        for step_idx in range(num_steps):
            frame_data = {
                "image": traj['rgb_images'][step_idx],  # (128, 128, 3) uint8
                "state": traj['qpos'][step_idx, :8].astype(np.float32),  # Take first 8 dims
                "actions": traj['actions'][step_idx].astype(np.float32),  # (8,) float32
                "task": task_name,  # Language instruction
            }
            dataset.add_frame(frame_data)

        # Save the episode
        dataset.save_episode()

    print(f"✓ Converted {len(trajectories)} episodes with {total_frames} total frames")

    # Print dataset statistics
    print(f"\n{'─'*80}")
    print("DATASET STATISTICS")
    print("─"*80)
    print(f"Total episodes:     {len(trajectories)}")
    print(f"Total frames:       {total_frames}")
    print(f"Avg frames/episode: {total_frames / len(trajectories):.1f}")
    print(f"Dataset size:       {output_path}")

    # Optionally push to Hugging Face Hub
    if push_to_hub:
        print(f"\n{'─'*80}")
        print("STEP 4: Pushing to Hugging Face Hub")
        print("─"*80)
        dataset.push_to_hub(
            tags=["maniskill", "panda", "pick-cube", "franka"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"✓ Pushed to Hugging Face Hub: {repo_id}")

    print(f"\n{'='*80}")
    print("✓ CONVERSION COMPLETE!")
    print("="*80)
    print(f"\nDataset location: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Create policy config file: src/openpi/policies/franka_policy.py")
    print(f"2. Create training config in: src/openpi/training/config.py")
    print(f"3. Compute normalization stats: uv run scripts/compute_norm_stats.py --config-name franka_config")
    print(f"4. Start training: uv run scripts/train.py franka_config --exp-name=my_experiment")
    print()


if __name__ == "__main__":
    tyro.cli(convert_to_lerobot)
