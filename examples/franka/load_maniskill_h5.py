"""
Load ManiSkill h5 trajectory data.

This module provides utilities to load and parse ManiSkill demonstration data
stored in HDF5 format.

Usage:
    uv run examples/franka/load_maniskill_h5.py --h5_path /path/to/trajectory.h5
"""

import h5py
import numpy as np
from typing import Dict, List
import tyro


def load_maniskill_trajectory(h5_path: str) -> List[Dict[str, np.ndarray]]:
    """
    Load all trajectories from a ManiSkill h5 file.

    Args:
        h5_path: Path to the ManiSkill h5 file

    Returns:
        List of trajectory dictionaries, each containing:
            - 'rgb_images': (T, 128, 128, 3) uint8 array
            - 'depth_images': (T, 128, 128, 1) int16 array
            - 'qpos': (T, 9) float32 array - robot joint positions
            - 'qvel': (T, 9) float32 array - robot joint velocities
            - 'actions': (T, 8) float32 array - actions (7 joints + 1 gripper)
            - 'tcp_pose': (T, 7) float32 array - TCP pose (position + quaternion)
            - 'success': (T,) bool array - success flag for each step

    Note:
        - Observations have T+1 frames (includes final state)
        - Actions have T frames
        - We align them by truncating observations to T frames
    """
    trajectories = []

    with h5py.File(h5_path, 'r') as f:
        # Get all trajectory keys (traj_0, traj_1, ...)
        traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')])
        print(f"Found {len(traj_keys)} trajectories in {h5_path}")

        for traj_key in traj_keys:
            traj = f[traj_key]

            # Extract data
            rgb_images = traj['obs/sensor_data/base_camera/rgb'][:]  # (T+1, 128, 128, 3) uint8
            depth_images = traj['obs/sensor_data/base_camera/depth'][:]  # (T+1, 128, 128, 1) int16
            qpos = traj['obs/agent/qpos'][:]  # (T+1, 9) float32
            qvel = traj['obs/agent/qvel'][:]  # (T+1, 9) float32
            tcp_pose = traj['obs/extra/tcp_pose'][:]  # (T+1, 7) float32
            actions = traj['actions'][:]  # (T, 8) float32
            success = traj['success'][:]  # (T,) bool

            # Time alignment: observations have T+1 frames, actions have T frames
            # We only keep the first T frames of observations
            num_steps = len(actions)

            trajectory_data = {
                'rgb_images': rgb_images[:num_steps],  # (T, 128, 128, 3)
                'depth_images': depth_images[:num_steps],  # (T, 128, 128, 1)
                'qpos': qpos[:num_steps],  # (T, 9)
                'qvel': qvel[:num_steps],  # (T, 9)
                'actions': actions,  # (T, 8)
                'tcp_pose': tcp_pose[:num_steps],  # (T, 7)
                'success': success,  # (T,)
            }

            trajectories.append(trajectory_data)

    return trajectories


def print_trajectory_info(trajectories: List[Dict[str, np.ndarray]]) -> None:
    """
    Print detailed information about loaded trajectories.

    Args:
        trajectories: List of trajectory dictionaries from load_maniskill_trajectory
    """
    print("\n" + "="*80)
    print("TRAJECTORY SUMMARY")
    print("="*80)

    print(f"\nTotal trajectories: {len(trajectories)}")

    # Calculate statistics
    total_steps = sum(len(traj['actions']) for traj in trajectories)
    avg_steps = total_steps / len(trajectories) if trajectories else 0
    min_steps = min(len(traj['actions']) for traj in trajectories) if trajectories else 0
    max_steps = max(len(traj['actions']) for traj in trajectories) if trajectories else 0
    success_rate = sum(traj['success'][-1] for traj in trajectories) / len(trajectories) if trajectories else 0

    print(f"Total steps: {total_steps}")
    print(f"Average steps per trajectory: {avg_steps:.1f}")
    print(f"Min steps: {min_steps}")
    print(f"Max steps: {max_steps}")
    print(f"Success rate: {success_rate*100:.1f}%")

    # Print first trajectory details
    if trajectories:
        print("\n" + "-"*80)
        print("FIRST TRAJECTORY DETAILS")
        print("-"*80)

        traj = trajectories[0]

        print("\nData shapes:")
        for key, value in traj.items():
            print(f"  {key:20s}: {str(value.shape):20s} dtype={value.dtype}")

        print("\nData ranges:")
        print(f"  RGB images:     min={traj['rgb_images'].min()}, max={traj['rgb_images'].max()}")
        print(f"  Depth images:   min={traj['depth_images'].min()}, max={traj['depth_images'].max()}")
        print(f"  qpos:           min={traj['qpos'].min():.4f}, max={traj['qpos'].max():.4f}")
        print(f"  qvel:           min={traj['qvel'].min():.4f}, max={traj['qvel'].max():.4f}")
        print(f"  actions:        min={traj['actions'].min():.4f}, max={traj['actions'].max():.4f}")
        print(f"  tcp_pose:       min={traj['tcp_pose'].min():.4f}, max={traj['tcp_pose'].max():.4f}")

        print("\nFirst step data sample:")
        print(f"  qpos[0]:     {traj['qpos'][0]}")
        print(f"  actions[0]:  {traj['actions'][0]}")
        print(f"  success[0]:  {traj['success'][0]}")

        print("\nLast step data sample:")
        print(f"  qpos[-1]:    {traj['qpos'][-1]}")
        print(f"  actions[-1]: {traj['actions'][-1]}")
        print(f"  success[-1]: {traj['success'][-1]}")

    print("\n" + "="*80)


def test_load(h5_path: str) -> None:
    """
    Test the load_maniskill_trajectory function.

    Args:
        h5_path: Path to the ManiSkill h5 file
    """
    print(f"Loading trajectories from: {h5_path}")

    try:
        # Load trajectories
        trajectories = load_maniskill_trajectory(h5_path)

        # Print information
        print_trajectory_info(trajectories)

        # Validate data format
        print("\n" + "="*80)
        print("VALIDATION CHECKS")
        print("="*80)

        all_valid = True

        for i, traj in enumerate(trajectories[:3]):  # Check first 3 trajectories
            print(f"\nTrajectory {i}:")

            # Check shapes
            T = len(traj['actions'])
            checks = {
                'rgb_images': (T, 128, 128, 3),
                'depth_images': (T, 128, 128, 1),
                'qpos': (T, 9),
                'qvel': (T, 9),
                'actions': (T, 8),
                'tcp_pose': (T, 7),
                'success': (T,),
            }

            for key, expected_shape in checks.items():
                actual_shape = traj[key].shape
                is_valid = actual_shape == expected_shape
                status = "✓" if is_valid else "✗"
                print(f"  {status} {key:20s}: {str(actual_shape):20s} (expected: {expected_shape})")
                all_valid = all_valid and is_valid

            # Check dtypes
            dtype_checks = {
                'rgb_images': np.uint8,
                'depth_images': np.int16,
                'qpos': np.float32,
                'qvel': np.float32,
                'actions': np.float32,
                'tcp_pose': np.float32,
                'success': bool,
            }

            for key, expected_dtype in dtype_checks.items():
                actual_dtype = traj[key].dtype
                is_valid = actual_dtype == expected_dtype
                status = "✓" if is_valid else "✗"
                if not is_valid:
                    print(f"  {status} {key:20s}: dtype={actual_dtype} (expected: {expected_dtype})")
                    all_valid = False

        print("\n" + "="*80)
        if all_valid:
            print("✓ ALL VALIDATION CHECKS PASSED!")
        else:
            print("✗ SOME VALIDATION CHECKS FAILED!")
        print("="*80)

    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    tyro.cli(test_load)
