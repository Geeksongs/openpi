"""
Policy mapping for Franka ManiSkill dataset.

This module defines the input/output transforms for the Franka robot with ManiSkill
demonstration data, enabling training and inference with pi0 models.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "state": np.random.rand(8),
        "prompt": "pick up the cube and place it at the goal position",
    }


def _parse_image(image) -> np.ndarray:
    """
    Parse image to uint8 (H, W, C) format.

    This handles:
    - float32 -> uint8 conversion (LeRobot stores as float32)
    - CHW -> HWC rearrangement (LeRobot stores as CHW)

    Args:
        image: Input image array

    Returns:
        Parsed image in (H, W, C) uint8 format
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """
    Maps Franka ManiSkill LeRobot dataset format to pi0 model input format.

    Used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on
    the comments below to pipe the correct elements of your dataset into the model.
    """

    # Determines which model will be used (pi0, pi0-FAST, or pi05)
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        """
        Transform LeRobot data to pi0 model format.

        Args:
            data: Dictionary containing:
                - "image": (128, 128, 3) or (3, 128, 128) image
                - "state": (8,) robot state
                - "actions": (8,) actions (only during training)
                - "prompt": str language instruction (optional)

        Returns:
            Dictionary with pi0 model format:
                - "image": dict of camera images
                - "image_mask": dict of camera masks
                - "state": (8,) robot state
                - "actions": (8,) actions (if present)
                - "prompt": str language instruction (if present)
        """
        # Parse image to uint8 (H, W, C) format
        # LeRobot automatically stores as float32 (C, H, W), so we need to convert
        # This step is skipped for policy inference since images are already in correct format
        base_image = _parse_image(data["image"])

        # Pi0 models expect 3 cameras:
        # - base_0_rgb: third-person view (we have this)
        # - left_wrist_0_rgb: left wrist view (we don't have, use padding)
        # - right_wrist_0_rgb: right wrist view (we don't have, use padding)
        #
        # If your dataset does not have a particular type of image (e.g. wrist images),
        # you can replace it with zeros like we do below.
        inputs = {
            "state": data["state"],  # (8,) - 7 joints + 1 gripper
            "image": {
                "base_0_rgb": base_image,  # Real camera image
                # Pad any non-existent images with zero-arrays of the appropriate shape
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,  # Real image - always True
                # We only mask padding images for pi0 model, not pi0-FAST
                # Do not change this for your own dataset
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """
    Maps pi0 model outputs back to Franka action format.

    Used for inference only.

    For your own dataset, you can copy this class and modify the action dimension
    based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        """
        Extract Franka-specific actions from pi0 model output.

        Args:
            data: Dictionary containing:
                - "actions": (action_horizon, 32) model output

        Returns:
            Dictionary containing:
                - "actions": (action_horizon, 8) Franka actions
        """
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Franka ManiSkill, we only return the first 8 actions (since the rest is padding).
        # For your own dataset, replace `8` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :8])}
