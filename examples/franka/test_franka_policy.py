"""
Test the Franka policy input/output transforms.

Usage:
    uv run examples/franka/test_franka_policy.py
"""

import numpy as np
from openpi.policies import franka_policy
from openpi.models import model as _model


def test_franka_inputs():
    """Test FrankaInputs transform."""
    print("="*80)
    print("TESTING FrankaInputs")
    print("="*80)

    # Create test data in LeRobot format
    test_data = {
        "image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "state": np.random.rand(8).astype(np.float32),
        "actions": np.random.rand(8).astype(np.float32),
        "prompt": "pick up the cube and place it at the goal position",
    }

    print("\nInput data:")
    for key, value in test_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:20s}: shape={str(value.shape):20s} dtype={value.dtype}")
        else:
            print(f"  {key:20s}: {value}")

    # Test with different model types
    for model_type in [_model.ModelType.PI0, _model.ModelType.PI0_FAST, _model.ModelType.PI05]:
        print(f"\n{'-'*80}")
        print(f"Testing with model_type={model_type.name}")
        print("-"*80)

        # Create transform
        transform = franka_policy.FrankaInputs(model_type=model_type)

        # Apply transform
        output = transform(test_data)

        print("\nOutput structure:")
        print(f"  Keys: {list(output.keys())}")

        print("\nOutput details:")
        print(f"  state: {output['state'].shape} {output['state'].dtype}")

        print(f"\n  image:")
        for cam_name, img in output['image'].items():
            print(f"    {cam_name:20s}: {img.shape} {img.dtype}")

        print(f"\n  image_mask:")
        for cam_name, mask in output['image_mask'].items():
            print(f"    {cam_name:20s}: {mask}")

        if "actions" in output:
            print(f"\n  actions: {output['actions'].shape} {output['actions'].dtype}")

        if "prompt" in output:
            print(f"\n  prompt: {output['prompt']}")

        # Validation checks
        print("\n  Validation:")
        assert output['state'].shape == (8,), f"Expected state shape (8,), got {output['state'].shape}"
        assert output['image']['base_0_rgb'].shape == (128, 128, 3), "Base camera shape mismatch"
        assert output['image']['left_wrist_0_rgb'].shape == (128, 128, 3), "Left wrist camera shape mismatch"
        assert output['image']['right_wrist_0_rgb'].shape == (128, 128, 3), "Right wrist camera shape mismatch"
        assert output['image_mask']['base_0_rgb'] == True, "Base camera mask should be True"

        if model_type == _model.ModelType.PI0_FAST:
            assert output['image_mask']['left_wrist_0_rgb'] == True, "PI0_FAST should have True masks"
            assert output['image_mask']['right_wrist_0_rgb'] == True, "PI0_FAST should have True masks"
        else:
            assert output['image_mask']['left_wrist_0_rgb'] == False, "PI0/PI05 should have False masks for padding"
            assert output['image_mask']['right_wrist_0_rgb'] == False, "PI0/PI05 should have False masks for padding"

        print("  ✓ All validations passed!")


def test_franka_outputs():
    """Test FrankaOutputs transform."""
    print("\n" + "="*80)
    print("TESTING FrankaOutputs")
    print("="*80)

    # Create test data (model output format)
    # Pi0 outputs 32-dim actions with action_horizon=50
    test_data = {
        "actions": np.random.rand(50, 32).astype(np.float32),
    }

    print("\nInput data:")
    print(f"  actions: shape={test_data['actions'].shape} dtype={test_data['actions'].dtype}")

    # Create transform
    transform = franka_policy.FrankaOutputs()

    # Apply transform
    output = transform(test_data)

    print("\nOutput data:")
    print(f"  actions: shape={output['actions'].shape} dtype={output['actions'].dtype}")

    # Validation
    print("\nValidation:")
    assert output['actions'].shape == (50, 8), f"Expected actions shape (50, 8), got {output['actions'].shape}"
    assert np.allclose(output['actions'], test_data['actions'][:, :8]), "Actions should match first 8 dims"
    print("  ✓ All validations passed!")


def test_make_example():
    """Test make_franka_example function."""
    print("\n" + "="*80)
    print("TESTING make_franka_example()")
    print("="*80)

    example = franka_policy.make_franka_example()

    print("\nGenerated example:")
    for key, value in example.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:20s}: shape={str(value.shape):20s} dtype={value.dtype}")
        else:
            print(f"  {key:20s}: {value}")

    # Validation
    print("\nValidation:")
    assert example['image'].shape == (128, 128, 3), "Image shape mismatch"
    assert example['image'].dtype == np.uint8, "Image dtype should be uint8"
    assert example['state'].shape == (8,), "State shape mismatch"
    assert isinstance(example['prompt'], str), "Prompt should be a string"
    print("  ✓ All validations passed!")


def test_image_parsing():
    """Test _parse_image function with different input formats."""
    print("\n" + "="*80)
    print("TESTING _parse_image()")
    print("="*80)

    # Test 1: uint8 HWC format (no conversion needed)
    print("\nTest 1: uint8 HWC format")
    img_hwc = np.random.randint(256, size=(128, 128, 3), dtype=np.uint8)
    parsed = franka_policy._parse_image(img_hwc)
    assert parsed.shape == (128, 128, 3), "Shape should remain HWC"
    assert parsed.dtype == np.uint8, "Dtype should remain uint8"
    assert np.array_equal(parsed, img_hwc), "Image should be unchanged"
    print(f"  Input:  shape={img_hwc.shape} dtype={img_hwc.dtype}")
    print(f"  Output: shape={parsed.shape} dtype={parsed.dtype}")
    print("  ✓ Passed")

    # Test 2: float32 HWC format (convert to uint8)
    print("\nTest 2: float32 HWC format")
    img_float_hwc = np.random.rand(128, 128, 3).astype(np.float32)
    parsed = franka_policy._parse_image(img_float_hwc)
    assert parsed.shape == (128, 128, 3), "Shape should remain HWC"
    assert parsed.dtype == np.uint8, "Dtype should be converted to uint8"
    print(f"  Input:  shape={img_float_hwc.shape} dtype={img_float_hwc.dtype}")
    print(f"  Output: shape={parsed.shape} dtype={parsed.dtype}")
    print("  ✓ Passed")

    # Test 3: uint8 CHW format (rearrange to HWC)
    print("\nTest 3: uint8 CHW format")
    img_chw = np.random.randint(256, size=(3, 128, 128), dtype=np.uint8)
    parsed = franka_policy._parse_image(img_chw)
    assert parsed.shape == (128, 128, 3), "Shape should be converted to HWC"
    assert parsed.dtype == np.uint8, "Dtype should remain uint8"
    print(f"  Input:  shape={img_chw.shape} dtype={img_chw.dtype}")
    print(f"  Output: shape={parsed.shape} dtype={parsed.dtype}")
    print("  ✓ Passed")

    # Test 4: float32 CHW format (convert and rearrange)
    print("\nTest 4: float32 CHW format")
    img_float_chw = np.random.rand(3, 128, 128).astype(np.float32)
    parsed = franka_policy._parse_image(img_float_chw)
    assert parsed.shape == (128, 128, 3), "Shape should be converted to HWC"
    assert parsed.dtype == np.uint8, "Dtype should be converted to uint8"
    print(f"  Input:  shape={img_float_chw.shape} dtype={img_float_chw.dtype}")
    print(f"  Output: shape={parsed.shape} dtype={parsed.dtype}")
    print("  ✓ Passed")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" FRANKA POLICY TEST SUITE")
    print("="*80 + "\n")

    try:
        test_franka_inputs()
        test_franka_outputs()
        test_make_example()
        test_image_parsing()

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
