import pathlib
import sys

try:
    print("=" * 60, flush=True)
    print("PyTorch 模型测试", flush=True)
    print("=" * 60, flush=True)
    print()

    print("Step 1: Importing modules...", flush=True)
    from openpi.training import config as _config

    print("  ✓ Config imported", flush=True)
    from openpi.policies import policy_config

    print("  ✓ Policy config imported", flush=True)
    print("  ✓ Download module imported", flush=True)
    from openpi.policies import droid_policy

    print("  ✓ Droid policy imported", flush=True)
    print()

    print("Step 2: Setting up paths...", flush=True)
    config_name = "pi05_droid"
    checkpoint_dir = "/18141169908/wyh/weights/pi05_droid_pytorch"

    # 检查检查点目录
    checkpoint_path = pathlib.Path(checkpoint_dir)
    print(f"  Checkpoint path: {checkpoint_path}", flush=True)

    if not checkpoint_path.exists():
        print(f"  ✗ ERROR: Checkpoint directory not found: {checkpoint_path}", flush=True)
        sys.exit(1)
    print("  ✓ Checkpoint directory exists", flush=True)

    # 检查是否有 model.safetensors
    model_file = checkpoint_path / "model.safetensors"
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ Found model.safetensors ({size_mb:.1f} MB)", flush=True)
    else:
        print(f"  ⚠ Warning: model.safetensors not found at {model_file}", flush=True)
        print("     Listing directory contents:", flush=True)
        for item in checkpoint_path.iterdir():
            print(f"       - {item.name}", flush=True)
    print()

    print("Step 3: Loading config...", flush=True)
    config = _config.get_config(config_name)
    print(f"  ✓ Config loaded: {config_name}", flush=True)
    print()

    print("Step 4: Creating policy (this may take several minutes)...", flush=True)
    print("  This will load the PyTorch model weights...", flush=True)
    import time

    start_time = time.time()

    policy = policy_config.create_trained_policy(config, checkpoint_dir)

    elapsed = time.time() - start_time
    print(f"  ✓ Policy created successfully! (took {elapsed:.1f} seconds)", flush=True)
    print()

    print("Step 5: Testing inference...", flush=True)
    # 创建一个简单的测试示例（根据 pi0_droid 的输入格式）
    example = droid_policy.make_droid_example()
    result = policy.infer(example)
    print(result)
    print("action shape: ", result["actions"].shape)
    # 注意：这里需要根据实际的输入格式调整
    # print("  Note: Creating a dummy example for testing...", flush=True)
    # print("  (You may need to adjust this based on your actual input format)", flush=True)

    # 暂时跳过推理测试,因为需要正确的输入格式
    # print("  ⚠ Skipping inference test (example not defined)", flush=True)
    # print("  To test inference, you need to provide proper input format", flush=True)

    print()
    print("=" * 60, flush=True)
    print("✅ All steps completed successfully!", flush=True)
    print("=" * 60, flush=True)

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"\n✗ ERROR: {e}", flush=True)
    import traceback

    traceback.print_exc()
    sys.exit(1)
