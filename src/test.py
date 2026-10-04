import pathlib
import sys

try:
    print("Step 1: Importing modules...", flush=True)
    from openpi.training import config

    print("Step 2: Imported config", flush=True)
    from openpi.policies import policy_config

    print("Step 3: Imported policy_config", flush=True)
    from openpi.shared import download

    print("Step 3.5: Imported download module", flush=True)

    model_name = "pi05_droid"

    # Use custom local path - you can change this to your actual path
    print("Step 4: Setting up checkpoint path...", flush=True)
    # Option 1: Use custom path directly
    custom_path = "/18141169908/wyh/weights/pi05_droid"
    local_checkpoint = pathlib.Path(custom_path).resolve()

    # Option 2: Or use environment variable (uncomment to use)
    # local_checkpoint = pathlib.Path(os.getenv("CHECKPOINT_PATH", custom_path)).resolve()

    # Option 3: Or use default cache location (uncomment to use)
    # cache_dir = pathlib.Path(os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    # local_checkpoint = cache_dir / "openpi-assets" / "checkpoints" / model_name
    # local_checkpoint = local_checkpoint.resolve()

    print(f"Step 5: Checking checkpoint at: {local_checkpoint}", flush=True)
    if local_checkpoint.exists():
        print(f"✓ Checkpoint found at: {local_checkpoint}", flush=True)
        # Verify it has the required subdirectories
        params_dir = local_checkpoint / "params"
        if params_dir.exists():
            print("✓ Params directory found", flush=True)
            print(f"  Params dir contents: {list(params_dir.iterdir())[:5]}...", flush=True)
        else:
            print(f"✗ ERROR: params directory not found at {params_dir}", flush=True)
            sys.exit(1)
        # Convert to string for maybe_download
        checkpoint_dir = str(local_checkpoint)
    else:
        print(f"✗ ERROR: Checkpoint not found at {local_checkpoint}", flush=True)
        print(f"Please run: bash scripts/download_checkpoint.sh {model_name}", flush=True)
        sys.exit(1)

    print("Step 6: Testing maybe_download with local path...", flush=True)
    # Test that maybe_download recognizes it as a local path
    test_result = download.maybe_download(checkpoint_dir)
    print(f"Step 6.5: maybe_download returned: {test_result}", flush=True)

    print("Step 7: Getting config...", flush=True)
    train_config = config.get_config(model_name)
    print("Step 8: Config loaded", flush=True)

    print("Step 9: Creating policy (this may take a while, loading model weights)...", flush=True)
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
    print("Step 10: Policy created successfully!", flush=True)
    print("\n✅ All steps completed successfully!", flush=True)

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"\n✗ ERROR at step above: {e}", flush=True)
    import traceback

    traceback.print_exc()
    sys.exit(1)
