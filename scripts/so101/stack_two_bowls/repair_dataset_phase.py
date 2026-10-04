"""
General utility to repair the 'phase' column in a LeRobot v2.1 dataset.

Fixes two common issues that arise when a set_phases_xxx.py script adds a
phase column to parquet files without properly syncing the dataset metadata:

  1. meta/info.json does not declare the 'phase' feature
     → LeRobot's .cast(features) raises "columns must be identical" ValueError.

  2. The 'phase' column is stored as list<float> instead of a float32 scalar
     → LeRobot's cast raises "Couldn't cast list<item: float> to float" TypeError.

Usage:
    # Check what would be changed (no files are modified):
    python scripts/so101/repair_dataset_phase.py --dataset_dir dataset/realworld/stackbowls_v2 --dry-run

    # Apply fixes:
    python scripts/so101/repair_dataset_phase.py --dataset_dir dataset/realworld/stackbowls_v2
"""

import argparse
import glob
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_parquet_files(dataset_dir: str) -> list[str]:
    pattern = os.path.join(dataset_dir, "data", "**", "*.parquet")
    return sorted(glob.glob(pattern, recursive=True))


def _check_phase_column(table) -> dict:
    """Return a dict describing the current state of the phase column."""
    if "phase" not in table.column_names:
        return {"present": False}

    pa_type = table.schema.field("phase").type
    is_list = pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type)
    return {
        "present": True,
        "is_list": is_list,
        "pa_type": str(pa_type),
    }


def repair_parquet(path: str, dry_run: bool) -> str:
    """
    Convert the phase column in one parquet file from list<float> to float32
    scalar if needed.  Returns a short status string.
    """
    table = pq.read_table(path)
    info = _check_phase_column(table)

    if not info["present"]:
        return "SKIP  (no phase column)"

    if not info["is_list"]:
        return "OK    (already scalar: {})".format(info["pa_type"])

    # Need to convert list → scalar
    phase_list = table.column("phase")
    phase_scalar = pa.array(
        [row.as_py()[0] if isinstance(row.as_py(), list) else float(row.as_py())
         for row in phase_list],
        type=pa.float32(),
    )

    if not dry_run:
        table = table.drop(["phase"])
        table = table.append_column("phase", phase_scalar)
        pq.write_table(table, path)

    return "FIXED (was {})".format(info["pa_type"])


def repair_info_json(dataset_dir: str, dry_run: bool) -> str:
    """
    Add the phase feature declaration to meta/info.json if missing.
    Returns a short status string.
    """
    info_path = os.path.join(dataset_dir, "meta", "info.json")
    if not os.path.exists(info_path):
        return "SKIP  (info.json not found at {})".format(info_path)

    with open(info_path, "r") as f:
        info = json.load(f)

    if "phase" in info.get("features", {}):
        current = info["features"]["phase"]
        return "OK    (already declared: dtype={}, shape={})".format(
            current.get("dtype"), current.get("shape")
        )

    if not dry_run:
        info.setdefault("features", {})["phase"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

    return "FIXED (added phase: dtype=float32, shape=[1])"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Repair 'phase' column and info.json in a LeRobot v2.1 dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the LeRobot dataset root (contains data/ and meta/ folders).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without modifying any files.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    dry_run = args.dry_run

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError("Dataset directory not found: {}".format(dataset_dir))

    print("Dataset : {}".format(dataset_dir))
    print("Mode    : {}\n".format("DRY RUN (no files modified)" if dry_run else "APPLY"))

    # 1. Fix meta/info.json
    print("── meta/info.json ──────────────────────────────────────────")
    status = repair_info_json(dataset_dir, dry_run)
    print("  {}".format(status))
    print()

    # 2. Fix parquet files
    parquet_files = _find_parquet_files(dataset_dir)
    if not parquet_files:
        print("No parquet files found under {}/data/".format(dataset_dir))
        return

    print("── parquet files ({} total) ─────────────────────────────────".format(
        len(parquet_files)
    ))
    fixed = skipped = already_ok = 0
    for path in parquet_files:
        status = repair_parquet(path, dry_run)
        label = os.path.relpath(path, dataset_dir)
        print("  {}  {}".format(status, label))
        if status.startswith("FIXED"):
            fixed += 1
        elif status.startswith("SKIP"):
            skipped += 1
        else:
            already_ok += 1

    print()
    print("── Summary ─────────────────────────────────────────────────")
    print("  Fixed : {}".format(fixed))
    print("  OK    : {}".format(already_ok))
    print("  Skip  : {}".format(skipped))
    if dry_run and fixed > 0:
        print("\nRe-run without --dry-run to apply the fixes.")


if __name__ == "__main__":
    main()
