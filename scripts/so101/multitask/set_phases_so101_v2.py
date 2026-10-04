"""
Phase labeling for SO101 bi-arm real-world datasets (LeRobot v2.1 parquet format).

5-phase schema (mirrors scripts/set_atom/macg/two_arms_stack_bowls/ simulator reference):
  Phase 1: first-arm approach/pick     (start → first-arm gripper closes)
  Phase 2: first-arm place/push        (first-arm closed → first-arm releases)
  Phase 3: second-arm approach/pick    (first-arm released → second-arm gripper closes)
  Phase 4: second-arm place/push       (second-arm closed → second-arm releases)
  Phase 5: recover                     (second-arm released → episode end)

Gripper thresholds are adaptive per episode per arm to handle SO100 calibration drift.
Arm order (which closes first) is auto-detected per episode.

The script writes a NEW dataset directory (--output), keeping the source intact.
It also expands tasks.jsonl to 5 per-phase dual-arm atomic instruction entries and
rewrites task_index per frame accordingly.

Usage:
    # Dry-run with visualisation
    python scripts/so101/multitask/set_phases_so101_v2.py \\
        --source dataset/macg_so101/macg_stack_bowls_v2 \\
        --output dataset/macg_so101/macg_stack_bowls_v2_phased \\
        --atoms "pick=Pick up the bowl.,action=Stack the bowl.,recover=Recover." \\
        --dry-run --visualize

    # Write for real
    python scripts/so101/multitask/set_phases_so101_v2.py \\
        --source dataset/macg_so101/macg_stack_bowls_v2 \\
        --output dataset/macg_so101/macg_stack_bowls_v2_phased \\
        --atoms "pick=Pick up the bowl.,action=Stack the bowl.,recover=Recover."
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Gripper column indices in the 12-D state/action vector
# ---------------------------------------------------------------------------
LEFT_GRIPPER_IDX = 5
RIGHT_GRIPPER_IDX = 11

# ---------------------------------------------------------------------------
# Default atomic instruction tables keyed by source dataset basename
# ---------------------------------------------------------------------------
DEFAULT_ATOMS: dict[str, dict[str, str]] = {
    "macg_push_cubes_v2": {
        "pick": "Grasp the cube.",
        "action": "Push the cube.",
        "recover": "Recover.",
        "idle": "Wait for the other robots.",
    },
    "macg_stack_bowls_v2": {
        "pick": "Pick up the bowl.",
        "action": "Stack the bowl.",
        "recover": "Recover.",
        "idle": "Wait for the other robots.",
    },
    "macg_stack_cubes_v2": {
        "pick": "Pick up the cube.",
        "action": "Stack the cube.",
        "recover": "Recover.",
        "idle": "Wait for the other robots.",
    },
}

IDLE = "Wait for the other robots."
RECOVER = "Recover."

SMOOTH_WINDOW = 5
HYSTERESIS_FRAC = 0.15
OPEN_PERCENTILE = 90
CLOSE_PERCENTILE = 10


# ---------------------------------------------------------------------------
# Gripper helpers
# ---------------------------------------------------------------------------

def smooth(x: np.ndarray, w: int = SMOOTH_WINDOW) -> np.ndarray:
    if w <= 1:
        return x.copy()
    return np.convolve(x, np.ones(w) / w, mode="same")


def adaptive_thresholds(gripper: np.ndarray) -> tuple[float, float]:
    """Return (close_thresh, open_thresh) adapted to this arm's range."""
    open_val = float(np.percentile(gripper, OPEN_PERCENTILE))
    close_val = float(np.percentile(gripper, CLOSE_PERCENTILE))
    span = open_val - close_val
    if span < 1.0:
        # Degenerate gripper trace — arm never moved significantly
        mid = (open_val + close_val) / 2.0
        return mid - 0.5, mid + 0.5
    close_thresh = close_val + HYSTERESIS_FRAC * span
    open_thresh = open_val - HYSTERESIS_FRAC * span
    return close_thresh, open_thresh


def find_close_open_events(
    gripper: np.ndarray,
    close_thresh: float,
    open_thresh: float,
) -> tuple[int | None, int | None]:
    """
    Returns (first_close_frame, first_open_after_close_frame).

    first_close: gripper transitions from above close_thresh to <= close_thresh.
      We use close_thresh (not full open_thresh) as the "was not closed" baseline so
      episodes that start at a partial-open position (e.g. 65 when open_thresh is 70)
      are still handled correctly.

    first_open_after_close: after first_close, gripper rises above mid_thresh =
      (close_thresh + open_thresh) / 2.  This handles real SO100 arms that recover to a
      partial-open position (e.g. 68) rather than back to the full-open value (~90).
    """
    n = len(gripper)
    # mid threshold for reopen detection (more permissive than open_thresh)
    mid_thresh = (close_thresh + open_thresh) / 2.0

    first_close: int | None = None
    first_open: int | None = None

    # Look for first drop from "above close_thresh" region to "at/below close_thresh"
    was_above_close = gripper[0] > close_thresh
    for i in range(1, n):
        if was_above_close and gripper[i] <= close_thresh:
            first_close = i
            break
        if gripper[i] > close_thresh:
            was_above_close = True

    if first_close is None:
        return None, None

    # After first_close, look for rise above mid_thresh
    was_below_mid = True
    for i in range(first_close + 1, n):
        if was_below_mid and gripper[i] >= mid_thresh:
            first_open = i
            break
        if gripper[i] < mid_thresh:
            was_below_mid = True

    return first_close, first_open


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

def detect_phases(
    left_g: np.ndarray,
    right_g: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Returns (phases array of shape [T], debug dict).

    debug keys:
      left_close_thresh, left_open_thresh,
      right_close_thresh, right_open_thresh,
      left_first_close, left_first_open,
      right_first_close, right_first_open,
      first_arm ("left" | "right"),
      boundaries: list[int] of 4 boundary frames [p1->p2, p2->p3, p3->p4, p4->p5],
      warning: str | None,
    """
    n = len(left_g)
    debug: dict[str, Any] = {"warning": None}

    left_s = smooth(left_g)
    right_s = smooth(right_g)

    left_ct, left_ot = adaptive_thresholds(left_s)
    right_ct, right_ot = adaptive_thresholds(right_s)
    debug.update(
        left_close_thresh=left_ct,
        left_open_thresh=left_ot,
        right_close_thresh=right_ct,
        right_open_thresh=right_ot,
    )

    left_fc, left_fo = find_close_open_events(left_s, left_ct, left_ot)
    right_fc, right_fo = find_close_open_events(right_s, right_ct, right_ot)
    debug.update(
        left_first_close=left_fc,
        left_first_open=left_fo,
        right_first_close=right_fc,
        right_first_open=right_fo,
    )

    # Validate that both arms have a clean close+open pair
    missing = []
    if left_fc is None:
        missing.append("left_close")
    if left_fo is None:
        missing.append("left_open")
    if right_fc is None:
        missing.append("right_close")
    if right_fo is None:
        missing.append("right_open")

    if missing:
        debug["warning"] = "Missing gripper events: " + ", ".join(missing)
        phases = np.ones(n, dtype=np.int32)
        debug["boundaries"] = []
        debug["first_arm"] = "unknown"
        return phases, debug

    # Determine arm order by which closes first
    if left_fc <= right_fc:
        first_arm = "left"
        b1 = left_fc    # p1 → p2: first arm closes
        b2 = left_fo    # p2 → p3: first arm opens
        b3 = right_fc   # p3 → p4: second arm closes
        b4 = right_fo   # p4 → p5: second arm opens
    else:
        first_arm = "right"
        b1 = right_fc
        b2 = right_fo
        b3 = left_fc
        b4 = left_fo

    debug["first_arm"] = first_arm

    # Clamp boundaries to valid range and ensure monotonicity
    boundaries = []
    prev = 0
    for b in [b1, b2, b3, b4]:
        b = max(prev + 1, min(b, n - 1))
        boundaries.append(b)
        prev = b
    debug["boundaries"] = boundaries
    b1, b2, b3, b4 = boundaries

    phases = np.ones(n, dtype=np.int32)
    phases[b1:b2] = 2
    phases[b2:b3] = 3
    phases[b3:b4] = 4
    phases[b4:] = 5

    return phases, debug


# ---------------------------------------------------------------------------
# Task string generation
# ---------------------------------------------------------------------------

def build_task_strings(
    atoms: dict[str, str],
    arm_order: str,  # "left_first" or "right_first"
) -> dict[int, str]:
    """
    Build {phase_number: "left_atom | right_atom"} for phases 1-5.

    If arm_order is "left_first":
      P1: left picks      → left=pick,   right=idle
      P2: left places     → left=action, right=idle
      P3: right picks     → left=idle,   right=pick
      P4: right places    → left=idle,   right=action
      P5: both recover    → left=recover, right=recover
    Swap left/right if right_first.
    """
    pick = atoms["pick"]
    action = atoms["action"]
    recover = atoms.get("recover", RECOVER)
    idle = atoms.get("idle", IDLE)

    if arm_order == "left_first":
        return {
            1: f"{pick} | {idle}",
            2: f"{action} | {idle}",
            3: f"{idle} | {pick}",
            4: f"{idle} | {action}",
            5: f"{recover} | {recover}",
        }
    else:  # right_first
        return {
            1: f"{idle} | {pick}",
            2: f"{idle} | {action}",
            3: f"{pick} | {idle}",
            4: f"{action} | {idle}",
            5: f"{recover} | {recover}",
        }


def arm_order_key(first_arm: str) -> str:
    return "left_first" if first_arm == "left" else "right_first"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(path: Path, items: list[dict]) -> None:
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def load_episode(data_dir: Path, ep_id: int) -> pa.Table:
    path = data_dir / f"episode_{ep_id:06d}.parquet"
    return pq.read_table(path)


def write_episode(data_dir: Path, ep_id: int, table: pa.Table) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"episode_{ep_id:06d}.parquet"
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_dataset(
    source: Path,
    output: Path,
    atoms: dict[str, str],
    dry_run: bool = False,
    visualize: bool = False,
    episodes_filter: list[int] | None = None,
) -> dict:
    """
    Phase-label one dataset.

    Returns summary dict with keys: total, written, skipped, skipped_list, arm_order_counts.
    """
    src_data = source / "data" / "chunk-000"
    src_meta = source / "meta"

    out_data = output / "data" / "chunk-000"
    out_meta = output / "meta"
    vis_dir = output / "phase_plots" if visualize else None

    # Collect episode IDs
    parquet_files = sorted(src_data.glob("episode_*.parquet"))
    all_ep_ids = [int(p.stem.split("_")[1]) for p in parquet_files]
    if episodes_filter is not None:
        all_ep_ids = [e for e in all_ep_ids if e in episodes_filter]

    src_info = json.loads((src_meta / "info.json").read_text())
    src_episodes = load_jsonl(src_meta / "episodes.jsonl")
    ep_stats_path = src_meta / "episodes_stats.jsonl"
    src_ep_stats = load_jsonl(ep_stats_path) if ep_stats_path.exists() else []

    # Determine the atoms dictionary
    atoms_for_episode: dict[int, dict[str, str]] = {}
    arm_order_per_episode: dict[int, str] = {}

    # ---- First pass: detect phases for all episodes ----
    results: dict[int, tuple[np.ndarray, dict]] = {}
    skipped: list[dict] = []

    print(f"Processing {len(all_ep_ids)} episodes (dry_run={dry_run})...")
    for ep_id in all_ep_ids:
        table = load_episode(src_data, ep_id)
        n = table.num_rows
        states = np.array([np.array(table.column("state")[i].as_py()) for i in range(n)])
        left_g = states[:, LEFT_GRIPPER_IDX]
        right_g = states[:, RIGHT_GRIPPER_IDX]

        phases, debug = detect_phases(left_g, right_g)

        if debug["warning"]:
            print(f"  ep {ep_id:3d} | WARNING: {debug['warning']}")
            skipped.append({"episode_index": ep_id, "reason": debug["warning"]})
        else:
            b = debug["boundaries"]
            cnt = {p: int((phases == p).sum()) for p in range(1, 6)}
            print(
                f"  ep {ep_id:3d} | n={n:4d} | first={debug['first_arm']}"
                f" | b={[int(x) for x in b]}"
                f" | P1:{cnt[1]} P2:{cnt[2]} P3:{cnt[3]} P4:{cnt[4]} P5:{cnt[5]}"
            )

        results[ep_id] = (phases, debug)

    # ---- Determine arm order consensus per dataset ----
    # Build frequency map to detect if any episode flips
    arm_order_counts: dict[str, int] = {"left_first": 0, "right_first": 0, "unknown": 0}
    for ep_id, (phases, debug) in results.items():
        if ep_id in [s["episode_index"] for s in skipped]:
            continue
        ao = arm_order_key(debug["first_arm"])
        arm_order_counts[ao] += 1
        arm_order_per_episode[ep_id] = ao

    print(f"\nArm order counts: {arm_order_counts}")

    # Build task-string sets for each possible arm order
    task_sets: dict[str, dict[int, str]] = {}
    for ao in ["left_first", "right_first"]:
        task_sets[ao] = build_task_strings(atoms, ao)

    # Build merged tasks.jsonl: up to 10 entries (5 per arm order if mixed)
    # task_index mapping: arm_order -> phase -> global task index
    used_arm_orders = [ao for ao in ["left_first", "right_first"] if arm_order_counts[ao] > 0]
    task_entries: list[dict] = []
    task_index_map: dict[str, dict[int, int]] = {}  # arm_order -> phase -> task_index
    ti = 0
    for ao in used_arm_orders:
        task_index_map[ao] = {}
        for phase in range(1, 6):
            task_str = task_sets[ao][phase]
            task_entries.append({"task_index": ti, "task": task_str})
            task_index_map[ao][phase] = ti
            ti += 1

    if dry_run:
        print("\n[DRY RUN] task_index_map:")
        for ao, pm in task_index_map.items():
            for phase, idx in pm.items():
                print(f"  {ao} P{phase} -> task_index={idx}: {task_sets[ao][phase]}")
        # Still generate visualizations if requested
        if visualize:
            _visualize_all(vis_dir, src_data, results, all_ep_ids, skipped)
        return {
            "total": len(all_ep_ids),
            "written": 0,
            "skipped": len(skipped),
            "skipped_list": skipped,
            "arm_order_counts": arm_order_counts,
        }

    # ---- Setup output dirs ----
    out_data.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    # Copy non-parquet meta files (videos etc) — skip, we only copy meta/
    # Copy all source files we don't rewrite
    for fname in ["episodes_stats.jsonl"]:
        src_f = src_meta / fname
        if src_f.exists():
            shutil.copy2(src_f, out_meta / fname)

    written = 0
    skipped_ep_ids = {s["episode_index"] for s in skipped}

    new_episodes: list[dict] = []

    for ep_id in all_ep_ids:
        if ep_id in skipped_ep_ids:
            print(f"  ep {ep_id:3d} | SKIPPING (no valid phase detection)")
            continue

        phases, debug = results[ep_id]
        ao = arm_order_per_episode[ep_id]
        phase_to_ti = task_index_map[ao]

        # Build per-frame task_index array
        task_index_arr = np.array([phase_to_ti[int(p)] for p in phases], dtype=np.int64)
        phase_arr = phases.astype(np.int32)

        # Read source parquet
        table = load_episode(src_data, ep_id)

        # Drop existing phase/task_index if present (idempotent re-runs)
        cols_to_drop = [c for c in ["phase", "task_index"] if c in table.column_names]
        if cols_to_drop:
            table = table.drop(cols_to_drop)

        # Append new columns
        table = table.append_column(
            "phase", pa.array(phase_arr.tolist(), type=pa.float32())
        )
        table = table.append_column(
            "task_index", pa.array(task_index_arr.tolist(), type=pa.int64())
        )

        write_episode(out_data, ep_id, table)

        # Build episode entry for episodes.jsonl
        # tasks list: unique task strings appearing in this episode
        episode_task_indices = sorted(set(task_index_arr.tolist()))
        episode_task_strings = [task_entries[ti]["task"] for ti in episode_task_indices]
        new_episodes.append({
            "episode_index": ep_id,
            "tasks": episode_task_strings,
            "length": len(phases),
        })

        written += 1

    # ---- Write updated meta ----
    # info.json: add phase feature, update total_tasks
    new_info = dict(src_info)
    features = dict(new_info.get("features", {}))
    features["phase"] = {"dtype": "float32", "shape": [1], "names": None}
    new_info["features"] = features
    new_info["total_tasks"] = len(task_entries)
    new_info["total_episodes"] = written

    # Recount total_frames from written episodes
    total_frames = sum(e["length"] for e in new_episodes)
    new_info["total_frames"] = total_frames
    new_info["splits"] = {"train": f"0:{written}"}

    (out_meta / "info.json").write_text(json.dumps(new_info, indent=4) + "\n")
    save_jsonl(out_meta / "tasks.jsonl", task_entries)
    save_jsonl(out_meta / "episodes.jsonl", new_episodes)

    # Copy episodes_stats.jsonl (stats are still valid for the original columns)
    if src_ep_stats:
        filtered_stats = [s for s in src_ep_stats if s["episode_index"] not in skipped_ep_ids]
        save_jsonl(out_meta / "episodes_stats.jsonl", filtered_stats)

    # Write skipped log
    if skipped:
        save_jsonl(output / "skipped.jsonl", skipped)
        print(f"\nWrote {len(skipped)} skipped episodes to {output}/skipped.jsonl")

    if visualize:
        _visualize_all(vis_dir, src_data, results, all_ep_ids, skipped)

    print(f"\nDone. Written: {written}, Skipped: {len(skipped)}")
    print(f"Output: {output}")
    print(f"\nTask strings ({len(task_entries)} total):")
    for te in task_entries:
        print(f"  [{te['task_index']}] {te['task']}")

    return {
        "total": len(all_ep_ids),
        "written": written,
        "skipped": len(skipped),
        "skipped_list": skipped,
        "arm_order_counts": arm_order_counts,
    }


# ---------------------------------------------------------------------------
# Visualization helper (called from process_dataset)
# ---------------------------------------------------------------------------

def _visualize_all(
    vis_dir: Path | None,
    src_data: Path,
    results: dict[int, tuple[np.ndarray, dict]],
    all_ep_ids: list[int],
    skipped: list[dict],
) -> None:
    """Generate phase plots for all episodes (imported lazily to avoid hard dep)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if vis_dir is None:
        return
    vis_dir.mkdir(parents=True, exist_ok=True)
    skipped_ids = {s["episode_index"] for s in skipped}

    for ep_id in all_ep_ids:
        phases, debug = results[ep_id]

        table = load_episode(src_data, ep_id)
        n = table.num_rows
        states = np.array([np.array(table.column("state")[i].as_py()) for i in range(n)])
        left_g = states[:, LEFT_GRIPPER_IDX]
        right_g = states[:, RIGHT_GRIPPER_IDX]

        t = np.arange(n)
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        title_warn = ""
        if ep_id in skipped_ids:
            title_warn = " [SKIPPED]"
        fig.suptitle(
            f"Episode {ep_id}{title_warn} | first_arm={debug.get('first_arm','?')} | "
            f"boundaries={debug.get('boundaries',[])}",
            fontsize=11,
        )

        # Panel 1: left gripper
        axes[0].plot(t, left_g, "b-", linewidth=1.2, label="left gripper")
        axes[0].axhline(debug.get("left_close_thresh", 0), color="b", linestyle=":", alpha=0.6, label="close thresh")
        axes[0].axhline(debug.get("left_open_thresh", 0), color="b", linestyle="--", alpha=0.6, label="open thresh")
        lfc = debug.get("left_first_close")
        lfo = debug.get("left_first_open")
        if lfc is not None:
            axes[0].axvline(lfc, color="b", linestyle=":", linewidth=1.5, label=f"close@{lfc}")
        if lfo is not None:
            axes[0].axvline(lfo, color="b", linestyle="--", linewidth=1.5, label=f"open@{lfo}")
        axes[0].set_ylabel("Left gripper")
        axes[0].legend(loc="upper right", fontsize=7)
        axes[0].grid(True, alpha=0.3)

        # Panel 2: right gripper
        axes[1].plot(t, right_g, "r-", linewidth=1.2, label="right gripper")
        axes[1].axhline(debug.get("right_close_thresh", 0), color="r", linestyle=":", alpha=0.6, label="close thresh")
        axes[1].axhline(debug.get("right_open_thresh", 0), color="r", linestyle="--", alpha=0.6, label="open thresh")
        rfc = debug.get("right_first_close")
        rfo = debug.get("right_first_open")
        if rfc is not None:
            axes[1].axvline(rfc, color="r", linestyle=":", linewidth=1.5, label=f"close@{rfc}")
        if rfo is not None:
            axes[1].axvline(rfo, color="r", linestyle="--", linewidth=1.5, label=f"open@{rfo}")
        axes[1].set_ylabel("Right gripper")
        axes[1].legend(loc="upper right", fontsize=7)
        axes[1].grid(True, alpha=0.3)

        # Panel 3: event markers
        axes[2].set_ylabel("Events")
        axes[2].set_ylim(-0.5, 1.5)
        axes[2].set_yticks([])
        for name, val, color in [
            ("L_close", lfc, "blue"),
            ("L_open", lfo, "cornflowerblue"),
            ("R_close", rfc, "red"),
            ("R_open", rfo, "salmon"),
        ]:
            if val is not None:
                axes[2].axvline(val, color=color, linewidth=2, label=f"{name}@{val}")
        for b_idx, b in enumerate(debug.get("boundaries", []), 1):
            axes[2].axvline(b, color="black", linestyle="--", linewidth=1.5, label=f"B{b_idx}@{b}")
        axes[2].legend(loc="upper right", fontsize=7)
        axes[2].grid(True, alpha=0.3)

        # Panel 4: phase bar
        phase_colors = {1: "tab:orange", 2: "tab:red", 3: "tab:green", 4: "tab:blue", 5: "tab:gray"}
        for phase_val in range(1, 6):
            mask = phases == phase_val
            axes[3].fill_between(t, 0, 1, where=mask, alpha=0.6,
                                 color=phase_colors[phase_val], label=f"P{phase_val}")
        for b in debug.get("boundaries", []):
            axes[3].axvline(b, color="black", linestyle="--", linewidth=1.5)
        axes[3].set_ylabel("Phase")
        axes[3].set_xlabel("Time Step")
        axes[3].set_yticks([])
        axes[3].legend(loc="upper right", fontsize=7)

        plt.tight_layout()
        save_path = vis_dir / f"phase_ep{ep_id:03d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(all_ep_ids)} visualizations to {vis_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_atoms(atoms_str: str | None, source_name: str) -> dict[str, str]:
    """Parse --atoms flag or fall back to default table."""
    if atoms_str is None:
        if source_name in DEFAULT_ATOMS:
            return DEFAULT_ATOMS[source_name]
        raise ValueError(
            f"No default atoms for '{source_name}'. "
            "Please pass --atoms 'pick=...,action=...,recover=...'"
        )
    atoms: dict[str, str] = {}
    for part in atoms_str.split(","):
        key, _, val = part.partition("=")
        atoms[key.strip()] = val.strip()
    # Fill missing keys with defaults
    if "idle" not in atoms:
        atoms["idle"] = IDLE
    if "recover" not in atoms:
        atoms["recover"] = RECOVER
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-label SO101 bi-arm LeRobot datasets.")
    parser.add_argument("--source", required=True, help="Path to source LeRobot dataset directory")
    parser.add_argument("--output", required=True, help="Path to output (new) dataset directory")
    parser.add_argument(
        "--atoms",
        default=None,
        help="Atomic instructions: 'pick=...,action=...,recover=...[,idle=...]'",
    )
    parser.add_argument("--dry-run", action="store_true", help="Detect phases only, no writes")
    parser.add_argument("--visualize", action="store_true", help="Save per-episode phase plots")
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated episode IDs to process (default: all)",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    source_name = source.name

    episodes_filter = None
    if args.episodes:
        episodes_filter = [int(x) for x in args.episodes.split(",")]

    atoms = parse_atoms(args.atoms, source_name)
    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Atoms: {atoms}")
    print(f"Dry run: {args.dry_run}")
    print()

    summary = process_dataset(
        source=source,
        output=output,
        atoms=atoms,
        dry_run=args.dry_run,
        visualize=args.visualize,
        episodes_filter=episodes_filter,
    )

    if summary["skipped"] > 0:
        print("\n=== SKIPPED EPISODES ===")
        for s in summary["skipped_list"]:
            print(f"  ep {s['episode_index']}: {s['reason']}")


if __name__ == "__main__":
    main()
