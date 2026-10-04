"""Merge multiple LeRobot datasets into a single combined dataset for multi-task training.

All source datasets must share the same feature schema (columns, dtypes, image sizes, fps).
Episodes, frames, and task indices are re-indexed to be globally unique.

Usage:
    python scripts/set_atom/macg/merge_multitask_datasets.py \
        --sources dataset/MACG/two_arms_stack_bowl/two_arms_stack_bowl_combined \
                  dataset/MACG/two_arms_stack_cube/two_arms_stack_cube_combined \
                  dataset/MACG/two_arms_push_cube/two_arms_push_cube_combined \
        --output dataset/MACG/multi_task/multi_task_combined
"""

import argparse
import json
import shutil
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(path: Path, items: list[dict]):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def validate_sources(sources: list[str], output: str, *, overwrite: bool = False) -> list[Path]:
    """Check inputs before creating or removing any output files."""
    source_paths = [Path(source).resolve() for source in sources]
    output_path = Path(output).resolve()
    if not source_paths or len(set(source_paths)) != len(source_paths):
        raise ValueError("Provide at least one source, without duplicate datasets.")
    for source in source_paths:
        if source == output_path or source in output_path.parents or output_path in source.parents:
            raise ValueError("Output and source dataset directories must not overlap.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Choose a new output or pass --overwrite.")
    reference = None
    for source in source_paths:
        info = json.loads((source / "meta/info.json").read_text())
        schema = (info.get("features"), info.get("fps"), info.get("robot_type"))
        if reference is not None and schema != reference:
            raise ValueError(f"Incompatible feature schema, FPS, or robot type: {source}")
        reference = schema
        if any(feature.get("dtype") == "video" for feature in info.get("features", {}).values()):
            raise ValueError(f"External video datasets must be converted to embedded images first: {source}")
        if info.get("total_chunks", 1) != 1:
            raise ValueError(f"This merger currently requires a single chunk-000 dataset: {source}")
        episodes = load_jsonl(source / "meta/episodes.jsonl")
        tasks = load_jsonl(source / "meta/tasks.jsonl")
        if not episodes or len(episodes) != info.get("total_episodes"):
            raise ValueError(f"Empty or inconsistent episode metadata: {source}")
        if len({episode["episode_index"] for episode in episodes}) != len(episodes):
            raise ValueError(f"Duplicate episode indices: {source}")
        if not tasks or len({task["task_index"] for task in tasks}) != len(tasks):
            raise ValueError(f"Empty or duplicate task indices: {source}")
        for episode in episodes:
            parquet = source / "data/chunk-000" / f"episode_{episode['episode_index']:06d}.parquet"
            if not parquet.is_file():
                raise FileNotFoundError(f"Episode parquet is missing: {parquet}")
    return source_paths


def merge_datasets(sources: list[str], output: str, *, overwrite: bool = False):
    source_paths = validate_sources(sources, output, overwrite=overwrite)
    # Import only after validation, so the preflight can run without the training stack.
    import pandas as pd

    output_path = Path(output)
    if output_path.exists():
        shutil.rmtree(output_path)

    output_data_dir = output_path / "data" / "chunk-000"
    output_meta_dir = output_path / "meta"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    reference_info = json.loads((source_paths[0] / "meta" / "info.json").read_text())

    global_episode_idx = 0
    global_frame_idx = 0
    task_offset = 0

    all_tasks = []
    all_episodes = []
    all_episode_stats = []
    total_frames = 0

    for src_idx, src_path in enumerate(source_paths):
        print(f"\nProcessing source {src_idx}: {src_path}")
        info = json.loads((src_path / "meta" / "info.json").read_text())
        tasks = load_jsonl(src_path / "meta" / "tasks.jsonl")
        episodes = load_jsonl(src_path / "meta" / "episodes.jsonl")

        episode_stats_path = src_path / "meta" / "episodes_stats.jsonl"
        episode_stats = load_jsonl(episode_stats_path) if episode_stats_path.exists() else []

        num_tasks_in_src = len(tasks)
        print(f"  Episodes: {info['total_episodes']}, Frames: {info['total_frames']}, Tasks: {num_tasks_in_src}")

        for task_number, t in enumerate(tasks):
            all_tasks.append({
                "task_index": task_number + task_offset,
                "task": t["task"],
            })

        old_to_new_task = {t["task_index"]: i + task_offset for i, t in enumerate(tasks)}

        src_frame_start = global_frame_idx

        for ep in sorted(episodes, key=lambda e: e["episode_index"]):
            old_ep_idx = ep["episode_index"]
            new_ep_idx = global_episode_idx

            parquet_file = src_path / "data" / f"chunk-000" / f"episode_{old_ep_idx:06d}.parquet"
            if not parquet_file.exists():
                raise FileNotFoundError(f"Episode parquet is missing: {parquet_file}")

            df = pd.read_parquet(parquet_file)

            ep_length = len(df)
            df["episode_index"] = new_ep_idx
            df["index"] = range(global_frame_idx, global_frame_idx + ep_length)
            if not df["task_index"].isin(old_to_new_task).all():
                raise ValueError(f"Unknown task_index in {parquet_file}")
            df["task_index"] = df["task_index"].map(old_to_new_task)

            out_parquet = output_data_dir / f"episode_{new_ep_idx:06d}.parquet"
            df.to_parquet(out_parquet)

            new_task_strings = []
            for task_str in ep.get("tasks", []):
                new_task_strings.append(task_str)

            all_episodes.append({
                "episode_index": new_ep_idx,
                "tasks": new_task_strings,
                "length": ep_length,
            })

            ep_stat_entry = next((s for s in episode_stats if s["episode_index"] == old_ep_idx), None)
            if ep_stat_entry is not None:
                new_stat = json.loads(json.dumps(ep_stat_entry))
                new_stat["episode_index"] = new_ep_idx
                stats = new_stat.get("stats", {})
                if "episode_index" in stats:
                    stats["episode_index"]["min"] = [new_ep_idx]
                    stats["episode_index"]["max"] = [new_ep_idx]
                    stats["episode_index"]["mean"] = [float(new_ep_idx)]
                if "index" in stats:
                    stats["index"]["min"] = [global_frame_idx]
                    stats["index"]["max"] = [global_frame_idx + ep_length - 1]
                    stats["index"]["mean"] = [global_frame_idx + (ep_length - 1) / 2.0]
                if "task_index" in stats:
                    values = df["task_index"].to_numpy()
                    stats["task_index"].update(
                        min=[int(values.min())],
                        max=[int(values.max())],
                        mean=[float(values.mean())],
                        std=[float(values.std())],
                    )
                all_episode_stats.append(new_stat)

            global_frame_idx += ep_length
            global_episode_idx += 1

        total_frames += info["total_frames"]
        task_offset += num_tasks_in_src
        print(f"  After this source: {global_episode_idx} episodes, {global_frame_idx} frames, {task_offset} tasks")

    merged_info = dict(reference_info)
    merged_info["total_episodes"] = global_episode_idx
    merged_info["total_frames"] = global_frame_idx
    merged_info["total_tasks"] = len(all_tasks)
    merged_info["total_chunks"] = 1
    merged_info["splits"] = {"train": f"0:{global_episode_idx}"}

    (output_meta_dir / "info.json").write_text(json.dumps(merged_info, indent=4) + "\n")
    save_jsonl(output_meta_dir / "tasks.jsonl", all_tasks)
    save_jsonl(output_meta_dir / "episodes.jsonl", all_episodes)
    if all_episode_stats:
        save_jsonl(output_meta_dir / "episodes_stats.jsonl", all_episode_stats)

    print(f"\nMerge complete!")
    print(f"  Output: {output_path}")
    print(f"  Total episodes: {global_episode_idx}")
    print(f"  Total frames: {global_frame_idx}")
    print(f"  Total tasks: {len(all_tasks)}")
    print(f"\nTasks:")
    for t in all_tasks:
        print(f"  [{t['task_index']}] {t['task']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets into one.")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Paths to source LeRobot dataset directories",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output merged dataset directory",
    )
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace the output dataset")
    args = parser.parse_args()
    merge_datasets(args.sources, args.output, overwrite=args.overwrite)
