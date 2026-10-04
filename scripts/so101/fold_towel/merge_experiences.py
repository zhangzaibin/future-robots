#!/usr/bin/env python3
"""Convert and append SO101 fold-towel experience datasets to foldtowel_v2."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DESTINATION = ROOT_DIR / "dataset" / "foldtowel_v2"
DEFAULT_CONFIG = "pi05_bi_so100_flodtowel_pretrained_3k"
HISTORY_FILENAME = "foldtowel_merge_history.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def jsonl_bytes(values: list[dict]) -> bytes:
    return "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values).encode()


def atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def source_fingerprint(source: Path) -> str:
    """Fingerprint metadata and payload layout without hashing large video files."""
    digest = hashlib.sha256()
    digest.update(b"foldtowel-experience-v1\0")

    metadata_root = source / "meta"
    for path in sorted(metadata_root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(source).as_posix().encode())
            digest.update(path.read_bytes())

    payload_files = sorted((source / "data").rglob("*.parquet"))
    payload_files += sorted((source / "videos").rglob("*.mp4"))
    for path in payload_files:
        stat = path.stat()
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(str(stat.st_size).encode())

    return digest.hexdigest()


def validate_v3_source(source: Path) -> None:
    info_path = source / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"找不到源数据集元信息：{info_path}")
    info = load_json(info_path)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{source} 不是 LeRobot v3.0 数据集")
    if info.get("robot_type") != "bi_so100_follower":
        raise ValueError(
            f"{source} 的 robot_type 是 {info.get('robot_type')!r}，"
            "预期为 'bi_so100_follower'"
        )


def field_signature(schema: pa.Schema) -> list[tuple[str, str]]:
    return [(field.name, str(field.type)) for field in schema]


def update_stat_indices(
    entry: dict,
    episode_index: int,
    frame_start: int,
    episode_length: int,
    task_index: int,
) -> dict:
    updated = json.loads(json.dumps(entry))
    updated["episode_index"] = episode_index
    stats = updated.get("stats", {})

    if "episode_index" in stats:
        stats["episode_index"].update(
            min=[episode_index],
            max=[episode_index],
            mean=[float(episode_index)],
            std=[0.0],
        )
    if "index" in stats:
        stats["index"].update(
            min=[frame_start],
            max=[frame_start + episode_length - 1],
            mean=[frame_start + (episode_length - 1) / 2.0],
        )
    if "task_index" in stats:
        stats["task_index"].update(
            min=[task_index],
            max=[task_index],
            mean=[float(task_index)],
            std=[0.0],
        )
    return updated


def convert_source(source: Path, converted: Path, prompt: str) -> None:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "convert_lerobot_v3_to_v2.py"),
        "--src",
        str(source),
        "--dst",
        str(converted),
        "--robot-type",
        "bi_so100_follower",
        "--default-prompt",
        prompt,
    ]
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def append_converted(
    converted: Path,
    destination: Path,
    source: Path,
    fingerprint: str,
    history: list[dict],
) -> tuple[int, int]:
    info_path = destination / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"目标数据集不存在：{destination}。请先用 train.sh 转换初始数据集。"
        )

    info = load_json(info_path)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"目标数据集 {destination} 不是 LeRobot v2.1")

    tasks_path = destination / "meta" / "tasks.jsonl"
    episodes_path = destination / "meta" / "episodes.jsonl"
    stats_path = destination / "meta" / "episodes_stats.jsonl"
    tasks = load_jsonl(tasks_path)
    episodes = load_jsonl(episodes_path)
    episode_stats = load_jsonl(stats_path)
    if len(tasks) != 1:
        raise ValueError(
            f"foldtowel_v2 应只有一个任务，实际发现 {len(tasks)} 个；"
            "为避免错误映射，已停止合并"
        )
    destination_task_index = int(tasks[0]["task_index"])
    destination_task = str(tasks[0]["task"])

    converted_info = load_json(converted / "meta" / "info.json")
    if converted_info.get("fps") != info.get("fps"):
        raise ValueError(
            f"FPS 不一致：目标为 {info.get('fps')}，新增数据为 {converted_info.get('fps')}"
        )

    source_episodes = load_jsonl(converted / "meta" / "episodes.jsonl")
    source_stats = {
        int(item["episode_index"]): item
        for item in load_jsonl(converted / "meta" / "episodes_stats.jsonl")
    }
    if not source_episodes:
        raise ValueError(f"转换后的数据集没有 episode：{converted}")

    next_episode = max((int(item["episode_index"]) for item in episodes), default=-1) + 1
    next_frame = int(info.get("total_frames", 0))
    chunk_size = int(info.get("chunks_size", 1000))
    reference_files = sorted((destination / "data").glob("chunk-*/*.parquet"))
    if not reference_files:
        raise ValueError(f"目标数据集没有 parquet 文件：{destination / 'data'}")
    reference_schema = pq.read_schema(reference_files[0])

    new_files: list[Path] = []
    new_episodes: list[dict] = []
    new_stats: list[dict] = []
    appended_frames = 0
    metadata_paths = [
        info_path,
        episodes_path,
        stats_path,
        destination / "meta" / HISTORY_FILENAME,
    ]
    metadata_backups = {
        path: path.read_bytes() if path.exists() else None for path in metadata_paths
    }

    try:
        for offset, episode in enumerate(
            sorted(source_episodes, key=lambda item: int(item["episode_index"]))
        ):
            old_episode = int(episode["episode_index"])
            new_episode = next_episode + offset
            source_file = (
                converted / "data" / "chunk-000" / f"episode_{old_episode:06d}.parquet"
            )
            if not source_file.exists():
                raise FileNotFoundError(f"找不到转换后的 episode：{source_file}")

            table = pq.read_table(source_file)
            if field_signature(table.schema) != field_signature(reference_schema):
                raise ValueError(
                    f"episode {old_episode} 的字段结构与 foldtowel_v2 不一致"
                )

            episode_length = table.num_rows
            frame_start = next_frame + appended_frames
            replacements = {
                "episode_index": pa.array(
                    [new_episode] * episode_length,
                    type=table.schema.field("episode_index").type,
                ),
                "index": pa.array(
                    range(frame_start, frame_start + episode_length),
                    type=table.schema.field("index").type,
                ),
                "task_index": pa.array(
                    [destination_task_index] * episode_length,
                    type=table.schema.field("task_index").type,
                ),
            }
            for name, values in replacements.items():
                column_index = table.schema.get_field_index(name)
                table = table.set_column(column_index, name, values)

            chunk = new_episode // chunk_size
            output_dir = destination / "data" / f"chunk-{chunk:03d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"episode_{new_episode:06d}.parquet"
            if output_file.exists():
                raise FileExistsError(f"目标 episode 已存在：{output_file}")
            temporary_file = output_file.with_name(f".{output_file.name}.tmp")
            pq.write_table(table, temporary_file)
            os.replace(temporary_file, output_file)
            new_files.append(output_file)

            new_episodes.append(
                {
                    "episode_index": new_episode,
                    "tasks": [destination_task],
                    "length": episode_length,
                }
            )
            if old_episode in source_stats:
                new_stats.append(
                    update_stat_indices(
                        source_stats[old_episode],
                        new_episode,
                        frame_start,
                        episode_length,
                        destination_task_index,
                    )
                )
            appended_frames += episode_length

        updated_info = dict(info)
        updated_info["total_episodes"] = len(episodes) + len(new_episodes)
        updated_info["total_frames"] = int(info.get("total_frames", 0)) + appended_frames
        updated_info["total_tasks"] = len(tasks)
        updated_info["total_chunks"] = max(
            1, (updated_info["total_episodes"] + chunk_size - 1) // chunk_size
        )
        updated_info["splits"] = {"train": f"0:{updated_info['total_episodes']}"}

        updated_history = [
            *history,
            {
                "source": str(source.resolve()),
                "fingerprint": fingerprint,
                "episodes": len(new_episodes),
                "frames": appended_frames,
                "destination_episode_start": next_episode,
                "destination_episode_end": next_episode + len(new_episodes) - 1,
            }
        ]

        atomic_write(info_path, json_bytes(updated_info))
        atomic_write(episodes_path, jsonl_bytes(episodes + new_episodes))
        if episode_stats or new_stats:
            atomic_write(stats_path, jsonl_bytes(episode_stats + new_stats))
        atomic_write(
            destination / "meta" / HISTORY_FILENAME,
            json_bytes(updated_history),
        )
    except Exception:
        for path in new_files:
            path.unlink(missing_ok=True)
        for path, content in metadata_backups.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write(path, content)
        raise

    return len(new_episodes), appended_frames


def recompute_norm_stats(destination: Path, config_name: str) -> None:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "compute_norm_stats.py"),
        "--config-name",
        config_name,
    ]
    subprocess.run(command, cwd=ROOT_DIR, check=True, env={**os.environ, "FUTURE_DATASET_DIR": str(destination.resolve())})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将一个或多个 LeRobot v3 fold-towel experience 数据集转换为 v2，"
            "并增量追加到 foldtowel_v2"
        )
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="待合并的数据集，例如 dataset/foldtowel_experience2",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"目标 v2 数据集（默认：{DEFAULT_DESTINATION}）",
    )
    parser.add_argument(
        "--prompt",
        default="fold the towel.",
        help="源数据没有任务文本时使用的提示词",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG,
        help="重新计算归一化统计使用的训练配置",
    )
    parser.add_argument(
        "--skip-norm-stats",
        action="store_true",
        help="合并后不重新计算 norm_stats（通常不建议）",
    )
    parser.add_argument(
        "--recompute-norm-stats",
        action="store_true",
        help="即使没有新增数据，也重新计算 norm_stats",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination.resolve()
    history_path = destination / "meta" / HISTORY_FILENAME
    history = load_json(history_path) if history_path.exists() else []
    known_fingerprints = {item["fingerprint"] for item in history}
    merged_episodes = 0
    merged_frames = 0

    for source_arg in args.sources:
        source = source_arg.resolve()
        validate_v3_source(source)
        fingerprint = source_fingerprint(source)
        if fingerprint in known_fingerprints:
            print(f"已合并过，跳过：{source}")
            continue

        print(f"\n准备转换并合并：{source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{source.name}_v2_",
            dir=destination.parent,
        ) as temporary_dir:
            converted = Path(temporary_dir)
            convert_source(source, converted, args.prompt)
            episodes, frames = append_converted(
                converted,
                destination,
                source,
                fingerprint,
                history,
            )
        history = load_json(history_path)
        known_fingerprints.add(fingerprint)
        merged_episodes += episodes
        merged_frames += frames
        print(f"已追加 {episodes} 个 episodes、{frames} 帧")

    should_compute_stats = merged_episodes or args.recompute_norm_stats
    if should_compute_stats and not args.skip_norm_stats:
        print("\n正在为合并后的完整数据集重新计算归一化统计...")
        recompute_norm_stats(destination, args.config_name)

    if merged_episodes:
        info = load_json(destination / "meta" / "info.json")
        print(
            f"\n合并完成：本次新增 {merged_episodes} 个 episodes、{merged_frames} 帧；"
            f"当前共 {info['total_episodes']} 个 episodes、{info['total_frames']} 帧。"
        )
    else:
        print("\n没有需要合并的新数据。")


if __name__ == "__main__":
    main()
