"""Finalize an interrupted Orbax checkpoint left as `*.orbax-checkpoint-tmp-N` dirs.

When a JAX/openpi training job is killed *between* writing data and the atomic
rename that Orbax's CheckpointManager performs, the on-disk layout looks like::

    <step>.orbax-checkpoint-tmp-0/
        _CHECKPOINT_METADATA            (item_handlers=null, commit_timestamp_nsecs=null)
        assets.orbax-checkpoint-tmp-1/
        params.orbax-checkpoint-tmp-2/
            _METADATA, _sharding, array_metadatas/, ocdbt.process_0/{d,manifest.ocdbt}
        train_state.orbax-checkpoint-tmp-3/

All the actual array data is on disk inside the per-process OCDBT subdir, but
the top-level OCDBT manifest/data files (which point to the per-process subdirs)
were never written, and the directories were never renamed to drop their tmp
suffix. The result is a checkpoint that no Orbax reader will accept.

This script performs exactly the steps Orbax would have performed during a
successful finalize:

  1. (optionally) delete the partial ``train_state.*`` item if you only want
     inference weights.
  2. rename ``<item>.orbax-checkpoint-tmp-N``  ->  ``<item>``
  3. for each PyTree item (params, train_state), call
     ``merge_ocdbt_per_process_files`` so that the top-level ``manifest.ocdbt``
     and ``d/`` are created and reference ``ocdbt.process_*`` subdirs.
  4. patch ``_CHECKPOINT_METADATA`` to fill in ``item_handlers`` and
     ``commit_timestamp_nsecs`` so that ``CheckpointManager`` recognizes the
     step.
  5. rename the outer ``<step>.orbax-checkpoint-tmp-0`` -> ``<step>``
  6. verify the result by reading the params metadata back with
     ``ocp.PyTreeCheckpointer``.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/finalize_orbax_checkpoint.py \
        checkpoints/pi05_three_arms_place_objects/pi05_three_arms_place_objects/19999.orbax-checkpoint-tmp-0 \
        --drop-train-state

You can also pass a parent run dir, e.g.

    .venv/bin/python scripts/finalize_orbax_checkpoint.py \
        checkpoints/pi05_three_arms_place_objects/pi05_three_arms_place_objects \
        --drop-train-state

and it will discover and finalize every ``<step>.orbax-checkpoint-tmp-*`` step
inside it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

from etils import epath
import orbax.checkpoint as ocp
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handlers
import tensorstore as ts


_TMP_RE = re.compile(r"^(?P<name>.+)\.orbax-checkpoint-tmp-(?P<idx>\d+)$")

ITEM_HANDLERS_DEFAULT = {
    "assets": "openpi.training.checkpoints.CallbackHandler",
    "params": "orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler",
    "train_state": "orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler",
}


def _strip_tmp(name: str) -> str | None:
    m = _TMP_RE.match(name)
    return m.group("name") if m else None


def _is_step_tmp_dir(p: Path) -> bool:
    """Return True if `p` looks like '<step>.orbax-checkpoint-tmp-N' with int step."""
    if not p.is_dir():
        return False
    base = _strip_tmp(p.name)
    if base is None:
        return False
    try:
        int(base)
    except ValueError:
        return False
    return True


def _merge_ocdbt(item_dir: Path) -> None:
    """Run merge_ocdbt_per_process_files on a PyTree item dir."""
    if not any(item_dir.glob(f"{ts_utils.PROCESS_SUBDIR_PREFIX}*")):
        logging.info("[%s] no per-process OCDBT subdir, skipping merge", item_dir.name)
        return

    if (item_dir / "manifest.ocdbt").exists() and (item_dir / "d").exists():
        logging.info("[%s] top-level OCDBT manifest already present, skipping merge", item_dir.name)
        return

    logging.info("[%s] merging per-process OCDBT files into top-level manifest...", item_dir.name)
    ts_context = ts_utils.get_ts_context(use_ocdbt=True)

    asyncio.run(
        type_handlers.merge_ocdbt_per_process_files(
            epath.Path(item_dir),
            ts_context=ts_context,
            use_zarr3=False,
            enable_validation=True,
        )
    )
    logging.info("[%s] merge complete", item_dir.name)


def _patch_metadata(step_dir: Path, present_items: list[str]) -> None:
    """Fill in item_handlers and commit_timestamp_nsecs in _CHECKPOINT_METADATA."""
    meta_path = step_dir / "_CHECKPOINT_METADATA"
    if not meta_path.exists():
        logging.warning("[%s] no _CHECKPOINT_METADATA found, skipping metadata patch", step_dir.name)
        return

    meta = json.loads(meta_path.read_text())

    needs_handlers = meta.get("item_handlers") in (None, {})
    needs_commit = meta.get("commit_timestamp_nsecs") is None

    if not (needs_handlers or needs_commit):
        logging.info("[%s] _CHECKPOINT_METADATA already finalized, no patch needed", step_dir.name)
        return

    if needs_handlers:
        meta["item_handlers"] = {k: ITEM_HANDLERS_DEFAULT[k] for k in present_items if k in ITEM_HANDLERS_DEFAULT}
    if needs_commit:
        now_ns = time.time_ns()
        if meta.get("init_timestamp_nsecs") is not None and now_ns <= meta["init_timestamp_nsecs"]:
            now_ns = meta["init_timestamp_nsecs"] + 1
        meta["commit_timestamp_nsecs"] = now_ns

    meta_path.write_text(json.dumps(meta))
    logging.info("[%s] patched _CHECKPOINT_METADATA (item_handlers=%s)", step_dir.name, list(meta["item_handlers"]))


def _verify_params(params_dir: Path) -> None:
    logging.info("[%s] verifying params can be read by orbax PyTreeCheckpointer...", params_dir.parent.name)
    with ocp.PyTreeCheckpointer() as ckptr:
        meta = ckptr.metadata(epath.Path(params_dir))
    top = list(meta.keys()) if isinstance(meta, dict) else None
    logging.info("[%s] verify OK. top-level keys: %s", params_dir.parent.name, top)


def finalize_step(
    step_tmp_dir: Path,
    *,
    drop_train_state: bool,
    dry_run: bool,
) -> Path:
    """Finalize a single `<step>.orbax-checkpoint-tmp-N` directory in place.

    Returns the final step directory path (with the tmp suffix removed).
    """
    if not _is_step_tmp_dir(step_tmp_dir):
        raise ValueError(f"Not a step tmp dir: {step_tmp_dir}")

    final_step_name = _strip_tmp(step_tmp_dir.name)
    final_step_dir = step_tmp_dir.parent / final_step_name
    if final_step_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing finalized step dir: {final_step_dir}")

    logging.info("=" * 70)
    logging.info("Finalizing %s -> %s%s", step_tmp_dir.name, final_step_name, " (DRY RUN)" if dry_run else "")
    logging.info("=" * 70)

    item_tmp_dirs: dict[str, Path] = {}
    for child in sorted(step_tmp_dir.iterdir()):
        base = _strip_tmp(child.name)
        if base is not None and child.is_dir():
            item_tmp_dirs[base] = child

    if not item_tmp_dirs:
        raise RuntimeError(f"No `*.orbax-checkpoint-tmp-*` item subdirs found in {step_tmp_dir}")
    logging.info("Found item subdirs: %s", sorted(item_tmp_dirs))

    if drop_train_state and "train_state" in item_tmp_dirs:
        ts_dir = item_tmp_dirs.pop("train_state")
        logging.info("[drop] removing train_state subdir: %s", ts_dir.name)
        if not dry_run:
            shutil.rmtree(ts_dir)

    final_item_dirs: dict[str, Path] = {}
    for base, src in item_tmp_dirs.items():
        dst = step_tmp_dir / base
        if dst.exists():
            raise FileExistsError(f"Cannot rename {src} -> {dst}: destination already exists")
        logging.info("[rename] %s -> %s", src.name, dst.name)
        if not dry_run:
            src.rename(dst)
        final_item_dirs[base] = dst

    for base in ("params", "train_state"):
        if base in final_item_dirs:
            if dry_run:
                logging.info("[%s] (dry-run) would merge OCDBT per-process files", base)
            else:
                _merge_ocdbt(final_item_dirs[base])

    if dry_run:
        logging.info("[meta] (dry-run) would patch _CHECKPOINT_METADATA for items: %s", sorted(final_item_dirs))
    else:
        _patch_metadata(step_tmp_dir, present_items=sorted(final_item_dirs))

    if dry_run:
        logging.info("[outer-rename] (dry-run) would rename %s -> %s", step_tmp_dir.name, final_step_name)
        return final_step_dir

    step_tmp_dir.rename(final_step_dir)
    logging.info("[outer-rename] %s -> %s", step_tmp_dir.name, final_step_dir.name)

    if "params" in final_item_dirs:
        _verify_params(final_step_dir / "params")

    logging.info("Done. Finalized checkpoint at: %s", final_step_dir)
    return final_step_dir


def discover_step_tmp_dirs(target: Path) -> list[Path]:
    if _is_step_tmp_dir(target):
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(target)
    return sorted(p for p in target.iterdir() if _is_step_tmp_dir(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "target",
        type=Path,
        help="Either a `<step>.orbax-checkpoint-tmp-N` dir, or a parent run dir containing one or more such dirs.",
    )
    ap.add_argument("--drop-train-state", action="store_true",
                    help="Delete the partial train_state subdir (only params+assets needed for inference).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done without modifying anything.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )

    target = args.target.resolve()
    candidates = discover_step_tmp_dirs(target)
    if not candidates:
        logging.error("No `*.orbax-checkpoint-tmp-*` step dirs found under: %s", target)
        return 1

    logging.info("Will finalize %d step(s):", len(candidates))
    for c in candidates:
        logging.info("  - %s", c)

    for c in candidates:
        finalize_step(c, drop_train_state=args.drop_train_state, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
