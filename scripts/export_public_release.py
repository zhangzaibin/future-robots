#!/usr/bin/env python3
"""Export current source files, excluding local settings, secrets, and Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from release_files import git_output
from release_files import public_files
from release_files import validate_files


def export_snapshot(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Export to a separate directory outside the source checkout.")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be new or empty: {output}")
    files, submodules = public_files(source)
    errors = validate_files(source, files)
    if errors:
        raise ValueError("\n".join(errors))
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "source_commit": git_output(source, "rev-parse", "HEAD").decode().strip(),
        "includes_working_tree_changes": True,
        "source_git_history_included": False,
        "paper_url": "https://arxiv.org/abs/2608.25864",
        "submodules": submodules,
        "sha256": hashes,
    }
    (output / "PUBLIC_RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = export_snapshot(args.source, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Exported {len(manifest['sha256'])} files to {args.output.resolve()}")
    print("Source Git history is excluded. See docs/publishing.md before pushing.")


if __name__ == "__main__":
    main()
