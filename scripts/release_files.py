"""File selection for a public snapshot; never copy source Git history."""

from __future__ import annotations

import configparser
from pathlib import Path
import re
import subprocess


EXCLUDED_DIRS = {".git", ".claude", ".codex", ".agents", ".vscode", "__pycache__", "release-export"}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,})\b"),
    re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
)


def git_output(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def public_files(root: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Select the current tracked and non-ignored untracked files, plus registered gitlinks."""
    raw = git_output(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    files = []
    for name in sorted(set(raw.decode().split("\0")) - {""}):
        path = Path(name)
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.name.startswith(".env") or path.suffix in {".pem", ".key"}:
            continue
        source = root / path
        if source.is_symlink():
            raise ValueError(f"Review symlink before public export: {path}")
        if source.is_file():
            files.append(path)
    registered = set()
    modules = root / ".gitmodules"
    if modules.exists():
        parser = configparser.ConfigParser()
        parser.read(modules)
        registered = {parser[section]["path"] for section in parser.sections()}
    submodules = []
    for record in git_output(root, "ls-files", "--stage", "-z").decode().split("\0"):
        if not record:
            continue
        metadata, name = record.split("\t", 1)
        mode, commit, stage = metadata.split()
        if mode == "160000" and name in registered:
            submodules.append({"path": name, "commit": commit})
    return files, submodules


def validate_files(root: Path, files: list[Path]) -> list[str]:
    errors = []
    for relative in files:
        path = root / relative
        if path.stat().st_size >= 100 * 1024 * 1024:
            errors.append(f"File exceeds the ordinary GitHub file-size limit: {relative}")
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            # Report filenames only; never print credential values.
            errors.append(f"Potential credential in: {relative}")
    return errors

