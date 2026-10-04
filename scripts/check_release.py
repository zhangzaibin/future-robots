#!/usr/bin/env python3
"""Dependency-free source, shell, documentation, and release-content checks."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

from release_files import public_files
from release_files import validate_files


ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *(ROOT / "docs" / name for name in (
    "README.md", "installation.md", "training.md", "data.md", "evaluation.md",
    "real_world.md", "deployment.md", "publishing.md",
)), ROOT / "scripts/so101/README.md"]


def heading_ids(text: str) -> set[str]:
    ids, counts = set(), {}
    for title in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", text, re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        ids.add(f"{slug}-{count}" if count else slug)
        counts[slug] = count + 1
    return ids


def check_links(path: Path) -> list[str]:
    errors = []
    for link in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", path.read_text()):
        if re.match(r"(?:https?://|mailto:)", link):
            continue
        target, _, anchor = unquote(link).partition("#")
        destination = (path.parent / target).resolve() if target else path
        if not destination.exists():
            errors.append(f"{path.relative_to(ROOT)}: missing link {link}")
        elif anchor and destination.suffix == ".md" and anchor not in heading_ids(destination.read_text()):
            errors.append(f"{path.relative_to(ROOT)}: missing anchor {link}")
    return errors


def main() -> int:
    files, _ = public_files(ROOT)
    errors = validate_files(ROOT, files)
    for relative in files:
        path = ROOT / relative
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(), filename=str(relative))
            except SyntaxError as exc:
                errors.append(str(exc))
        if path.suffix == ".sh":
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            if result.returncode:
                errors.append(f"{relative}: {result.stderr.strip()}")
    for path in DOCS:
        errors.extend(check_links(path))
    config = ROOT / "src/openpi/training/config.py"
    tree = ast.parse(config.read_text())
    names = [keyword.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "TrainConfig"
             for keyword in node.keywords if keyword.arg == "name" and isinstance(keyword.value, ast.Constant)]
    if len(names) != len(set(names)):
        errors.append("Duplicate training configuration names")
    for path in (ROOT / "scripts/macg_new").rglob("*.sh"):
        for name in re.findall(r"scripts/(?:train|serve)_recipe.sh\"\s+([\w]+)", path.read_text()):
            if name not in names:
                errors.append(f"{path.relative_to(ROOT)}: unknown configuration {name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Release checks passed: {len(files)} files, {len(names)} configs, {len(DOCS)} workflow documents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
