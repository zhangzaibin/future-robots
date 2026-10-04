# Publish the code to future-robots

[Documentation](README.md) · [Target repository](https://github.com/zhangzaibin/future-robots)

This release replaces `main` with a **new root commit**, as requested by the repository owner. Neither the experimental repository's history nor the previous public `main` history is an ancestor of the new release. Keep the experimental checkout's `origin` unchanged and publish from a separate clean export.

The MA-VLA paper is linked directly on [arXiv](https://arxiv.org/abs/2608.25864). No repository copy of the PDF is required.

## Validate and export

From the experimental checkout:

```bash
python3 scripts/check_release.py
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/export_public_release.py --output ../future-robots-public
```

The export directory must be new or empty. The exporter copies tracked and non-ignored new source files, excludes Git history and local assistant/editor settings, rejects detected credential material, and records file hashes plus registered submodule revisions in `PUBLIC_RELEASE_MANIFEST.json`. The manifest includes local uncommitted changes; it describes the current files, not just `HEAD`.

An obsolete, unregistered `lerobot` gitlink is excluded. uv installs the pinned LeRobot dependency. The registered ALOHA and LIBERO submodule revisions are retained in the manifest.

## Create the new history

In the exported directory, initialize a new repository and stage the snapshot:

```bash
cd ../future-robots-public
git init -b main
git remote add origin https://github.com/zhangzaibin/future-robots.git
# Stage exactly the scanned snapshot, including source files ignored by legacy rules.
# Restore the registered submodule gitlinks without downloading their repositories.
python3 - <<'PY'
import json
import subprocess
from pathlib import Path
manifest = json.loads(Path('PUBLIC_RELEASE_MANIFEST.json').read_text())
subprocess.run([
    'git', 'add', '--force', '--', *manifest['sha256'], 'PUBLIC_RELEASE_MANIFEST.json',
], check=True)
for module in manifest['submodules']:
    subprocess.run([
        'git', 'update-index', '--add', '--cacheinfo',
        '160000', module['commit'], module['path'],
    ], check=True)
PY

python3 scripts/check_release.py
python3 -m unittest discover -s tests -p 'test_*.py'
git diff --cached --stat
git status --short
git commit -m "Release Future: MA-VLA and AE-VLA research ecosystem"
git rev-list --count HEAD
# Expected: 1
```

If the export has already been initialized and committed, inspect that commit instead of repeating initialization. Never initialize or rewrite history inside the experimental checkout.

## Replace remote main

Read the current remote commit and pass that exact value as the lease. This authorizes replacement of the version you inspected and rejects a concurrent update:

```bash
git ls-remote origin refs/heads/main
# Copy the returned commit ID into REMOTE_MAIN below.
REMOTE_MAIN=PASTE_THE_REMOTE_COMMIT_ID_HERE
git push --force-with-lease="refs/heads/main:${REMOTE_MAIN}" origin main:main

git rev-parse HEAD
git ls-remote origin refs/heads/main
# The two commit IDs must match.
```

This force-push replaces `main` with the new single-commit history. Other remote branches, tags, releases, and repository settings are unchanged. Branch protection may require the owner to allow this specific operation. Do not push the source experiment repository's history.

If Git reports DNS or authentication errors, restore network access or authenticate on the publishing machine before rerunning the push. A local export or commit is not evidence of a successful push; verify the remote commit and final files on GitHub.

## Credential history

Embedded SSH private keys were removed from the current scripts during release preparation. The old experimental history still contains those keys. Revoke/rotate the affected credentials. A fresh root commit avoids importing that history, but rewriting `main` is not a guarantee that old objects, forks, caches, or other references have disappeared from GitHub.
