"""Non-destructive probe to check whether an orbax-checkpoint-tmp directory contains
intact PyTree data that can be read by ocp.PyTreeCheckpointer.

Usage: python scripts/probe_orbax_tmp.py <path-to-params-dir>
"""

import sys
from pathlib import Path

import orbax.checkpoint as ocp


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-params-dir>")
        sys.exit(2)

    params_dir = Path(sys.argv[1]).resolve()
    print(f"[probe] reading params dir: {params_dir}")
    print(f"[probe] exists={params_dir.exists()} is_dir={params_dir.is_dir()}")

    with ocp.PyTreeCheckpointer() as ckptr:
        try:
            metadata = ckptr.metadata(params_dir)
            print("[probe] metadata() OK. Top-level keys:", list(metadata.keys()))
        except Exception as e:
            print(f"[probe] metadata() FAILED: {type(e).__name__}: {e}")
            return 1

        try:
            n_arrays = sum(1 for _ in _walk_leaves(metadata))
            print(f"[probe] number of arrays in metadata: {n_arrays}")
        except Exception as e:
            print(f"[probe] walking metadata failed: {e}")

    print("[probe] OK - params directory looks readable by orbax.")
    return 0


def _walk_leaves(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk_leaves(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _walk_leaves(v)
    else:
        yield node


if __name__ == "__main__":
    sys.exit(main())
