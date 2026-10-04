# Installation

[Documentation](README.md) · [Training](training.md)

## Environment

Use Linux with an NVIDIA GPU, a CUDA 12-compatible driver, Python **3.11+**, and [uv](https://docs.astral.sh/uv/getting-started/installation/). The project pins JAX CUDA 12 and PyTorch in `uv.lock`; this training environment is not a macOS CPU environment. `.python-version` selects Python 3.11.

```bash
git clone https://github.com/zhangzaibin/future-robots.git
cd future-robots
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
uv run python -c 'import jax; print(jax.devices())'
```

Install FFmpeg using your Linux package manager if working with video datasets. HDF5 conversion dependencies are included through the locked environment. The default `uv sync` also installs the development group, which includes plotting tools used by the configurations.

The LeRobot dependency comes from the exact Git revision in `pyproject.toml`; no separate editable LeRobot checkout is needed. Only initialize the optional simulator submodules when using them:

```bash
git submodule update --init --recursive third_party/aloha third_party/libero
```

For training, `jax.devices()` must list the intended GPU devices. Most released multi-task configs use `fsdp_devices=2` and a global batch size of 32. The global batch must be divisible by the visible device count. Single-GPU training requires `FSDP_DEVICES=1` and sufficient GPU memory; no fixed VRAM minimum has been validated for this release.

## Paths and pretrained weights

No edits to cluster-specific paths are needed for the main recipes. Set variables **before** launching Python:

| Variable | Default / meaning |
| --- | --- |
| `FUTURE_DATA_ROOT` | `<repository>/dataset`; root for configured local datasets |
| `FUTURE_CHECKPOINT_ROOT` | `<repository>/checkpoints`; training outputs and local initialization paths |
| `FUTURE_PI05_PARAMS` | `gs://openpi-assets/checkpoints/pi05_base/params` |
| `FUTURE_PI0_PARAMS` | `gs://openpi-assets/checkpoints/pi0_base/params` |
| `FUTURE_PI0_FAST_PARAMS` | `gs://openpi-assets/checkpoints/pi0_fast_base/params` |
| `FUTURE_DATASET_DIR` | Optional exact dataset override used by config lookup |
| `FUTURE_ASSETS_DIR` | Optional normalization assets root; statistics live in `<root>/<asset_id>/norm_stats.json` |
| `FUTURE_FOLDTOWEL_PARAMS` | Optional local parameter directory for towel checkpoint initialization |
| `PALIGEMMA_TOKENIZER_PATH` | Optional local `paligemma_tokenizer.model`; otherwise downloaded |
| `OPENPI_DATA_HOME` | OpenPI download cache; normally `~/.cache/openpi` |

For example:

```bash
export FUTURE_DATA_ROOT=/data/future/dataset
export FUTURE_CHECKPOINT_ROOT=/data/future/checkpoints
export FUTURE_PI05_PARAMS=/models/pi05_base/params
export PALIGEMMA_TOKENIZER_PATH=/models/paligemma_tokenizer.model
bash scripts/macg_new/train_mavla.sh --dry-run
```

The weight override must identify the **`params` directory**, whereas deployment uses a complete **checkpoint step directory** containing `params/` and `assets/`. First use of public weights/tokenizers requires network access; local overrides support offline machines once all dependencies and assets are available.

The shell recipes also accept `DATASET_DIR` and `NORM_ASSETS_DIR`, which forward to `FUTURE_DATASET_DIR` and `FUTURE_ASSETS_DIR`. `scripts/setup_env.sh` preserves values already set by the user. Changing a dataset does not change its robot transforms: choose a configuration with the correct embodiment and action layout.

## Logging and checks

W&B is enabled by the training configs. Authenticate in your own environment, use `WANDB_MODE=offline`, or pass `--no-wandb-enabled` to a training recipe. Keep tokens outside the repository.

The publication checks require only Python 3.11+ and Bash:

```bash
python3 scripts/check_release.py
python3 -m unittest discover -s tests -p 'test_*.py'
```

These checks validate source files, documentation, and launcher behavior. They do not run GPU training, download model weights, or establish benchmark performance.
