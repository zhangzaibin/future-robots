[![Future — Evolving Machines Lab. Compositional robot intelligence.](docs/media/future-eml-banner.svg)](https://zhangzaibin.github.io/evolving-machines/)

# Future

**Compositional robot intelligence, from atomic skills to open-ended agents.**

[**Evolving Machines Lab**](https://zhangzaibin.github.io/evolving-machines/) · [MA-VLA paper](https://arxiv.org/abs/2608.25864) · [AE-VLA paper](https://arxiv.org/abs/2610.06184) · [Dataset](https://huggingface.co/datasets/MrBean2024/macg_data) · [Model](https://huggingface.co/MrBean2024/AE-VLA-full) · [Research film & blog](https://zhangzaibin.github.io/evolving-machines/research/arm-wise-cg/)

**Future** is the embodied-intelligence research ecosystem of **[Evolving Machines Lab](https://zhangzaibin.github.io/evolving-machines/)**. We study how robots can reuse atomic skills, assign them to different arms, and compose them under new task and coordination requirements. Our long-term goal is **agentic, open-ended vision-language-action (VLA) policies** that connect task decomposition, compositional execution, and feedback.

This repository brings together **MA-VLA** (ECCV 2026) and **ACG-Bench / AE-VLA** (arXiv 2026), with training recipes, attention and SkillLoRA ablations, data preparation, and policy serving. The current benchmark studies execution with supplied per-arm prompts; high-level autonomous planning remains a research direction.

[Research](#research) · [Public resources](#public-resources) · [Quick start](#quick-start) · [Evaluation](#evaluation) · [Documentation](#documentation) · [Citation](#citation)

## Research

### MA-VLA · Multi-arm collaboration

[**MA-VLA: Multi-arm Vision-Language-Action Model for Collaboration and Compositional Generalization**](https://arxiv.org/abs/2608.25864) introduces atomic action assignment: cooperative tasks are expressed as mid-level instructions for individual arms. **Arm Shuffle** permutes arm observations, states, and assigned prompts during training to support reuse beyond fixed execution roles. Published at **ECCV 2026**.

The repository's [MA-VLA recipe](docs/training.md#ma-vla) uses a Pi0.5 backbone with phase-derived atomic instructions, arm shuffling, and image masking. It is the current two-arm multi-task implementation; the paper also studies other settings.

### ACG-Bench / AE-VLA · Arm-wise composition

[**Arm-wise Compositional Generalization in Dual-Arm Vision-Language-Action Models**](https://arxiv.org/abs/2610.06184) introduces **ACG-Bench** and studies three complementary design choices in **AE-VLA**:

- **Arm-token grouping:** represent each arm's actions with its own token group in a shared action expert.
- **SkillLoRA:** reuse skill-specific low-rank adapters, selected by a per-arm router.
- **Arm-wise attention:** structure information flow by arm while preserving shared scene context.

The full model combines all three. See the [research blog and demonstrations](https://zhangzaibin.github.io/evolving-machines/research/arm-wise-cg/) for the motivation, architecture, and robot videos, or the [training guide](docs/training.md#ae-vla) for the full model and ablation recipes.

## Public resources

| Resource | Link | Contents |
| --- | --- | --- |
| **Training data** | [MrBean2024/macg_data](https://huggingface.co/datasets/MrBean2024/macg_data) | Converted multi-task LeRobot data, per-arm instructions, phase annotations, and SkillLoRA labels |
| **AE-VLA checkpoint** | [MrBean2024/AE-VLA-full](https://huggingface.co/MrBean2024/AE-VLA-full) | Full SkillLoRA + arm-wise attention checkpoint, with parameters and normalization assets |
| **MA-VLA paper** | [arXiv:2608.25864](https://arxiv.org/abs/2608.25864) | Multi-arm collaboration and compositional generalization · ECCV 2026 |
| **ACG-Bench / AE-VLA paper** | [arXiv:2610.06184](https://arxiv.org/abs/2610.06184) | Benchmark, architecture study, and simulation / real-robot evaluation |
| **Research film & blog** | [Evolving Machines Lab](https://zhangzaibin.github.io/evolving-machines/research/arm-wise-cg/) | Method overview and simulation / SO101 demonstrations |

The public dataset contains **700 episodes / 395,759 frames at 30 FPS**, with three RGB views, two arms' joint and gripper states, 16-dimensional actions, and phase labels. Its [metadata](https://huggingface.co/datasets/MrBean2024/macg_data/blob/main/meta/info.json) identifies LeRobot **v2.1** format and a training split; the 34 task entries are instruction entries, not the number of benchmark task families. Images are embedded in parquet files. See [data preparation](docs/data.md#public-training-dataset) for the file layout.

## Quick start

### 1. Install the environment

Training and inference use **Linux, NVIDIA GPUs, a CUDA 12-compatible driver, and Python 3.11+**. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/zhangzaibin/future-robots.git
cd future-robots
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
uv run python -c 'import jax; print(jax.devices())'
```

Follow the [installation guide](docs/installation.md) for pretrained Pi0.5 weights, local tokenizer paths, offline setup, and GPU configuration. The project retains the `openpi` Python import namespace.

### 2. Download the training data

Run from the repository root. This places the public dataset at the path used by the multi-task recipes:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MrBean2024/macg_data",
    repo_type="dataset",
    local_dir="dataset/multi_task_combined",
)
PY
```

The release includes `meta/skill_labels.jsonl`, required for full AE-VLA training. To use another location, set `DATASET_DIR=/path/to/dataset` when launching a recipe. For your own demonstrations, follow the [conversion and merging workflow](docs/data.md#simulation-data). Downloads use Hugging Face's [`snapshot_download`](https://huggingface.co/docs/huggingface_hub/guides/download); record the Hub revision for reproducible runs.

### 3. Train a policy

```bash
# Full AE-VLA: arm-token grouping + SkillLoRA + arm-wise attention.
# Preview the commands without loading data, weights, or a GPU.
bash scripts/macg_new/train_skill_lora.sh --dry-run
EXP_NAME=aevla_full_run1 bash scripts/macg_new/train_skill_lora.sh

# MA-VLA: atomic instructions with arm shuffle and image masking.
EXP_NAME=mavla_run1 bash scripts/macg_new/train_mavla.sh
```

Choose a matched baseline or ablation as needed:

| Model / variant | Launcher | Configuration |
| --- | --- | --- |
| **Full AE-VLA** | `scripts/macg_new/train_skill_lora.sh` | `aevla_pi05_skill_lora_two_arms_multitask` |
| **MA-VLA** | `scripts/macg_new/train_mavla.sh` | `pi05_two_arms_multi_task_mavla_shuffle06_mask04` |
| Arm-token grouping + arm-wise attention, without SkillLoRA | `scripts/macg_new/train_aevla.sh` | `aevla_pi05_two_arms_multi_task` |
| Arm-token grouping with full attention, without SkillLoRA | `scripts/macg_new/train_tokengrouponly.sh` | `aevla_pi05_two_arms_multi_task_full_attn` |
| SkillLoRA with full attention | `scripts/macg_new/train_fullattn.sh` | `aevla_pi05_skill_lora_two_arms_multitask_full_attn` |
| Unified Pi0.5 | `scripts/macg_new/pi05/train.sh` | `pi05_two_arms_multi_task` |
| Separate Pi0.5, left / right | [Per-arm launchers](docs/training.md#separate-pi05) | `pi05_single_arm_left_multi_task` / `pi05_single_arm_right_multi_task` |

Each recipe computes normalization statistics before training. These multi-task configs default to **30,000 steps, global batch size 32, and two FSDP devices**. Existing experiments are preserved by default; use `--resume` to continue a run. See [common options](docs/training.md#common-options) for dataset overrides, single-GPU settings, seeds, and logging.

### 4. Run the released checkpoint

To use pretrained AE-VLA, download the model and serve it directly. Training and a local copy of the training dataset are not required for this serving path.

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MrBean2024/AE-VLA-full",
    local_dir="checkpoints/AE-VLA-full",
)
PY

CHECKPOINT_DIR=checkpoints/AE-VLA-full/aevla_pi05_skill_lora_two_arms_multitask/29999 \
  PORT=20011 bash scripts/serve_recipe.sh aevla_pi05_skill_lora_two_arms_multitask
```

The checkpoint path must point to the **step directory containing `params/` and `assets/`**. The public model uses `aevla_pi05_skill_lora_two_arms_multitask`; select that same configuration for serving. The server loads checkpoint normalization assets and skips training-only SkillLoRA label loading.

The HTTP server defaults to `127.0.0.1:20011`; `GET /health` reports model status. Connect a compatible robot or simulator client to execute actions. See [deployment](docs/deployment.md) for the observation format, trusted-network HTTP protocol, separate-arm policies, and WebSocket options.

## Evaluation

**ACG-Bench** contains **23 task–condition pairs across eight task families**: six in-domain conditions and 17 unseen compositions. All methods receive the same per-arm atomic prompts; success requires the task goal and its physical, order, and timing constraints. [Paper and protocol](https://arxiv.org/abs/2610.06184).

| Held-out composition | What changes? |
| --- | --- |
| **Reorder** | Which arm acts first |
| **Sync** | When designated events must complete together |
| **Sync + Reorder** | Timing and event order jointly |
| **Cross-task** | Which skills from different source tasks are combined |

Reported mean success on unseen conditions:

| Policy | Simulation · 17 conditions | SO101 robots · 5 conditions |
| --- | ---: | ---: |
| Single Pi0.5 | 2.94% | 2.00% |
| MA-VLA, adapted to Pi0.5 | 3.06% | 6.00% |
| Dual Pi0.5 | 5.53% | 10.00% |
| **AE-VLA** | **21.53%** | **39.00%** |

These are published results from the [AE-VLA paper](https://arxiv.org/abs/2610.06184) and [lab's experiment summary](https://zhangzaibin.github.io/evolving-machines/research/arm-wise-cg/), not measurements produced by this checkout. Simulation and real-robot conditions differ. The repository provides training and serving entry points; the complete benchmark rollout harness and explicit held-out split manifest are not bundled. See the [evaluation guide](docs/evaluation.md) for release scope and experiment records.

## Documentation

| I want to… | Start here |
| --- | --- |
| Set up GPUs, pretrained weights, or an offline tokenizer | [Installation and paths](docs/installation.md) |
| Download data, convert trajectories, or merge multi-task data | [Data preparation](docs/data.md) |
| Compute statistics without mixing arms or augmentations | [Normalization](docs/data.md#normalization) |
| Train models or compare attention / SkillLoRA ablations | [Training recipes](docs/training.md) · [SkillLoRA design](docs/skill_lora.md) |
| Define held-out compositions and report evaluation | [Evaluation protocol](docs/evaluation.md) |
| Train on SO101 / Bi-SO100 or fine-tune towel folding | [Real-world workflows](docs/real_world.md) |
| Serve a checkpoint or independent per-arm policies | [Deployment](docs/deployment.md) |
| Find model implementations and earlier experiments | [Documentation index and code map](docs/README.md#code-map) |
| Contribute or prepare a public code snapshot | [Contributing](CONTRIBUTING.md) · [Publishing](docs/publishing.md) |

## Citation

If you use Future, its models, data, or benchmark in your research, please cite the relevant papers:

```bibtex
@inproceedings{zhang2026ma,
  title={MA-VLA: Multi-arm Vision-Language-Action Model for Collaboration and Compositional Generalization},
  author={Zhang, Zaibin and Xiao, Junlan and Zhang, Zhongbo and Wang, Yifan and Kang, Li and Qin, Yiran and Xia, Changxing and Zhou, Heng and Fu, Talas and Zhou, Enshen and others},
  booktitle={European Conference on Computer Vision},
  pages={399--417},
  year={2026},
  organization={Springer}
}

@article{zhang2026arm,
  title={Arm-wise Compositional Generalization in Dual-Arm Vision-Language-Action Models},
  author={Zhang, Zaibin and Ran, Binghao and Wu, Yuhan and Zhang, Zhongbo and Wang, Yifan and Jiang, Junwei and Xiao, Junlan and Shi, Wangcheng and Kang, Li and Qin, Yiran and others},
  journal={arXiv preprint arXiv:2610.06184},
  year={2026}
}
```

## Acknowledgments and license

Future builds on [OpenPI](https://github.com/Physical-Intelligence/openpi), [LeRobot](https://github.com/huggingface/lerobot), and the open robotics community. Code is provided under [Apache 2.0](LICENSE), retaining upstream notices. See [Gemma's license](LICENSE_GEMMA.txt) for the included model license text; pretrained weights and external datasets retain their respective terms.

---

**Build what evolves.**

[Evolving Machines Lab](https://zhangzaibin.github.io/evolving-machines/) · Digital intelligence × Embodied intelligence

[Research and projects](https://zhangzaibin.github.io/evolving-machines/) · [Contact](mailto:dlutzzb@gmail.com)
