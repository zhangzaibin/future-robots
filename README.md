# 🌱 Future

**Compositional robot intelligence, from atomic skills to open-ended agents.**

[**Evolving Machines Lab**](https://zhangzaibin.github.io/evolving-machines/) · [Installation](docs/installation.md) · [Training](docs/training.md) · [Real robots](docs/real_world.md) · [Documentation](docs/README.md)

**Future** is a research ecosystem working toward one goal: **agentic, open-ended VLA policies** — agents that decompose a goal into atomic skills, compose them on the fly, and coordinate multiple effectors to solve tasks they were never explicitly trained on.

Developed at **[Evolving Machines Lab](https://zhangzaibin.github.io/evolving-machines/)**, Future studies how robots can generalize through composition: reuse learned skills, rearrange their order, and coordinate them across arms and tasks.

Its first work, [**MA-VLA**](https://arxiv.org/abs/2608.25864) *(Multi-Arm Vision-Language-Action Model for Collaboration and Compositional Generalization)*, introduces **multi-arm compositional generalization** and is accepted to **ECCV 2026**.

The next work, **AE-VLA**, advances both **evaluation and methods for compositional generalization**. This repository includes its current training implementations, attention ablations, and Skill-LoRA experiments. More works are on the way. 🚀

## Research and code

| Work / baseline | Role | Start here |
| --- | --- | --- |
| **MA-VLA** · ECCV 2026 | Multi-arm collaboration and compositional generalization | [Training recipe](docs/training.md#ma-vla) · [Paper](https://arxiv.org/abs/2608.25864) |
| **AE-VLA** · ongoing work | Evaluation and methods for composing learned skills across tasks | [Training and ablations](docs/training.md#ae-vla) · [Evaluation protocol](docs/evaluation.md) |
| **Separate Pi0.5** | One independent policy for each arm | [Left / right recipes](docs/training.md#separate-pi05) |
| **Unified Pi0.5** | One standard policy controlling both arms | [Baseline](docs/training.md#unified-pi05) |
| **Real-world experiments** | SO101 / Bi-SO100 multi-task manipulation and towel folding | [Data → training → deployment](docs/real_world.md) |

The MA-VLA entry below is the current **Pi0.5 multi-task recipe** requested for this code release. Configuration names and model families are listed explicitly so experiments can be reproduced. AE-VLA's final benchmark splits, paper, checkpoints, and quantitative results are not bundled here; see [release scope](docs/evaluation.md#release-scope).

## Get started

Training uses Linux, NVIDIA GPUs, and Python 3.11+. Follow the [installation guide](docs/installation.md) to install the pinned environment and configure data and pretrained weights.

```bash
git clone https://github.com/zhangzaibin/future-robots.git
cd future-robots
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
```

Prepare a phase-annotated LeRobot dataset at `dataset/multi_task_combined` using the [data guide](docs/data.md), then choose a recipe:

```bash
# MA-VLA: inspect the commands first; --dry-run requires no GPU or uv.
bash scripts/macg_new/train_mavla.sh --dry-run
bash scripts/macg_new/train_mavla.sh

# AE-VLA: arm-wise attention and per-agent prompts.
bash scripts/macg_new/train_aevla.sh

# Separate Pi0.5: train independent left and right policies.
bash scripts/macg_new/per-arm-policy/train_pi05_single_arm_left_multi_task.sh
bash scripts/macg_new/per-arm-policy/train_pi05_single_arm_right_multi_task.sh
```

Each training recipe computes normalization statistics before training. Existing experiments are preserved by default; use `--resume` to continue a run or explicitly pass `--overwrite` to replace it. Most multi-task configs use **two FSDP devices**; [single-GPU overrides](docs/training.md#common-options) are available when memory permits.

## Find your workflow

| I want to… | Documentation |
| --- | --- |
| Set up GPUs, pretrained weights, or an offline tokenizer | [Installation and paths](docs/installation.md#paths-and-pretrained-weights) |
| Convert trajectories and merge multi-task data | [Data preparation](docs/data.md#simulation-data) |
| Compute statistics without mixing arms or augmentations | [Normalization](docs/data.md#normalization) |
| Train MA-VLA, AE-VLA, or separate Pi0.5 policies | [Training recipes](docs/training.md) |
| Run full-attention and Skill-LoRA experiments | [AE-VLA ablations](docs/training.md#ae-vla) · [Skill-LoRA design](docs/skill_lora.md) |
| Define held-out compositions and report evaluation | [Evaluation protocol](docs/evaluation.md#compositional-generalization-protocol) |
| Train on SO101 multi-task data | [Real-world multi-task pipeline](docs/real_world.md#so101-multi-task) |
| Fine-tune towel folding, merge DAgger data, or resume | [Towel-folding workflow](docs/real_world.md#towel-folding) |
| Serve unified or separate policies | [Deployment](docs/deployment.md) |
| Navigate model code and older experiments | [Code map](docs/README.md#code-map) |
| Publish a clean code snapshot to `future-robots` | [Publishing guide](docs/publishing.md) |

## Acknowledgments and license

Future builds on [OpenPI](https://github.com/Physical-Intelligence/openpi), [LeRobot](https://github.com/huggingface/lerobot), and the broader open robotics community. The Python package retains the `openpi` import namespace for compatibility.

Code is provided under [Apache 2.0](LICENSE), retaining upstream notices. See [Gemma's license](LICENSE_GEMMA.txt) for the included model license text; pretrained weights and external datasets retain their respective terms.

For MA-VLA, please cite the [paper](https://arxiv.org/abs/2608.25864). For AE-VLA, refer to this repository until its paper and citation metadata are released. Research updates and contact information are on the [Evolving Machines Lab website](https://zhangzaibin.github.io/evolving-machines/).
