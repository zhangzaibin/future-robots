# Future documentation

[Project overview](../README.md) · [Evolving Machines Lab](https://zhangzaibin.github.io/evolving-machines/)

Follow one complete route:

- **Released model:** [install](installation.md) → [download AE-VLA](../README.md#4-run-the-released-checkpoint) → [serve](deployment.md#released-ae-vla-checkpoint).
- **Simulation:** [install](installation.md) → [download training data](data.md#public-training-dataset) or [prepare your own](data.md#simulation-data) → [train](training.md) → [evaluate](evaluation.md) → [serve](deployment.md).
- **Real robots:** [install](installation.md) → [SO101 / Bi-SO100](real_world.md) → [serve](deployment.md).
- **Public release:** [validate and export](publishing.md) → review the replacement → push a branch to `future-robots`.

## Code map

| Location | Purpose |
| --- | --- |
| [`src/openpi/training/config.py`](../src/openpi/training/config.py) | Named training configurations; authoritative hyperparameters |
| [`src/openpi/training/paths.py`](../src/openpi/training/paths.py) | Portable data, checkpoint, and pretrained-weight defaults |
| [`scripts/train_recipe.sh`](../scripts/train_recipe.sh) | Shared normalization / training launcher |
| [`scripts/macg_new/`](../scripts/macg_new/) | MA-VLA, AE-VLA, Pi0.5, and attention ablations |
| [`src/openpi/models/pi0_multi_agent_attn.py`](../src/openpi/models/pi0_multi_agent_attn.py) | Arm-wise attention implementation |
| [`src/openpi/models/pi0_skill_lora_composer.py`](../src/openpi/models/pi0_skill_lora_composer.py) | Skill-LoRA composition implementation |
| [`src/openpi/policies/droid_policy.py`](../src/openpi/policies/droid_policy.py) | Multi-arm observation / action transforms |
| [`scripts/set_atom/macg/`](../scripts/set_atom/macg/) | Task-specific phase labeling, conversion, and merging |
| [`scripts/so101/`](../scripts/so101/) | Real-world data processing and training |
| [`packages/openpi-client/`](../packages/openpi-client/) | Policy client for a separate robot environment |

## Technical references

- [Skill-LoRA](skill_lora.md), [token router design](token_router_design.md), [multi-agent attention](MACG/README_multi_agent_attention.md).
- [Placefactory pipeline](MACG/README_placefactory.md), [phase conversion](../scripts/README_phase_conversion.md).
- [SO101 phase detection notes](../scripts/so101/README.md).
- [OpenPI normalization background](norm_stats.md), [remote inference](remote_inference.md), [Docker](docker.md).
- [UR5](../examples/ur5/README.md), [ALOHA](../examples/aloha_real/README.md), [LIBERO](../examples/libero/README.md), [PyTorch LIBERO notes](Colla_vla/QUICKSTART_PYTORCH_LIBERO.md).

The workflow guides above describe the supported release entry points. Older experiment scripts and technical notes remain for research provenance; some contain machine-specific paths and should be adapted before use. Refer to the [AE-VLA paper](https://arxiv.org/abs/2610.06184) for the published benchmark protocol and results, and the [README citations](../README.md#citation) for both papers.
