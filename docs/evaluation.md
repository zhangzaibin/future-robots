# Compositional generalization evaluation

[Documentation](README.md) · [Training](training.md) · [Deployment](deployment.md)

The [ACG-Bench / AE-VLA paper](https://arxiv.org/abs/2610.06184) studies arm-wise compositional generalization in dual-arm policies. ACG-Bench evaluates reordering, synchronization, their combination, and cross-task composition. All methods receive the same per-arm atomic prompts; success includes physical milestones and the specified order or timing constraints. This guide defines what to record for a reproducible comparison with the code currently available.

## Release scope

This snapshot includes training configurations, task-specific data preparation, policy serving, attention ablations, and SkillLoRA experiments. The [training data](https://huggingface.co/datasets/MrBean2024/macg_data) and [full AE-VLA checkpoint](https://huggingface.co/MrBean2024/AE-VLA-full) are publicly available on Hugging Face. The [README](../README.md#evaluation) summarizes published results, and the [lab blog](https://zhangzaibin.github.io/evolving-machines/research/arm-wise-cg/) presents demonstrations.

The code snapshot does **not** bundle an explicit held-out benchmark split manifest or a complete simulator/robot rollout harness. The public dataset metadata describes the training split; it does not replace the paper's evaluation conditions. Serving scripts expose policies; starting a server does not run an evaluation.

Use the [AE-VLA paper](https://arxiv.org/abs/2610.06184) for benchmark conditions, success criteria, and reported results, and the [MA-VLA paper](https://arxiv.org/abs/2608.25864) for its original experimental settings. The protocol below is a reporting checklist for experiments with this code.

## Compositional generalization protocol

1. **Define the atomic skill vocabulary.** Record instructions, phase labels, embodiment, arm identities, and the state/action representation.
2. **Split by composition.** Hold out combinations of skills, task orders, or arm assignments. Record which component skills appear in training. A random trajectory split alone does not demonstrate unseen-composition generalization.
3. **Prevent leakage.** Keep held-out trajectories out of data merging, normalization, checkpoint selection, and hyperparameter tuning. Publish explicit train/validation/test manifests when releasing the benchmark.
4. **Compare matched settings.** Use the same observations, demonstrations, horizons, task instructions, seed list, and rollout budget for MA-VLA, AE-VLA, unified Pi0.5, and separate Pi0.5. Record model size and total compute, including both separate policies.
5. **Report both composition and control outcomes.** Include full-task success, per-arm/skill completion where measurable, coordination failures, episode length, and per-composition results. Define each success criterion before evaluation and report seeds/trial counts and uncertainty.

## Experiment record

For every run, retain the repository commit, config name, command and environment overrides, dataset/split identifiers, normalization assets, pretrained checkpoint, trained checkpoint step, evaluation seeds, and robot/simulator version. Include augmentation probabilities, batch size, FSDP device count, and any manually changed prompts.

Training loss alone does not establish compositional generalization. Compare closed-loop rollouts on the held-out compositions with the same action execution and reset protocol. For separate policies, send synchronized observations and concatenate actions in the same physical arm order used during training.
