# Compositional generalization evaluation

[Documentation](README.md) · [Training](training.md) · [Deployment](deployment.md)

AE-VLA extends Future's research into both evaluation and methods for compositional generalization. This guide defines what to record for a reproducible comparison with the code currently available.

## Release scope

This snapshot includes training configurations, task-specific data preparation, policy serving, attention ablations, and Skill-LoRA experiments. It does **not** include a finalized AE-VLA benchmark split manifest, a complete simulator/robot rollout harness, released benchmark checkpoints, or quantitative result tables. The serving scripts expose policies; starting a server does not run an evaluation.

The protocol below is a recommended reporting structure, not a claim about AE-VLA's final paper protocol or measured performance. MA-VLA's published settings should be taken from its [paper](https://arxiv.org/abs/2608.25864).

## Compositional generalization protocol

1. **Define the atomic skill vocabulary.** Record instructions, phase labels, embodiment, arm identities, and the state/action representation.
2. **Split by composition.** Hold out combinations of skills, task orders, or arm assignments. Record which component skills appear in training. A random trajectory split alone does not demonstrate unseen-composition generalization.
3. **Prevent leakage.** Keep held-out trajectories out of data merging, normalization, checkpoint selection, and hyperparameter tuning. Publish explicit train/validation/test manifests when releasing the benchmark.
4. **Compare matched settings.** Use the same observations, demonstrations, horizons, task instructions, seed list, and rollout budget for MA-VLA, AE-VLA, unified Pi0.5, and separate Pi0.5. Record model size and total compute, including both separate policies.
5. **Report both composition and control outcomes.** Include full-task success, per-arm/skill completion where measurable, coordination failures, episode length, and per-composition results. Define each success criterion before evaluation and report seeds/trial counts and uncertainty.

## Experiment record

For every run, retain the repository commit, config name, command and environment overrides, dataset/split identifiers, normalization assets, pretrained checkpoint, trained checkpoint step, evaluation seeds, and robot/simulator version. Include augmentation probabilities, batch size, FSDP device count, and any manually changed prompts.

Training loss alone does not establish compositional generalization. Compare closed-loop rollouts on the held-out compositions with the same action execution and reset protocol. For separate policies, send synchronized observations and concatenate actions in the same physical arm order used during training.
