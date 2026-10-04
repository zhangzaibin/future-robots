# Policy deployment

[Documentation](README.md) · [Training](training.md) · [Evaluation](evaluation.md)

Use the exact training configuration and a complete checkpoint **step directory**, including `params/` and `assets/`. Inference loads statistics from the checkpoint. Stochastic training augmentation is disabled during inference, preserving physical arm order.

## Unified policy

For the existing HTTP robot-client protocol:

```bash
CHECKPOINT_DIR=/data/checkpoints/aevla_pi05_two_arms_multi_task/run1/29999 \
  bash scripts/serve_recipe.sh aevla_pi05_two_arms_multi_task --dry-run

CHECKPOINT_DIR=/data/checkpoints/aevla_pi05_two_arms_multi_task/run1/29999 \
  PORT=20011 bash scripts/serve_recipe.sh aevla_pi05_two_arms_multi_task
```

The launcher uses `uv run --with flask deploy_minimal.py`, supplying Flask as an optional serving dependency. For offline deployment, install/cache Flask first. `deploy_minimal.py` strips training-only Skill-LoRA label-file loading, so serving a Skill-LoRA checkpoint does not require its training dataset.

`GET /health` reports whether a model is loaded. The legacy `POST /infer` endpoint accepts a JSON `example` containing a base64-encoded Python pickle, with the observation's `task` field used as the prompt. The response contains a similarly encoded `result` and `actions_shape`.

**This compatibility endpoint deserializes Python pickle. Bind it only to a trusted robot network and trusted clients.** The default host is `127.0.0.1`; set `HOST` to the intended interface when the robot is remote. Do not expose it directly to the public Internet.

## Separate policies

Start one server per trained arm, with independent checkpoints and ports:

```bash
CUDA_VISIBLE_DEVICES=0 CHECKPOINT_DIR=/data/left_run/29999 \
  bash scripts/macg_new/per-arm-policy/deploy_left_arm0.sh

# A second terminal:
CUDA_VISIBLE_DEVICES=1 CHECKPOINT_DIR=/data/right_run/29999 \
  bash scripts/macg_new/per-arm-policy/deploy_right_arm1.sh
```

Default ports are **20021** (left / arm 0) and **20022** (right / arm 1). `PORT`, `HOST`, and `CUDA_VISIBLE_DEVICES` are explicit overrides. These policies were trained from full two-arm observations; send the matching complete observation layout and let each configured transform select its arm. Synchronize observations and combine physical arm actions in left-then-right order in the client. Two running servers alone do not implement robot coordination or rollout evaluation.

## WebSocket clients

OpenPI's WebSocket server is also available for standard attention / Pi0.5 checkpoints:

```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config aevla_pi05_two_arms_multi_task \
  --policy.dir /data/checkpoints/aevla_run/29999
```

See [remote inference](remote_inference.md) and [`openpi-client`](../packages/openpi-client/) for the observation transport. The HTTP pickle and WebSocket protocols are different; match the client to the server. The existing WebSocket server binds all interfaces, so use it on a trusted network. Use the HTTP minimal launcher for Skill-LoRA when training-only label files are unavailable.
