"""Minimal HTTP deployment server for the trained Skill-LoRA checkpoint."""

import argparse
import dataclasses
import logging

import server_agent0
from openpi.policies import policy_config
from openpi.training import config as training_config


DEFAULT_CONFIG = "aevla_pi05_skill_lora_two_arms_multitask"


def deployment_config(name: str):
    """Remove training-only skill-label file loading from a config."""
    config = training_config.get_config(name)
    model_transforms = config.data.model_transforms
    if hasattr(model_transforms, "skill_labels_path"):
        model_transforms = dataclasses.replace(model_transforms, skill_labels_path=None)
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, model_transforms=model_transforms))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=20011)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    server_agent0.policy = policy_config.create_trained_policy(
        deployment_config(args.config),
        args.checkpoint_dir,
    )
    server_agent0.app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
