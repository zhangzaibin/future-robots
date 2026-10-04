"""Small configuration helpers shared by statistics and inference."""

import dataclasses
from pathlib import Path


def without_augmentation(config):
    """Disable stochastic training transforms while retaining arm selection and deltas."""
    factory = getattr(config.data, "data_transforms", None)
    if not dataclasses.is_dataclass(factory):
        return config
    probabilities = {"shuffle_prob", "image_mask_prob", "image_noise_prob", "wait_augment_prob"}
    overrides = {field.name: 0.0 for field in dataclasses.fields(factory) if field.name in probabilities}
    if not overrides:
        return config
    return dataclasses.replace(
        config, data=dataclasses.replace(config.data, data_transforms=dataclasses.replace(factory, **overrides))
    )


def norm_stats_directory(config, data_config) -> Path:
    """Use the same assets directory and ID that the training loader reads."""
    if data_config.asset_id is None:
        raise ValueError("An asset_id is required to save normalization statistics.")
    return Path(config.data.assets.assets_dir or config.assets_dirs) / data_config.asset_id
