from openpi.shared import download
from openpi.training import config

config = config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_droid")
print(checkpoint_dir)
