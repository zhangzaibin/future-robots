from google.cloud import storage
import os

bucket_name = "openpi-assets"
prefix = "checkpoints/pi0_fast_base"
local_dir = "./pi0_fast_base"

client = storage.Client.create_anonymous_client()
bucket = client.bucket(bucket_name)
blobs = client.list_blobs(bucket_name, prefix=prefix)

os.makedirs(local_dir, exist_ok=True)

for blob in blobs:
    rel_path = os.path.relpath(blob.name, prefix)
    if rel_path == ".":
        continue
    local_path = os.path.join(local_dir, rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading {blob.name} -> {local_path}")
    blob.download_to_filename(local_path)

print("Done.")