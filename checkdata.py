# check_dataset.py
from datasets import load_from_disk
from tqdm import tqdm

dataset_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/stack_two_bowls"
ds = load_from_disk(dataset_path)

corrupted_indices = []
for i in tqdm(range(len(ds["train"])), desc="Checking"):
    try:
        item = ds["train"][i]
        # 尝试访问图像
        for key in ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]:
            if key in item:
                img = item[key]
                if hasattr(img, 'load'):
                    img.load()
    except Exception as e:
        print(f"Corrupted at index {i}: {e}")
        corrupted_indices.append(i)

print(f"\nTotal corrupted: {len(corrupted_indices)}")
print(f"Corrupted indices: {corrupted_indices}")