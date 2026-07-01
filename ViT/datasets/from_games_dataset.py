import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm, trange
from concurrent.futures import ProcessPoolExecutor
import cv2

class from_games_dataset(torch.utils.data.Dataset):
    """
    Dataset for from_games images only.
    1914 by just over 1000 (e.g. 1052)
    289800 images in total.
    """
    # The shortest sequence only has 20 images.
    def __init__(self, dir="/fs/vulcan-datasets/from_games/images"):
        super().__init__()
        self.images = os.listdir(dir)
        # (sequence, image #)
        self.image_paths = []
        for image in tqdm(self.images, desc="Precomputing Dynamic Replica image file paths"):
            self.image_paths.append(os.path.join(dir, image))
        self.len = len(self.image_paths)
        self.crop = transforms.RandomCrop(512)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            return self.crop(
                transforms.functional.to_dtype(
                    torchvision.io.decode_image(self.image_paths[idx]), torch.float32, scale = True
                )
            )
        except Exception as e:
            raise e


# 1. Define a top-level function so it can be pickled by multiprocessing
def verify_index(dataset, idx):
    try:
        # The worker accesses the globally scoped 'dataset'
        # and performs the heavy I/O on its own CPU core.
        _ = dataset[idx]
        return None
    except Exception:
        # Catch the exception locally so the ProcessPool doesn't crash
        return dataset.image_paths[idx]

if __name__ == "__main__":
    from functools import partial
    dataset = from_games_dataset()
    total_files = len(dataset)
    print(total_files)
    print(dataset[0].shape)
    invalid_files = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(partial(verify_index, dataset), range(total_files), chunksize=128)
        with tqdm(total=total_files, desc="Processing", unit="file", mininterval = 1) as pbar:
            for result in results:
                if result is not None:
                    invalid_files.append(result)
                pbar.update(1)

    print(f"\nScan complete. Found {len(invalid_files)} invalid images.")
    if invalid_files:
        with open("from_games_invalid_files.txt", "w") as f:
            for path in invalid_files:
                f.write(f"{path}\n")