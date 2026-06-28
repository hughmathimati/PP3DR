import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class flying_things_3d_dataset(torch.utils.data.Dataset):
    """
    NOTICE: cannot open ... for reading: Permission denied

    Dataset for FlyingThings3D images only.
    540x960
    22390 images in total.
    """
    # The shortest sequence only has 20 images.
    def __init__(self, dir="/fs/vulcan-datasets/FlyingThings3D/frames_cleanpass/TRAIN"):
        self.sequences_A = os.listdir(os.path.join(dir, "A"))
        self.sequences_B = os.listdir(os.path.join(dir, "B"))
        self.sequences_C = os.listdir(os.path.join(dir, "C"))
        # (sequence, image #)
        self.image_paths = []

        for sequence in tqdm(self.sequences_A, desc="Precomputing FlyingThings3D image file paths (A)"):
            # We'll just take the left views for now.
            sequence_dir = os.path.join(dir, "A", sequence, "left")
            for image in os.listdir(sequence_dir):
                self.image_paths.append(os.path.join(sequence_dir, image))

        for sequence in tqdm(self.sequences_B, desc="Precomputing FlyingThings3D image file paths (B)"):
            # We'll just take the left views for now.
            sequence_dir = os.path.join(dir, "B", sequence, "left")
            for image in os.listdir(sequence_dir):
                self.image_paths.append(os.path.join(sequence_dir, image))

        for sequence in tqdm(self.sequences_C, desc="Precomputing FlyingThings3D image file paths (C)"):
            # We'll just take the left views for now.
            sequence_dir = os.path.join(dir, "C", sequence, "left")
            for image in os.listdir(sequence_dir):
                self.image_paths.append(os.path.join(sequence_dir, image))
        self.len = len(self.image_paths)
        self.crop = transforms.RandomCrop(512)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            return self.crop(transforms.functional.to_dtype(torchvision.io.decode_image(self.image_paths[idx]), torch.float32, scale = True))
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
    dataset = flying_things_3d_dataset()
    total_files = len(dataset)
    print(total_files)
    invalid_files = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(partial(verify_index, dataset), range(total_files), chunksize=64)
        with tqdm(total=total_files, desc="Processing", unit="file", mininterval = 1) as pbar:
            for result in results:
                if result is not None:
                    invalid_files.append(result)
                pbar.update(1)

    print(f"\nScan complete. Found {len(invalid_files)} invalid images.")
    if invalid_files:
        with open("invalid_flying_things_3d_files.txt", "w") as f:
            for path in invalid_files:
                f.write(f"{path}\n")