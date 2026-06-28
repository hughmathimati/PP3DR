import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm, trange
from concurrent.futures import ProcessPoolExecutor

class object_net_dataset(torch.utils.data.Dataset):
    """
    Dataset for ObjectNet images only.
    Varying resolutions.
    50273 images in total.
    """
    # The shortest sequence only has 20 images.
    def __init__(self, dir="/fs/vulcan-datasets/ObjectNet/objectnet-1.0/images"):
        super().__init__()
        self.sequences = os.listdir(dir)
        # (sequence, image #)
        self.image_paths = []
        for sequence in tqdm(self.sequences, desc="Precomputing ObjectNet image file paths"):
            sequence_dir = os.path.join(dir, sequence)
            for image in os.listdir(sequence_dir):
                self.image_paths.append(os.path.join(sequence_dir, image))
        self.len = len(self.image_paths)
        self.crop = transforms.RandomCrop(512)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            # remove A from RGBA
            image =  transforms.functional.to_dtype(
                torchvision.io.decode_image(self.image_paths[idx]), torch.float32, scale=True
            )[:3]
            # image.shape[0] == 3
            while min(image.shape[1:]) < 512:
                image = transforms.functional.resize(image, 512, transforms.functional.InterpolationMode.BICUBIC)
            return self.crop(image)
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
    dataset = object_net_dataset()
    total_files = len(dataset)
    print(total_files)
    print(dataset[0].shape)
    invalid_files = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(partial(verify_index, dataset), range(total_files), chunksize=32)
        with tqdm(total=total_files, desc="Processing", unit="file") as pbar:
            for result in results:
                if result is not None:
                    invalid_files.append(result)
                pbar.update(1)

    print(f"\nScan complete. Found {len(invalid_files)} invalid images.")
    if invalid_files:
        with open("invalid_object_net_files.txt", "w") as f:
            for path in invalid_files:
                f.write(f"{path}\n")