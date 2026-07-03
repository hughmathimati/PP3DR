import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


class open_images_dataset(torch.utils.data.Dataset):
    """
    NOTICE: cannot open ... for reading: Permission denied

    Dataset for OpenImagesv4 images only.
    Varying resolutions.
    1743042 images in total.
    """

    # The shortest sequence only has 20 images.
    def __init__(self, dir="/fs/vulcan-datasets/OpenImagesv4/train_"):
        train_sets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        # (sequence, image #)
        self.image_paths = []
        for image_set in train_sets:
            folder = dir + image_set
            for image in tqdm(os.listdir(folder), desc=f"Precomputing OpenImagesv4 image file paths (train set {image_set})"):
                self.image_paths.append(os.path.join(folder, image))

        self.len = len(self.image_paths)
        self.crop = transforms.RandomCrop(512)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            image = transforms.functional.to_dtype(
                torchvision.io.decode_image(
                    self.image_paths[idx],
                    mode=torchvision.io.ImageReadMode.RGB  # Forces 1-channel to 3-channel
                ),
                torch.float32,
                scale=True
            )
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

    dataset = open_images_dataset()
    total_files = len(dataset)
    print(total_files)
    print(dataset[0].shape)
    invalid_files = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(partial(verify_index, dataset), range(total_files), chunksize=total_files // 1024)
        with tqdm(total=total_files, desc="Processing", unit="file", mininterval=1) as pbar:
            for result in results:
                if result is not None:
                    invalid_files.append(result)
                pbar.update(1)

    print(f"\nScan complete. Found {len(invalid_files)} invalid images.")
    if invalid_files:
        with open("invalid_open_images_files.txt", "w") as f:
            for path in invalid_files:
                f.write(f"{path}\n")
#