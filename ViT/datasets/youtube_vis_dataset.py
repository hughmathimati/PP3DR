import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class youtube_vis_dataset(torch.utils.data.Dataset):
    """
    Dataset for YoutubeVis-2021 images only.
    Varying resolutions.
    90160 images in total.
    """
    # The shortest sequence only has 20 images.
    def __init__(self,
                 dir="/fs/vulcan-datasets/YouTubeVIS-2021/train/train/JPEGImages/",
                 invalid_files_list = "/vulcanscratch/hughma/ViT/datasets/invalid_youtube_vis_files.txt"
                 ):
        super().__init__()
        self.sequences = os.listdir(dir)
        # (sequence, image #)
        self.image_paths = []
        for sequence in tqdm(self.sequences, desc="Precomputing YouTubeVIS-2021 image file paths"):
            sequence_dir = os.path.join(dir, sequence)
            for image in os.listdir(sequence_dir):
                self.image_paths.append(os.path.join(sequence_dir, image))

        invalid_set = set()
        if os.path.exists(invalid_files_list):
            with open(invalid_files_list, 'r') as f:
                # .strip() is mandatory to remove the hidden '\n' from each line
                invalid_set = {line.strip() for line in f}
            print(f"YoutubeVis-2021: Loaded {len(invalid_set)} known invalid file paths.")
        else:
            print(f"YoutubeVis-2021: Warning: Exclusion list '{invalid_files_list}' not found. Proceeding unfiltered.")

        # 3. Filter the main list
        # We cast `p` to string just in case your discovery method returns Pathlib objects
        self.image_paths = [
            p for p in self.image_paths
            if str(p) not in invalid_set
        ]

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
    dataset = youtube_vis_dataset()
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
        with open("invalid_youtube_vis_files.txt", "w") as f:
            for path in invalid_files:
                f.write(f"{path}\n")