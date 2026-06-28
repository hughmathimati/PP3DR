import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm

class dtu_dataset(torch.utils.data.Dataset):
    """
    Dataset for ETH3D images only.

    1078 images in total.
    """
    # The shortest sequence only has 20 images.
    def __init__(self, dir="/vulcanscratch/hughma/data/dtu/"):
        self.sequences = os.listdir(dir)
        # (sequence, image #)
        self.image_paths = []
        for sequence in tqdm(self.sequences, desc="Precomputing all image file paths"):
            sequence_dir = os.path.join(dir, sequence, "images")
            for image in os.listdir(sequence_dir):
                self.image_paths.append(os.path.join(sequence_dir, image))
        self.len = len(self.image_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return transforms.functional.to_dtype(torchvision.io.decode_image(self.image_paths[idx]), torch.float32, scale = True)

if __name__ == "__main__":
    dataset = dtu_dataset()
    print(len(dataset))
    print(dataset[0].shape)