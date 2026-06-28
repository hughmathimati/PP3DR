import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm

class eth3d_dataset(torch.utils.data.Dataset):
    """
    WARNING: These images are freaking 6k. And the low-res ones are apparently greyscale.
    Dataset for ETH3D images only.
    3024 x 2016
    454 images in total.
    """
    # The shortest sequence only has 20 images.
    def __init__(self, dir="/vulcanscratch/hughma/data/eth3d/"):
        super().__init__()
        self.sequences = os.listdir(dir)
        # (sequence, image #)
        self.image_paths = []
        for sequence in tqdm(self.sequences, desc="Precomputing ETH3D image file paths"):
            sequence_dir = os.path.join(dir, sequence, "images", "dslr_images")
            for image in os.listdir(sequence_dir):
                self.image_paths.append(os.path.join(sequence_dir, image))
        self.len = len(self.image_paths)
        self.crop = transforms.RandomCrop(512)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.crop(
            transforms.functional.to_dtype(
                torchvision.io.decode_image(self.image_paths[idx]), torch.float32, scale = True
            ),
        )

if __name__ == "__main__":
    dataset = eth3d_dataset()
    print(len(dataset))
    print(dataset[0].shape)