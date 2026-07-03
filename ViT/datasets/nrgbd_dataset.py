import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm

class nrgbd_dataset(torch.utils.data.Dataset):
    """
    Dataset for NRGBD images only.
    640 x 480
    10970 images in total.
    """
    # The shortest sequence only has 20 images.
    def __init__(self, dir="/vulcanscratch/hughma/data/nrgbd/"):
        super().__init__()
        self.sequences = os.listdir(dir)
        # (sequence, image #)
        self.image_paths = []
        for sequence in tqdm(self.sequences, desc="Precomputing NRGBD image file paths"):
            sequence_dir = os.path.join(dir, sequence, "images")
            for image in os.listdir(sequence_dir):
                self.image_paths.append(os.path.join(sequence_dir, image))
        self.len = len(self.image_paths)
        self.crop = transforms.RandomCrop(512)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = transforms.functional.to_dtype(torchvision.io.decode_image(self.image_paths[idx]), torch.float32, scale = True)
        # image.shape[0] == 3
        image = transforms.functional.resize(image, 512, transforms.functional.InterpolationMode.BICUBIC)
        return self.crop(image)

if __name__ == "__main__":
    dataset = nrgbd_dataset()
    print(len(dataset))
    print(dataset[0].shape)