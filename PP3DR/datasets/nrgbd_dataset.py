import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm

class nrgbd_dataset(torch.utils.data.Dataset):
    """
    Dataset for NRGBD images
    680 x 480
    9 sequences with a little over a thousand frames each. We're going to pick a random starting frame before the last
    500 frames, then randomly sample 100 frames from the subsequent 500 frames.
    """
    # The shortest sequence only has 20 images.
    def __init__(self, dir="/vulcanscratch/hughma/data/nrgbd/"):
        super().__init__()
        self.sequence_paths = os.listdir(dir)
        # (sequence, image #)
        self.sequences = [dict(images = [], depths = [])] * len(self.sequence_paths)
        for i, sequence in tqdm(enumerate(self.sequence_paths), desc="Precomputing NRGBD image file paths"):
            for image in os.listdir(os.path.join(dir, sequence, "images")):
                self.sequences[i]['images'].append(os.path.join(sequence_dir, image))
            for depth in os.listdir(os.path.join(dir, sequence, "depth")):
                self.sequences[i]['depths'].append(os.path.join(sequence_dir, image))
            self.sequences[i]['poses'] = os.path.join(dir, sequence, "poses.txt")
            self.sequences[i]['focal'] = os.path.join(dir, sequence, "focal.txt")

        self.len = len(self.image_paths)
        self.crop = transforms.RandomCrop(512)

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        image = transforms.functional.to_dtype(torchvision.io.decode_image(self.image_paths[idx]), torch.float32, scale = True)
        # image.shape[0] == 3
        image = transforms.functional.resize(image, 512, transforms.functional.InterpolationMode.BICUBIC)
        return self.crop(image)

if __name__ == "__main__":
    dataset = nrgbd_dataset()
    print(len(dataset))
    print(dataset[0].shape)