import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
from tqdm import tqdm

class scannet_dataset(torch.utils.data.Dataset):
    """
    Dataset for Scannetv2 images only.

    104810 images in total.
    """
    def __init__(self, dir="/vulcanscratch/hughma/data/scannetv2/"):
        self.sequences = os.listdir(dir)
        self.image_paths = []
        for sequence in tqdm(self.sequences, desc="Precomputing all image file paths"):
            sequence_dir = os.path.join(dir, sequence, "color")
            for image in os.listdir(sequence_dir):
                self.image_paths.append(os.path.join(sequence_dir, image))
        self.len = len(self.image_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return transforms.functional.to_dtype(torchvision.io.decode_image(self.image_paths[idx]), torch.float32, scale = True)

if __name__ == "__main__":
    dataset = scannet_dataset()
    print(len(dataset))
    print(dataset[0].shape)