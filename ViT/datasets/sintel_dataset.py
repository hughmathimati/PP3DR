import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
from tqdm import tqdm

class sintel_dataset(torch.utils.data.Dataset):
    """
    Dataset for Sintel images only.

    1064 images in total.
    """
    def __init__(self, dir="/vulcanscratch/hughma/data/sintel/training/final"):
        super().__init__()
        self.sequences = os.listdir(dir)
        self.image_paths = []
        for sequence in tqdm(self.sequences, desc="Precomputing Sintel image file paths"):
            sequence_dir = os.path.join(dir, sequence)
            for image in os.listdir(sequence_dir):
                self.image_paths.append(os.path.join(sequence_dir, image))
        self.len = len(self.image_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return transforms.functional.to_dtype(torchvision.io.decode_image(self.image_paths[idx]), torch.float32, scale = True)

if __name__ == "__main__":
    dataset = sintel_dataset()
    print(len(dataset))
    print(dataset[0].shape)