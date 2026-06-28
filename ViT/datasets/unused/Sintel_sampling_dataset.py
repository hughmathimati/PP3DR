import torch
import torchvision
import os
import numpy as np
from tqdm import tqdm

class sintel_image_dataset(torch.utils.data.Dataset):
    """
    Random-sampling Dataset for Sintel images only.
    Notably, each batch returned by the DataLoader will have 5 dimensions, because __getitem__() here returns a tensor
    with four dimensions.
    """
    # The shortest sequence only has 20 images.
    def __init__(self, images_per_sequence=20, dir="/vulcanscratch/hughma/Pi3/data/sintel/training/final"):
        self.images_per_sequence = images_per_sequence
        self.sequences = os.listdir(dir)
        # (sequence, image #)
        self.image_paths = []
        for sequence in tqdm(self.sequences, desc="Precomputing all image file paths"):
            images = []
            sequence_dir = os.path.join(dir, sequence)
            for image in os.listdir(sequence_dir):
                images.append(os.path.join(sequence_dir, image))
            self.image_paths.append(images)
        self.num_sequences = len(self.sequences)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # I suspect I'm ending up with five dimensions because the DataLoader stacks the output tensors.
        # My solution is to assume idx is only ever a single number.
        # idx will specify which dataset we're pulling from.
        output = torch.empty((self.images_per_sequence, 3, 436, 1024))
        selected = np.random.choice(self.image_paths[idx], self.images_per_sequence, replace = False)
        for j, image_path in enumerate(selected):
            output[j] = torchvision.transforms.v2.functional.to_dtype(torchvision.io.decode_image(image_path), torch.float32, scale = True)
        return output

if __name__ == "__main__":
    dataset = sintel_image_dataset()
    print(len(dataset))
    print(dataset[0].shape)