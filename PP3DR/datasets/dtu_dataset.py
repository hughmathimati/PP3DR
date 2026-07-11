import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
try:
    from .dataset_base import DatasetBase
except:
    from dataset_base import DatasetBase

class dtu_dataset(DatasetBase):
    """
    22 sequences, each with 48 frames.
    1600 x 1200
    W2C extrinsics with OpenCV coordinate convention.
    """
    # The shortest sequence only has 20 images.
    def __init__(self, dir="/vulcanscratch/hughma/data/dtu/"):
        super().__init__(dir)
        for sequence in self.sequences:
            sequence['cams'] = []
        for i, sequence in tqdm(enumerate(self.sequence_names), desc="Precomputing DTU image file paths"):
            sequence_dir = os.path.join(dir, sequence)
            image_dir = os.path.join(sequence_dir, "images")
            depth_dir = os.path.join(sequence_dir, "depths")
            cams_dir = os.path.join(sequence_dir, "cams")

            for image in os.listdir(image_dir):
                self.sequences[i]['images'].append(os.path.join(image_dir, image))
            for depth in os.listdir(depth_dir):
                self.sequences[i]['depths'].append(os.path.join(depth_dir, depth))
            for cam in os.listdir(cams_dir):
                self.sequences[i]['cams'].append(os.path.join(cams_dir, cam))

    def depths_helper(self, sequence_index, frame_indices):
        """
        DTU uses .npy for its depth files, rather than a PNG image.

        Parameters
        ----------
        sequence_index
        frame_indices: Randomly-sampled indices of the frames in self.sequences[sequence_index] constituting this
                       randomly-sampled sequence.
        Returns
        -------
        (L, H, W) tensor of the depths for this randomly-sampled sequence.
        """
        # Here we want to use torch.stack instead of torch.cat because the inner array is of shape (H, W) instead of
        # (1, H, W).
        return torch.stack(
            [
                torch.from_numpy(
                    np.load(self.sequences[sequence_index]['depths'][i])
                )
                for i in frame_indices
            ]
        )

    def extrinsics_helper(self, sequence_index, frame_indices):
        # We construct the extrinsics matrix as 4x4 so we can then invert it.
        extrinsics = torch.empty(len(frame_indices), 4, 4)
        extrinsics[..., 3, 3] = 1
        for i, frame_index in enumerate(frame_indices):
            with open(self.sequences[sequence_index]['cams'][frame_index], 'r') as f:
                lines = f.readlines()
                for j in range(3):
                    floats = lines[1 + j].split(' ')
                    for k in range(4):
                        extrinsics[i, j, k] = float(floats[k])
        return torch.linalg.inv(extrinsics)[:, :3]

    def intrinsics_helper(self, sequence_index, frame_indices):
        intrinsics = torch.empty(len(frame_indices), 3, 3)
        for i, frame_index in enumerate(frame_indices):
            with open(self.sequences[sequence_index]['cams'][frame_index], 'r') as f:
                lines = f.readlines()
                for j in range(3):
                    floats = lines[7 + j].split(' ')
                    for k in range(3):
                        intrinsics[i, j, k] = float(floats[k])
        return intrinsics

if __name__ == "__main__":
    dataset = dtu_dataset()
    print(len(dataset))
    first = dataset[0]
    for key in first:
        print(key)
        print(first['depths'].min(), first['depths'].max())
        print(first[key].shape)
    print(first['depths'].min(), first['depths'].max())
