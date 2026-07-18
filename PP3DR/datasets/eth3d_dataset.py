import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
try:
    from .dataset_base import DatasetBase
except:
    from dataset_base import DatasetBase

class eth3d_dataset(DatasetBase):
    """
    13 sequences, each with around 38 frames.
    6048 x 4032
    """
    # The shortest sequence only has 20 images.
    # Because the extrinsic and intrinsic matrices are stored together for each frame's file path, I'm going to
    # pre-load all extrinsic/intrinsic matrices.
    def __init__(self, dir="/vulcanscratch/hughma/data/eth3d/"):
        super().__init__(dir)
        with ThreadPoolExecutor() as executor:
            for i, sequence in tqdm(enumerate(self.sequence_names), desc="Precomputing ETH3D image file paths"):
                sequence_dir = os.path.join(dir, sequence)
                image_dir = os.path.join(sequence_dir, "images/custom_undistorted")
                depth_dir = os.path.join(sequence_dir, "ground_truth_depth/custom_undistorted")
                cams_dir = os.path.join(sequence_dir, "custom_undistorted_cam")
                images_list = os.listdir(image_dir)
                seq_length = len(images_list)
                self.sequences[i]['extrinsics'] = torch.empty(seq_length, 3, 4)
                self.sequences[i]['intrinsics'] = torch.empty(seq_length, 3, 3)

                executor.submit(self.cams_helper, cams_dir, i)
                for image in images_list:
                    self.sequences[i]['images'].append(os.path.join(image_dir, image))
                for depth in os.listdir(depth_dir):
                    self.sequences[i]['depths'].append(os.path.join(depth_dir, depth))
            # for image in self.sequences[0]['images']:
            #     print(image)
            # for depth in self.sequences[0]['depths']:
            #     print(depth)

    def cams_helper(self, cams_dir, sequence_index):
        for j, cam in enumerate(os.listdir(cams_dir)):
            a = np.load(os.path.join(cams_dir, cam), 'r')
            self.sequences[sequence_index]['intrinsics'][j] = torch.from_numpy(a['intrinsics'])
            self.sequences[sequence_index]['extrinsics'][j] = torch.from_numpy(np.linalg.inv(a['extrinsics'])[:3])

    def depths_helper(self, sequence_index, frame_indices):
        """
        ETH3D apparently stores depths as dense binary files with an erroneous JPG file extension.

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
                transforms.functional.to_dtype(
                    torch.from_numpy(
                        # You must specify the data type explicitly, because it's a binary file.
                        np.fromfile(self.sequences[sequence_index]['depths'][i], dtype = np.float32).reshape(4032, 6048)
                    ),
                    torch.float32
                    # DON'T scale the depth.
                )
                for i in frame_indices
            ]
        )

if __name__ == "__main__":
    dataset = eth3d_dataset()
    empty = []
    for i, data in tqdm(enumerate(dataset), desc="Scanning sequences..."):
        if len(data['images']) == 0:
            empty.append(i)
    print(empty)