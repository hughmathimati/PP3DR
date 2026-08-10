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
    def __init__(
            self,
            cache_path="/vulcanscratch/hughma/PP3DR/datasets/eth3d_dataset_cache.pth",
            dir="/vulcanscratch/hughma/data/eth3d/"
    ):
        super().__init__(dir)
        if os.path.exists(cache_path):
            self.images, self.depths, self.extrinsics, self.intrinsics, self.starts, self.lengths = torch.load(cache_path, weights_only=False)
            print(f"Loaded dataset from {cache_path}.")
            return

        print(f"{cache_path} not found. Initialising ETH3D dataset from scratch...")
        images, depths, extrinsics, intrinsics, starts, lengths = [], [], [], [], [], []
        futures = []
        index = 0
        with ThreadPoolExecutor() as executor:
            for i, sequence in tqdm(enumerate(self.sequence_names), desc="Precomputing ETH3D image file paths"):
                sequence_dir = os.path.join(dir, sequence)
                image_dir = os.path.join(sequence_dir, "images/custom_undistorted")
                depth_dir = os.path.join(sequence_dir, "ground_truth_depth/custom_undistorted")
                cams_dir = os.path.join(sequence_dir, "custom_undistorted_cam")
                images_list = os.listdir(image_dir)

                seq_length = len(images_list)
                starts.append(index)
                lengths.append(seq_length)

                futures.append(executor.submit(self.cams_helper, cams_dir, seq_length))
                images.extend([os.path.join(image_dir, image) for image in sorted(images_list)])
                depths.extend([os.path.join(depth_dir, depth) for depth in sorted(os.listdir(depth_dir))])

                index += seq_length

        for future in futures:
            sequence_extrinsics, sequence_intrinsics = future.result()
            extrinsics.append(sequence_extrinsics)
            intrinsics.append(sequence_intrinsics)

        self.images = np.array(images, dtype='U255') # 256-char string
        self.depths = np.array(depths, dtype='U255')
        self.extrinsics = torch.cat(extrinsics, dim=0) # (Total_Frames, 3, 4)
        self.intrinsics = torch.cat(intrinsics, dim=0) # (Total_Frames, 3, 3)
        self.starts = torch.tensor(starts, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

        torch.save([self.images, self.depths, self.extrinsics, self.intrinsics, self.starts, self.lengths], cache_path)
        print(f"Saved to {cache_path}")

    def cams_helper(self, cams_dir, seq_length):
        extrinsics = torch.empty(seq_length, 3, 4)
        intrinsics = torch.empty(seq_length, 3, 3)
        for j, cam in enumerate(os.listdir(cams_dir)):
            a = np.load(os.path.join(cams_dir, cam), 'r')
            extrinsics[j] = torch.from_numpy(np.linalg.inv(a['extrinsics'])[:3])
            intrinsics[j] = torch.from_numpy(a['intrinsics'])

        return extrinsics, intrinsics

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
                        np.fromfile(self.depths[i], dtype = np.float32).reshape(4032, 6048)
                    ),
                    torch.float32
                    # DON'T scale the depth.
                )
                for i in self.starts[sequence_index] + frame_indices
            ]
        )

if __name__ == "__main__":
    dataset = eth3d_dataset()
    empty = []
    for i, data in tqdm(enumerate(dataset), desc="Scanning sequences..."):
        if len(data['images']) == 0:
            empty.append(i)
    print(empty)