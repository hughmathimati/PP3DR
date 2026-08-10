import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datasets.dataset_base import DatasetBase
from datasets.sintel.sintel_io import depth_read, cam_read
import numpy as np


class sintel_dataset(DatasetBase):
    """
    23 sequences, each with between 20 to 50 frames each.
    1024x436
    """

    def __init__(
            self,
            cache_path="/vulcanscratch/hughma/PP3DR/datasets/sintel/sintel_dataset_cache.pth",
            dir="/vulcanscratch/hughma/data/sintel/training/"
    ):
        super().__init__(dir)
        if os.path.exists(cache_path):
            self.images, self.depths, self.extrinsics, self.intrinsics, self.starts, self.lengths = torch.load(cache_path, weights_only=False)
            print(f"Loaded dataset from {cache_path}.")
            return

        print(f"{cache_path} not found. Initialising Sintel dataset from scratch...")
        images, depths, extrinsics, intrinsics, starts, lengths = [], [], [], [], [], []
        futures = []
        index = 0
        # Because sintel_io.cam_read outputs both the extrinsic and intrinsic matrices for each frame's file path, I'm going to
        # pre-load all extrinsic/intrinsic matrices.
        images_dir = os.path.join(dir, "final")
        depths_dir = os.path.join(dir, "depth")
        cams_dir = os.path.join(dir, "camdata_left")
        # Redefine self.sequences because this dataset's folder structure is different from usual
        with ThreadPoolExecutor() as executor:
            for i, sequence in tqdm(enumerate(os.listdir(images_dir))):
                _images = os.path.join(images_dir, sequence)
                _depths = os.path.join(depths_dir, sequence)
                _cams = os.path.join(cams_dir, sequence)
                images_list = os.listdir(_images)

                seq_length = len(images_list)
                starts.append(index)
                lengths.append(seq_length)

                futures.append(executor.submit(self.cams_helper, _cams, seq_length))
                images.extend([os.path.join(_images, image) for image in sorted(images_list)])
                depths.extend([os.path.join(_depths, depth) for depth in sorted(os.listdir(_depths))])

                index += seq_length

        for future in futures:
            sequence_extrinsics, sequence_intrinsics = future.result()
            extrinsics.append(sequence_extrinsics)
            intrinsics.append(sequence_intrinsics)

        self.images = np.array(images, dtype='U255')  # 256-char string
        self.depths = np.array(depths, dtype='U255')
        self.extrinsics = torch.cat(extrinsics, dim=0)  # (Total_Frames, 3, 4)
        self.intrinsics = torch.cat(intrinsics, dim=0)  # (Total_Frames, 3, 3)
        self.starts = torch.tensor(starts, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

        torch.save((self.images, self.depths, self.extrinsics, self.intrinsics, self.starts, self.lengths), cache_path)
        print(f"Saved to {cache_path}")

    def cams_helper(self, cams, seq_length):
        extrinsics = torch.empty(seq_length, 3, 4)
        intrinsics = torch.empty(seq_length, 3, 3)

        # We must ensure the directory is sorted!
        for j, cam in enumerate(sorted(os.listdir(cams))):
            intrinsic, extrinsic = cam_read(os.path.join(cams, cam))
            intrinsics[j] = torch.from_numpy(intrinsic).float()

            # Sintel's `extrinsic` matrix is World-to-Camera (W2C).
            # Our pipeline strictly expects Camera-to-World (C2W).
            W2C_gl = extrinsic  # 4x4 or 3x4 W2C matrix
            R_w2c = W2C_gl[:3, :3]
            t_w2c = W2C_gl[:3, 3]

            # 1. Invert the W2C matrix to get the C2W matrix
            R_c2w = R_w2c.T
            t_c2w = -R_w2c.T @ t_w2c

            # Create a full 4x4 C2W matrix in OpenGL format
            C2W_gl = np.eye(4)
            C2W_gl[:3, :3] = R_c2w
            C2W_gl[:3, 3] = t_c2w

            # 2. Convert from OpenGL (+X right, +Y up, -Z forward)
            # to OpenCV (+X right, +Y down, +Z forward)
            C2W_cv = C2W_gl.copy()

            extrinsics[j] = torch.from_numpy(C2W_cv[:3, :4]).float()

        return extrinsics, intrinsics

    def depths_helper(self, sequence_index, frame_indices):
        """
        Sintel uses a binary format for its depth files, rather than a PNG image.

        Parameters
        ----------
        sequence_index
        frame_indices: Randomly-sampled indices of the frames in self.sequences[sequence_index] constituting this
                       randomly-sampled sequence.
        Returns
        -------
        (L, H, W) tensor of the depths for this randomly-sampled sequence.
        """
        return torch.stack([
            torch.from_numpy(depth_read(self.depths[i]))
            for i in self.starts[sequence_index] + frame_indices
        ])