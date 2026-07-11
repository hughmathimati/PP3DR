import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import cv2
cv2.setNumThreads(0)


class DatasetBase(torch.utils.data.Dataset):
    """
    Base dataset class.

    You have to write __init__(), extrinsics_helper(), and intrinsics_helper().

    images_helper() and depths_helper() are optional overrides. Overriding depths_helper() is much more common.

    __getitem__() is intentionally implemented here, to be inherited by all children classes.
    """

    def __init__(self, dir):
        """
        Must create self.sequences, a list where each sequence corresponds to a dict with at least the following elements:
            {
                'images': Paths to the images for this sequence.
                'depths': Paths to the depths for this sequence.
            }
        """
        super().__init__()
        self.max_sequence_length = 10
        self.sequence_names = os.listdir(dir)
        self.sequences = [dict(images=[], depths=[]) for _ in range(len(self.sequence_names))]

    def images_helper(self, sequence_index, frame_indices):
        """
        Parameters
        ----------
        sequence_index
        frame_indices: Randomly-sampled indices of the frames in self.sequences[sequence_index] constituting this
                       randomly-sampled sequence.
        Returns
        -------
        (L, 3, H, W) tensor of the images for this randomly-sampled sequence.
        """
        return torch.stack(
            [
                transforms.functional.to_dtype(
                    torchvision.io.decode_image(
                        self.sequences[sequence_index]['images'][i]
                    ),
                    torch.float32,
                    scale=True
                )[:3] # I'm including this here for the RGBA datasets.
                for i in frame_indices
            ]
        )

    def depths_helper(self, sequence_index, frame_indices):
        """
        Parameters
        ----------
        sequence_index
        frame_indices: Randomly-sampled indices of the frames in self.sequences[sequence_index] constituting this
                       randomly-sampled sequence.
        Returns
        -------
        (L, H, W) tensor of the depths for this randomly-sampled sequence.
        """
        # We're using torch.cat here instead of torch.stack because we want to get rid of the singleton channel dimension.
        return torch.cat(
            [
                transforms.functional.to_dtype(
                    torchvision.io.decode_image(
                        self.sequences[sequence_index]['depths'][i]
                    ),
                    torch.float32
                    # DON'T scale the depth.
                )
                for i in frame_indices
            ]
        )

    def extrinsics_helper(self, sequence_index, frame_indices):
        """
        If you choose not to override this function, it will assume you have an (L, 4, 3) tensor for each
        self.sequences[sequence_index]['extrinsic'].

        Parameters
        ----------
        sequence_index
        frame_indices: Randomly-sampled indices of the frames in self.sequences[sequence_index] constituting this
                       randomly-sampled sequence.
        Returns
        -------
        (L, 3, 4) tensor of the 3x4 camera extrinsic matrices for this randomly-sampled sequence.
        """
        return self.sequences[sequence_index]['extrinsics'][frame_indices]

    def intrinsics_helper(self, sequence_index, frame_indices):
        """
        If you choose not to override this function, it will assume you have an (L, 4, 3) tensor for each
        self.sequences[sequence_index]['intrinsic'].

        Parameters
        ----------
        sequence_index

        Returns
        -------
        (L, 3, 3) tensor of the 3x3 camera intrinsic matrix for this randomly-sampled sequence.
        """
        return self.sequences[sequence_index]['intrinsics'][frame_indices]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        """
        Pick a random starting frame before the last 3 * x frames, then randomly sample x frames from the subsequent
        3 * x frames.
        We probably can't do a random crop due to the camera pose, but if we wanted, we could do rescale...
        """
        true_length = len(self.sequences[index]['images'])
        num_frames = min(self.max_sequence_length, true_length)
        window_size = 3 * self.max_sequence_length
        if true_length < window_size:
            start = 0
        else:
            max_start = max(0, true_length - window_size)
            start = torch.randint(low=0, high=max_start, size=())
        frame_indices = torch.randint(low = start, high = min(start + window_size, true_length), size=(num_frames,))
        frame_indices = frame_indices.sort().values
        print(frame_indices)
        # Ensure no frames are more than 10 apart
        if true_length >= self.max_sequence_length + 10:
            for i in range(1, num_frames):
                if frame_indices[i] - frame_indices[i - 1] > 10:
                    frame_indices[i] - frame_indices[i - 1] + 10
        output = {}
        # ProcessPoolExecutor -> Cannot re-initialize CUDA in forked subprocess.
        with ThreadPoolExecutor() as executor:
            a = executor.submit(partial(self.images_helper, index, frame_indices))
            b = executor.submit(partial(self.depths_helper, index, frame_indices))
            c = executor.submit(partial(self.extrinsics_helper, index, frame_indices))
            d = executor.submit(partial(self.intrinsics_helper, index, frame_indices))
            output['images'] = a.result()
            output['depths'] = b.result()
            output['extrinsics'] = c.result()
            output['intrinsics'] = d.result()
        return output