import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial


class DatasetBase(torch.utils.data.Dataset):
    """
    Base dataset class.
    You have to write __init__(), poses_helper(), and intrinsics_helper().
    """

    def __init__(self):
        """
        Must create self.sequences, a list where each sequence corresponds to a dict with at least the following elements:
            {
                'images': Paths to the images for this sequence.
                'depths': Paths to the depths for this sequence.
            }
        """
        super().__init__()

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
                )
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
        (L, 1, H, W) tensor of the depths for this randomly-sampled sequence.
        """
        return torch.stack(
            [
                transforms.functional.to_dtype(
                    torchvision.io.decode_image(
                        self.sequences[sequence_index]['depths'][i]
                    ),
                    torch.float32,
                    scale=True
                )
                for i in frame_indices
            ]
        )

    def extrinsics_helper(self, sequence_index, frame_indices):
        """
        Parameters
        ----------
        sequence_index
        frame_indices: Randomly-sampled indices of the frames in self.sequences[sequence_index] constituting this
                       randomly-sampled sequence.
        Returns
        -------
        (L, 3, 4) tensor of the 3x4 camera extrinsic matrices for this randomly-sampled sequence.
        """
        pass

    def intrinsic_helper(self, sequence_index):
        """
        Parameters
        ----------
        sequence_index

        Returns
        -------
        (3, 3) tensor of the 3x3 camera intrinsic matrix for this randomly-sampled sequence.
        """
        pass

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        """
        Pick a random starting frame before the last 500 frames, then randomly sample 100 frames from the subsequent
        500 frames.
        We probably can't do a random crop due to the camera pose, but if we wanted, we could do rescale...
        """
        true_length = len(self.sequences[index]['images'])
        # We should never receive a sequence with only one frame...
        end_index = max(1, true_length - 500)
        num_frames = min(100, true_length)
        frame_indices = torch.randint(low=0, high=end_index, size=(num_frames,))
        frame_indices.sort()
        # Ensure no frames are more than 10 apart
        if true_length >= 120:
            for i in range(1, num_frames):
                if frame_indices[i] - frame_indices[i - 1] > 10:
                    frame_indices[i] - frame_indices[i - 1] + 10
        jobs = [
            partial(self.images_helper, index, frame_indices),
            partial(self.depths_helper, index, frame_indices),
            partial(self.extrinsics_helper, index, frame_indices),
            partial(self.intrinsic_helper, index),
        ]
        output = {}
        # ProcessPoolExecutor -> Cannot re-initialize CUDA in forked subprocess.
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(job) for job in jobs]
            # futures.as_completed returns futures in the order they complete, not their original order.
            for future in as_completed(futures):
                match futures.index(future):
                    case 0:
                        output['images'] = future.result()
                    case 1:
                        output['depths'] = future.result()
                    case 2:
                        output['extrinsics'] = future.result()
                    case 3:
                        output['intrinsic'] = future.result()
        return output
