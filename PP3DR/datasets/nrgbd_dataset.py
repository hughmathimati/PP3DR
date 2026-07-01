import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

class nrgbd_dataset(torch.utils.data.Dataset):
    """
    Dataset for NRGBD images
    680 x 480
    9 sequences with around a thousand frames each. We're going to pick a random starting frame before the last
    500 frames, then randomly sample 100 frames from the subsequent 500 frames.
    """
    def __init__(self, dir="/vulcanscratch/hughma/data/nrgbd/"):
        super().__init__()
        self.sequence_names = os.listdir(dir)
        self.sequences = [dict(images = [], depths = []) for _ in range(len(self.sequence_names))]
        for i, sequence in tqdm(enumerate(self.sequence_names), desc="Precomputing NRGBD image file paths"):
            sequence_dir = os.path.join(dir, sequence)
            image_dir = os.path.join(sequence_dir, "images")
            depth_dir = os.path.join(sequence_dir, "depth")
            for image in os.listdir(image_dir):
                self.sequences[i]['images'].append(os.path.join(image_dir, image))
            for depth in os.listdir(depth_dir):
                self.sequences[i]['depths'].append(os.path.join(depth_dir, depth))
            self.sequences[i]['poses'] = os.path.join(dir, sequence, "poses.txt")
            self.sequences[i]['focal'] = os.path.join(dir, sequence, "focal.txt")

        self.len = len(self.sequences)
        self.crop = transforms.RandomCrop(512)

    def __len__(self):
        return len(self.sequences)

    def images_helper(self, sequence_index, frame_indices):
        return torch.stack(
            [
                transforms.functional.to_dtype(
                    torchvision.io.decode_image(
                        self.sequences[sequence_index]['images'][i]
                    ),
                    torch.float32,
                    scale = True
                )
                for i in frame_indices
            ]
        )
        return torch.tensor(output)

    def depths_helper(self, sequence_index, frame_indices):
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

    def poses_helper(self, sequence_index, frame_indices):
        poses = torch.empty(len(frame_indices), 4, 4)
        with open(self.sequences[sequence_index]['poses']) as f:
            lines = f.readlines()
            for i, frame_index in enumerate(frame_indices):
                for j in range(4):
                    floats = lines[frame_index * 4 + j].split(' ')
                    for k in range(4):
                        poses[i, j, k] = float(floats[k])
        return poses


    def focal_helper(self, sequence_index):
        with open(self.sequences[sequence_index]['focal']) as f:
            focal = float(next(f))
        return focal

    def __getitem__(self, index):
        """
        Pick a random starting frame before the last 500 frames, then randomly sample 100 frames from the subsequent
        500 frames.
        We probably can't do a random crop due to the camera pose, but if we wanted, we could do rescale...
        """
        end_index = max(1, len(self.sequences[index]['images']) - 500)
        num_frames = min(100, len(self.sequences[index]['images']))
        frame_indices = torch.randint(low = 0, high = end_index, size = (num_frames,))
        frame_indices.sort()
        # Ensure no frames are more than 10 apart
        for i in range(1, num_frames):
            if frame_indices[i] - frame_indices[i - 1] > 10:
                frame_indices[i] - frame_indices[i - 1] + 10
        jobs = [
            partial(self.images_helper, index, frame_indices),
            partial(self.depths_helper, index, frame_indices),
            partial(self.poses_helper, index, frame_indices),
            partial(self.focal_helper, index),
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
                        output['poses'] = future.result()
                    case 3:
                        output['focal'] = future.result()

        return output

if __name__ == "__main__":
    dataset = nrgbd_dataset()
    print(len(dataset))
    first = dataset[0]
    print(first['images'].shape)
    print(first['depths'].shape)
    print(first['poses'].shape)
    print(first)