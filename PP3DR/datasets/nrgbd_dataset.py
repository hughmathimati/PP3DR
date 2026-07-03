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


class nrgbd_dataset(DatasetBase):
    """
    Dataset for NRGBD images
    640 x 480
    9 sequences with around a thousand frames each. We're going to pick a random starting frame before the last
    500 frames, then randomly sample 100 frames from the subsequent 500 frames.
    """

    def __init__(self, dir="/vulcanscratch/hughma/data/nrgbd/"):
        super().__init__()
        self.sequence_names = os.listdir(dir)
        self.sequences = [dict(images=[], depths=[]) for _ in range(len(self.sequence_names))]
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

    def extrinsics_helper(self, sequence_index, frame_indices):
        """
        NRGBD's poses.txt file provides a 4x4 extrinsic matrix per frame. We're discarding the bottom row, as it's just
        [0 0 0 1].
        Additionally, it allegedly uses the OpenGL coordinate convention, as opposed to the OpenCV coordinate convention,
        meaning I have to flip the signs of the second and third columns (allegedly).
        """
        poses = torch.empty(len(frame_indices), 3, 4)
        with open(self.sequences[sequence_index]['poses']) as f:
            lines = f.readlines()
            for i, frame_index in enumerate(frame_indices):
                for j in range(3):
                    floats = lines[frame_index * 4 + j].split(' ')
                    # First column of rotation matrix is unchanged
                    poses[i, j, 0] = float(floats[0])
                    # Second and third columns of rotation matrix have their signs flipped
                    poses[i, j, 1] = -float(floats[1])
                    poses[i, j, 2] = -float(floats[2])
                    # Translation column of extrinsic matrix is unchanged
                    poses[i, j, 3] = float(floats[3])
        return poses

    def intrinsic_helper(self, sequence_index):
        H, W = 480, 640
        with open(self.sequences[sequence_index]['focal']) as f:
            focal = float(next(f))
        intrinsic = torch.zeros(3, 3)
        intrinsic[0, 0] = intrinsic[1, 1] = focal
        intrinsic[0, 2], intrinsic[1, 2] = W // 2, H // 2
        intrinsic[2, 2] = 1
        return intrinsic


if __name__ == "__main__":
    import cv2

    dataset = nrgbd_dataset()
    print(len(dataset))
    first = dataset[0]
    for key in first:
        print(key)
        print(first[key].shape)
    # I need to know what kind of depth they're using.
    # depth = first['depths'][0]
    # dmin, dmax = depth.min(), depth.max()
    # depth = (255 * (depth - dmin) / (dmax - dmin)).permute(1, 2, 0)
    # print(depth.shape)
    # cv2.imwrite("test_nrgbd_depth.png", depth.detach().numpy(force = True).astype(np.uint8))
