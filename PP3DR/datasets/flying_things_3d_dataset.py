import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import re
try:
    from .dataset_base import DatasetBase
except:
    from dataset_base import DatasetBase

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode("ascii")
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception(f'Not a PFM file. Header was: {header}')

    dim_line = file.readline().decode('ascii')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', dim_line)
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM dimensions.')

    scale = float(file.readline().rstrip().decode("ascii"))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    # .copy() is basically .contiguous(), since np.flipud actually just sets a negative stride (-_-)
    data = np.flipud(data).copy()
    return data, scale

class flying_things_3d_dataset(DatasetBase):
    """
    2239 sequences in total, each with 10 frames.
    540x960
    """
    # The shortest sequence only has 20 images.
    def __init__(self,
                 images_dir = "/fs/vulcan-datasets/FlyingThings3D/frames_cleanpass/TRAIN",
                 disparities_dir = "/fs/vulcan-datasets/FlyingThings3D/disparity/TRAIN",
                 extrinsics_dir = "/vulcanscratch/hughma/data/FlyingThings3D/camera_data/TRAIN"
    ):
        super().__init__("/fs/vulcan-datasets/FlyingThings3D/frames_cleanpass/TRAIN")
        # Redefine self.sequences because this dataset's folder structure is different from usual
        self.sequences = [dict(images=[], depths=[], disparities=[], extrinsics=[]) for _ in range(746 + 746 + 747)]
        i = 0
        for scene in ("A", "B", "C"):
            scene_images = os.path.join(images_dir, scene)
            scene_disparities = os.path.join(disparities_dir, scene)
            scene_extrinsics = os.path.join(extrinsics_dir, scene)
            for sequence in tqdm(os.listdir(scene_images), desc=f"Precomputing FlyingThings3D image file paths ({scene})"):
                # We'll just take the left views for now.
                images = os.path.join(scene_images, sequence, "left")
                for image in os.listdir(images):
                    self.sequences[i]['images'].append(os.path.join(images, image))

                disparities = os.path.join(scene_disparities, sequence, "left")
                for disparity in os.listdir(disparities):
                    self.sequences[i]['disparities'].append(os.path.join(disparities, disparity))

                self.sequences[i]['extrinsics'] = os.path.join(scene_extrinsics, sequence, "camera_data.txt")
                i += 1

    def depths_helper(self, sequence_index, frame_indices):
        """
        flying_things_3d_dataset overrides this function because FlyingThings3D provides disparities, not depths.

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
        # depth = depth = focallength * baseline / disparity. The focal length (in pixels, not mm) here is 1050, and the
        # baseline is 1, as mentioned on the dataset's webpage.
        return 1050 / torch.stack(
            [
                transforms.functional.to_dtype(
                    torch.from_numpy(
                        # Discard the second element of the returned tuple, which is the "scale" (not used).
                        readPFM(self.sequences[sequence_index]['disparities'][i])[0]
                    ),
                    torch.float32
                    # DON'T scale the depth.
                )
                for i in frame_indices
            ]
        )

    def extrinsics_helper(self, sequence_index, frame_indices):
        extrinsics = torch.empty(len(frame_indices), 3, 4)
        with open(self.sequences[sequence_index]['extrinsics'], 'r') as f:
            lines = f.readlines()
            for i, frame_index in enumerate(frame_indices):
                # First character is 'L'.
                items = lines[1 + 4 * frame_index].split(' ')[1:]
                row, col, j = 0, 0, 0
                while row < 3:
                    # Flip signs of 2nd and 3rd columns to convert OpenGL coordinates to OpenCV
                    if col in [1, 2]:
                        extrinsics[i, row, col] = -float(items[j])
                    else:
                        extrinsics[i, row, col] = float(items[j])
                    j += 1
                    col += 1
                    if col == 4:
                        row, col = row + 1, 0
        return extrinsics

    def intrinsics_helper(self, sequence_index, frame_indices):
        return torch.tensor([
            [1050,  0,      479.5],
            [0,     1050,   269.5],
            [0,     0,      1]
        ],
        dtype = torch.float32).unsqueeze(0).expand(len(frame_indices), -1, -1)

# 1. Define a top-level function so it can be pickled by multiprocessing
def verify_index(dataset, idx):
    try:
        # The worker accesses the globally scoped 'dataset'
        # and performs the heavy I/O on its own CPU core.
        _ = dataset[idx]
        return None
    except Exception:
        # Catch the exception locally so the ProcessPool doesn't crash
        return dataset.image_paths[idx]

if __name__ == "__main__":
    from functools import partial
    dataset = flying_things_3d_dataset()
    first = dataset[0]
    for key in first:
        print(key)
        print(first[key].shape)
    print(first['depths'].min(), first['depths'].max())
    # invalid_files = []
    # with ProcessPoolExecutor() as executor:
    #     results = executor.map(partial(verify_index, dataset), range(total_files), chunksize=64)
    #     with tqdm(total=total_files, desc="Processing", unit="file", mininterval = 1) as pbar:
    #         for result in results:
    #             if result is not None:
    #                 invalid_files.append(result)
    #             pbar.update(1)
    #
    # print(f"\nScan complete. Found {len(invalid_files)} invalid images.")
    # if invalid_files:
    #     with open("invalid_flying_things_3d_files.txt", "w") as f:
    #         for path in invalid_files:
    #             f.write(f"{path}\n")