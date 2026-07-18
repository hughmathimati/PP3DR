import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
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
    However, we sometimes only have 9 extrinsics. In such cases, we must only take 9 of those frames.
    540x960
    """
    def __init__(self,
                 images_dir = "/fs/vulcan-datasets/FlyingThings3D/frames_cleanpass/TRAIN",
                 disparities_dir = "/fs/vulcan-datasets/FlyingThings3D/disparity/TRAIN",
                 extrinsics_dir = "/vulcanscratch/hughma/data/FlyingThings3D/camera_data/TRAIN"
    ):
        super().__init__("/fs/vulcan-datasets/FlyingThings3D/frames_cleanpass/TRAIN")
        self.images_dir = images_dir
        self.disparities_dir = disparities_dir
        self.extrinsics_dir = extrinsics_dir
        # Redefine self.sequences because this dataset's folder structure is different from usual
        self.sequences = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.init_helper, scene) for scene in ["A", "B", "C"]]
            self.intrinsics = torch.tensor([
                [1050, 0, 479.5],
                [0, 1050, 269.5],
                [0, 0, 1]
            ], dtype=torch.float32).unsqueeze(0).expand(self.sequence_length, -1, -1)
            for future in futures:
                self.sequences += future.result() # Append.

    def init_helper(self, scene):
        my_sequences = []
        scene_images = os.path.join(self.images_dir, scene)
        scene_disparities = os.path.join(self.disparities_dir, scene)
        scene_extrinsics = os.path.join(self.extrinsics_dir, scene)
        for sequence in tqdm(os.listdir(scene_images), desc=f"Precomputing FlyingThings3D paths ({scene})"):
            valid_extrinsics = self.parse_camera_data(os.path.join(scene_extrinsics, sequence, "camera_data.txt"))
            seq_dict = {'images': [], 'disparities': [], 'extrinsics': []}
            # We'll just take the left views.
            images = os.path.join(scene_images, sequence, "left")
            disparities = os.path.join(scene_disparities, sequence, "left")
            for frame in os.listdir(images):
                frame = frame.split('.')[0]
                frame_num = int(frame)
                if frame_num not in valid_extrinsics:
                    continue
                seq_dict['extrinsics'].append(valid_extrinsics[frame_num])
                seq_dict['images'].append(os.path.join(images, frame) + ".png")
                seq_dict['disparities'].append(os.path.join(disparities, frame) + ".pfm")
            seq_dict['extrinsics'] = torch.stack(seq_dict['extrinsics'])
            my_sequences.append(seq_dict)
        return my_sequences

    def parse_camera_data(self, filepath):
        """
        Parses camera_data.txt and returns a dictionary mapping
        the integer frame number to its 3x4 Extrinsic Tensor (Left Camera).
        """
        valid_extrinsics = {}
        with open(filepath, 'r') as f:
            lines = f.readlines()

        current_frame = None
        for line in lines:
            line = line.strip()
            if line.startswith("Frame"):
                # e.g., "Frame 6" -> 6
                current_frame = int(line.split()[1])
            elif line.startswith("L ") and current_frame is not None:
                # Parse left camera matrix
                items = line.split(' ')[1:]

                # Load the 16 values into a 4x4 tensor, then crop to 3x4
                mat = torch.tensor([float(x) for x in items]).view(4, 4)[:3, :4]

                # Flip signs of 2nd and 3rd columns to convert OpenGL coordinates to OpenCV
                mat[:, 1] *= -1.0
                mat[:, 2] *= -1.0

                valid_extrinsics[current_frame] = mat

        return valid_extrinsics

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

    def intrinsics_helper(self, sequence_index, frame_indices):
        return self.intrinsics

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
    dataset = flying_things_3d_dataset()
    empty = []
    for i, data in tqdm(enumerate(dataset), desc="Scanning sequences..."):
        if len(data['images']) == 0:
            empty.append(i)
    print(empty)