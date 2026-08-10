import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re
from datasets.dataset_base import DatasetBase

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
                 cache_path="/vulcanscratch/hughma/PP3DR/datasets/flying_things_3d/flying_things_3d_dataset_cache.pth",
                 images_dir="/fs/vulcan-datasets/FlyingThings3D/frames_cleanpass/TRAIN",
                 disparities_dir="/fs/vulcan-datasets/FlyingThings3D/disparity/TRAIN",
                 extrinsics_dir="/vulcanscratch/hughma/data/FlyingThings3D/camera_data/TRAIN"
                 ):
        super().__init__("/fs/vulcan-datasets/FlyingThings3D/frames_cleanpass/TRAIN")

        # We need to init this regardless of whether we load the cache or not.
        self.intrinsics = torch.tensor([
            [1050, 0, 479.5],
            [0, 1050, 269.5],
            [0, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).expand(self.sequence_length, -1, -1)

        if os.path.exists(cache_path):
            cache = torch.load(cache_path, weights_only=False)
            self.images = cache['images']
            self.disparities = cache['disparities']
            self.extrinsics = cache['extrinsics']
            self.starts = cache['starts']
            self.lengths = cache['lengths']
            print(f"Loaded dataset from {cache_path}.")
            return

        print(f"{cache_path} not found. Initialising FlyingThings3D dataset from scratch...")

        # 1. Initialize temporary flat lists
        temp_images = []
        temp_disparities = []
        temp_extrinsics = []
        temp_starts = []
        temp_lengths = []

        # 2. Gather all tasks (Scene + Sequence combinations)
        tasks = []
        for scene in ["A", "B", "C"]:
            scene_images = os.path.join(images_dir, scene)
            if not os.path.exists(scene_images): continue

            for sequence in os.listdir(scene_images):
                tasks.append((scene, sequence, images_dir, disparities_dir, extrinsics_dir))

        current_idx = 0

        # 3. Process all sequences concurrently
        # Network drives have high latency but massive bandwidth.
        # 32 threads allows us to parse 32 folders at the exact same time.
        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(tqdm(
                executor.map(self._process_sequence, tasks),
                total=len(tasks),
                desc="Precomputing FlyingThings3D paths"
            ))

        # 4. Sequentially merge the thread results into our flat arrays
        for res in results:
            if res is None: continue

            img_list, disp_list, ext_list = res
            seq_len = len(img_list)

            if seq_len == 0: continue

            # Record the boundaries
            temp_starts.append(current_idx)
            temp_lengths.append(seq_len)

            # Extend flat lists
            temp_images.extend(img_list)
            temp_disparities.extend(disp_list)
            temp_extrinsics.extend(ext_list)

            current_idx += seq_len

        # =========================================================================
        # THE ZERO-COPY TRANSFORMATION
        # =========================================================================
        # Force NumPy to allocate fixed-length C-strings (max 255 chars)
        self.images = np.array(temp_images, dtype='U255')
        self.disparities = np.array(temp_disparities, dtype='U255')
        self.extrinsics = torch.stack(temp_extrinsics)
        self.starts = torch.tensor(temp_starts, dtype=torch.long)
        self.lengths = torch.tensor(temp_lengths, dtype=torch.long)

        self.sequence_length = len(self.starts)

        torch.save({
            'images': self.images,
            'disparities': self.disparities,
            'extrinsics': self.extrinsics,
            'starts': self.starts,
            'lengths': self.lengths
        }, cache_path)
        print(f"Saved to {cache_path}")

    def _process_sequence(self, args):
        """
        Background worker function.
        Reads files and returns local lists to prevent thread locking.
        """
        scene, sequence, images_dir, disparities_dir, extrinsics_dir = args

        scene_images = os.path.join(images_dir, scene)
        scene_disparities = os.path.join(disparities_dir, scene)
        scene_extrinsics = os.path.join(extrinsics_dir, scene)

        camera_data_path = os.path.join(scene_extrinsics, sequence, "camera_data.txt")
        if not os.path.exists(camera_data_path):
            return None

        valid_extrinsics = self.parse_camera_data(camera_data_path)

        img_dir = os.path.join(scene_images, sequence, "left")
        disp_dir = os.path.join(scene_disparities, sequence, "left")

        if not os.path.exists(img_dir) or not os.path.exists(disp_dir):
            return None

        img_list, disp_list, ext_list = [], [], []

        # Make sure we sort the list so frames align perfectly!
        for frame in sorted(os.listdir(img_dir)):
            frame_name = frame.split('.')[0]
            frame_num = int(frame_name)

            if frame_num not in valid_extrinsics:
                continue

            img_list.append(os.path.join(img_dir, frame))
            # PFMs have the exact same name but end in .pfm
            disp_list.append(os.path.join(disp_dir, frame_name + ".pfm"))
            ext_list.append(valid_extrinsics[frame_num])

        if len(img_list) > 0:
            return img_list, disp_list, ext_list

        return None

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
                        readPFM(self.disparities[i])[0]
                    ),
                    torch.float32
                    # DON'T scale the depth.
                )
                for i in self.starts[sequence_index] + frame_indices
            ]
        )

    def intrinsics_helper(self, sequence_index, frame_indices):
        return self.intrinsics