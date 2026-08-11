import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

try:
    from .dataset_base import DatasetBase
except:
    from dataset_base import DatasetBase


class nrgbd_dataset(DatasetBase):
    """
    Dataset for NRGBD images
    640 x 480
    9 sequences with around a thousand frames each.
    C2W extrinsics with OpenGL coordinate convention.
    """

    def __init__(
            self,
            cache_path="/vulcanscratch/hughma/PP3DR/datasets/nrgbd_dataset_cache.pth",
            dir="/vulcanscratch/hughma/data/nrgbd/"
    ):
        super().__init__(dir)

        # Load from cache if available (safely allowing NumPy arrays)
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, weights_only=False)
            self.images = cache['images']
            self.depths = cache['depths']
            self.extrinsics = cache['extrinsics']
            self.intrinsics = cache['intrinsics']
            self.starts = cache['starts']
            self.lengths = cache['lengths']
            print(f"Loaded dataset from {cache_path}.")
            return

        print(f"{cache_path} not found. Initialising NRGBD dataset from scratch...")

        temp_images = []
        temp_depths = []
        temp_extrinsics = []
        temp_intrinsics = []
        temp_starts = []
        temp_lengths = []

        # Build tasks for multithreaded parsing
        tasks = [(seq_name, dir) for seq_name in self.sequence_names]
        current_idx = 0

        # Use ThreadPool to process all 9 sequences concurrently
        with ThreadPoolExecutor(max_workers=9) as executor:
            results = list(tqdm(
                executor.map(self._process_sequence, tasks),
                total=len(tasks),
                desc="Parsing NRGBD sequences and text files"
            ))

        # Sequentially flatten the thread results to preserve deterministic ordering
        for res in results:
            if res is None: continue

            img_paths, depth_paths, ext_tensor, int_tensor = res
            seq_len = len(img_paths)

            if seq_len == 0: continue

            # Record boundaries
            temp_starts.append(current_idx)
            temp_lengths.append(seq_len)

            # Extend lists
            temp_images.extend(img_paths)
            temp_depths.extend(depth_paths)
            temp_extrinsics.append(ext_tensor)
            temp_intrinsics.append(int_tensor)

            current_idx += seq_len

        # =========================================================================
        # THE ZERO-COPY TRANSFORMATION
        # =========================================================================
        self.images = np.array(temp_images, dtype='U255')
        self.depths = np.array(temp_depths, dtype='U255')

        self.extrinsics = torch.cat(temp_extrinsics, dim=0)  # (Total_Frames, 3, 4)
        self.intrinsics = torch.cat(temp_intrinsics, dim=0)  # (Total_Frames, 3, 3)

        # Safely use int64 (long) to prevent overflow traps
        self.starts = torch.tensor(temp_starts, dtype=torch.long)
        self.lengths = torch.tensor(temp_lengths, dtype=torch.long)

        torch.save({
            'images': self.images,
            'depths': self.depths,
            'extrinsics': self.extrinsics,
            'intrinsics': self.intrinsics,
            'starts': self.starts,
            'lengths': self.lengths
        }, cache_path)

        print(f"Saved to {cache_path}")

    def _process_sequence(self, args):
        """
        Background worker function to isolate folder crawling and text parsing.
        """
        sequence_name, base_dir = args
        sequence_dir = os.path.join(base_dir, sequence_name)

        image_dir = os.path.join(sequence_dir, "images")
        depth_dir = os.path.join(sequence_dir, "depth")
        poses_file = os.path.join(sequence_dir, "poses.txt")
        focal_file = os.path.join(sequence_dir, "focal.txt")

        if not os.path.exists(poses_file) or not os.path.exists(focal_file):
            return None

        # 1. Parse and sort images
        image_files = sorted(
            [f for f in os.listdir(image_dir) if f.startswith('img')],
            key=lambda x: int(x.split('.')[0][3:])
        )
        depth_files = sorted(
            [f for f in os.listdir(depth_dir) if f.startswith('depth')],
            key=lambda x: int(x.split('.')[0][5:])
        )

        img_paths = [os.path.join(image_dir, f) for f in image_files]
        depth_paths = [os.path.join(depth_dir, f) for f in depth_files]

        seq_length = min(len(img_paths), len(depth_paths))
        if seq_length == 0:
            return None

        # 2. Parse Intrinsics (focal.txt)
        H, W = 480, 640
        with open(focal_file, 'r') as f:
            focal = float(next(f).strip())

        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = W / 2.0
        K[1, 2] = H / 2.0

        intrinsics_tensor = K.unsqueeze(0).expand(seq_length, -1, -1)

        # 3. Parse Extrinsics (poses.txt)
        extrinsics_tensor = torch.empty(seq_length, 3, 4, dtype=torch.float32)

        with open(poses_file, 'r') as f:
            # Drop empty lines safely
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Extract the matrices (4 lines per frame, we only need the top 3 rows)
        for i in range(seq_length):
            for j in range(3):
                floats = list(map(float, lines[i * 4 + j].split()))
                extrinsics_tensor[i, j, 0] = floats[0]

                # Convert OpenGL to OpenCV (Flip Y and Z axes)
                extrinsics_tensor[i, j, 1] = -floats[1]
                extrinsics_tensor[i, j, 2] = -floats[2]

                extrinsics_tensor[i, j, 3] = floats[3]

        return img_paths[:seq_length], depth_paths[:seq_length], extrinsics_tensor, intrinsics_tensor

    def depths_helper(self, sequence_index, frame_indices):
        start_idx = self.starts[sequence_index].item()

        # Convert to pure Python list of integers for iterating over the string array
        global_indices_list = (start_idx + frame_indices).tolist()

        depths = []
        for idx in global_indices_list:
            # 1. Decode the image -> Shape: (1, H, W)
            # 2. Convert to float32
            # 3. Squeeze the channel dimension -> Shape: (H, W)
            # 4. Convert millimeters to meters
            depth_img = transforms.functional.to_dtype(
                torchvision.io.decode_image(str(self.depths[idx])),
                torch.float32,
                scale=False
            ).squeeze(0) / 1000.0

            depths.append(depth_img)

        # Stacks (H, W) into (L, H, W)
        return torch.stack(depths)