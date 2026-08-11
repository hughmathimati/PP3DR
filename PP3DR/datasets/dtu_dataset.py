import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from datasets.dataset_base import DatasetBase

import torch
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

try:
    from .dataset_base import DatasetBase
except:
    from dataset_base import DatasetBase


class dtu_dataset(DatasetBase):
    """
    22 sequences, each with 48 frames.
    1600 x 1200
    W2C extrinsics with OpenCV coordinate convention.
    """

    # The shortest sequence only has 20 images.
    def __init__(
            self,
            cache_path="/vulcanscratch/hughma/PP3DR/datasets/dtu_dataset_cache.pth",
            dir="/vulcanscratch/hughma/data/dtu/"
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

        print(f"{cache_path} not found. Initialising DTU dataset from scratch...")

        temp_images = []
        temp_depths = []
        temp_extrinsics = []
        temp_intrinsics = []
        temp_starts = []
        temp_lengths = []

        tasks = [(seq_name, dir) for seq_name in self.sequence_names]
        current_idx = 0

        # Process all 22 sequences concurrently
        with ThreadPoolExecutor(max_workers=22) as executor:
            results = list(tqdm(
                executor.map(self._process_sequence, tasks),
                total=len(tasks),
                desc="Parsing DTU sequences and text files"
            ))

        # Sequentially flatten the thread results to preserve deterministic ordering
        for res in results:
            if res is None: continue

            img_paths, depth_paths, ext_tensor, int_tensor = res
            seq_len = len(img_paths)

            if seq_len == 0: continue

            temp_starts.append(current_idx)
            temp_lengths.append(seq_len)

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
        depth_dir = os.path.join(sequence_dir, "depths")
        cams_dir = os.path.join(sequence_dir, "cams")

        if not os.path.exists(image_dir) or not os.path.exists(depth_dir) or not os.path.exists(cams_dir):
            return None

        # 1. Enforce strict sorting to prevent Sintel-style parallax drift!
        image_files = sorted(os.listdir(image_dir))
        depth_files = sorted(os.listdir(depth_dir))
        cam_files = sorted(os.listdir(cams_dir))

        img_paths = [os.path.join(image_dir, f) for f in image_files]
        depth_paths = [os.path.join(depth_dir, f) for f in depth_files]
        cam_paths = [os.path.join(cams_dir, f) for f in cam_files]

        seq_length = min(len(img_paths), len(depth_paths), len(cam_paths))
        if seq_length == 0:
            return None

        extrinsics_tensor = torch.empty(seq_length, 3, 4, dtype=torch.float32)
        intrinsics_tensor = torch.empty(seq_length, 3, 3, dtype=torch.float32)

        # 2. Pre-parse and Pre-invert Camera Matrices
        for i in range(seq_length):
            with open(cam_paths[i], 'r') as f:
                lines = f.readlines()

                # --- Extrinsics (Lines 1-3) ---
                w2c = np.eye(4, dtype=np.float32)
                for j in range(3):
                    floats = lines[1 + j].split()
                    for k in range(4):
                        w2c[j, k] = float(floats[k])

                # Invert World-to-Camera to Camera-to-World!
                # DTU uses OpenCV coordinate conventions, so no axis flipping is necessary.
                c2w = np.linalg.inv(w2c)
                extrinsics_tensor[i] = torch.from_numpy(c2w[:3, :4])

                # --- Intrinsics (Lines 7-9) ---
                for j in range(3):
                    floats = lines[7 + j].split()
                    for k in range(3):
                        intrinsics_tensor[i, j, k] = float(floats[k])

        return img_paths[:seq_length], depth_paths[:seq_length], extrinsics_tensor, intrinsics_tensor

    def depths_helper(self, sequence_index, frame_indices):
        start_idx = self.starts[sequence_index].item()

        # Convert to pure Python list of integers for iterating over the string array
        global_indices_list = (start_idx + frame_indices).tolist()

        depths = []
        for idx in global_indices_list:
            # Load the .npy file, explicitly convert to tensor, and force float32 for safety
            depth_npy = np.load(str(self.depths[idx]))
            depth_tensor = torch.from_numpy(depth_npy).to(torch.float32)
            depths.append(depth_tensor)

        # The inner arrays are already (H, W), so stack combines them to (L, H, W)
        return torch.stack(depths)