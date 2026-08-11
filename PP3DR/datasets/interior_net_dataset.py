import os
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from datasets.dataset_base import DatasetBase
from concurrent.futures import ThreadPoolExecutor


class interior_net_dataset(DatasetBase):
    def __init__(
            self,
            cache_path="/vulcanscratch/hughma/PP3DR/datasets/interior_net_dataset_cache.pth",
            dir="/fs/vulcan-datasets/InteriorNet"
    ):
        super().__init__(dir)
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

        print(f"{cache_path} not found. Initialising dataset from scratch...")

        temp_images = []
        temp_depths = []
        temp_extrinsics = []
        temp_intrinsics = []
        temp_starts = []
        temp_lengths = []

        gt_root = os.path.join(dir, "GroundTruth_HD1-HD6")

        # 1. Gather all tasks (HD Folder -> Scene -> Trajectory)
        tasks = []
        hd_folders = [d for d in os.listdir(dir) if d.startswith("HD") and d != "HD7"]

        for hd in hd_folders:
            hd_path = os.path.join(dir, hd)
            if not os.path.isdir(hd_path): continue

            for scene in os.listdir(hd_path):
                scene_path = os.path.join(hd_path, scene)
                if not os.path.isdir(scene_path): continue

                for traj in os.listdir(scene_path):
                    if not (traj.startswith("original_") or traj.startswith("random_")): continue
                    tasks.append((hd_path, scene, traj, gt_root))

        current_idx = 0

        # 2. Process all trajectories in parallel
        # 32 workers will massively accelerate parsing deeply nested text files
        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(tqdm(
                executor.map(self._process_trajectory, tasks),
                total=len(tasks),
                desc="Parsing InteriorNet trajectories"
            ))

        # 3. Flatten the results sequentially into the master lists
        for res in results:
            if res is None: continue

            img_list, depth_list, ext_tensor, int_tensor = res
            seq_len = len(img_list)

            # Safety check: ensure sequence is long enough to sample from
            if seq_len < self.sequence_length:
                continue

            temp_starts.append(current_idx)
            temp_lengths.append(seq_len)

            temp_images.extend(img_list)
            temp_depths.extend(depth_list)
            temp_extrinsics.append(ext_tensor)
            temp_intrinsics.append(int_tensor)

            current_idx += seq_len

        self.images = np.array(temp_images, dtype='U255')
        self.depths = np.array(temp_depths, dtype='U255')
        self.extrinsics = torch.cat(temp_extrinsics, dim=0)
        self.intrinsics = torch.cat(temp_intrinsics, dim=0)
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

    def _process_trajectory(self, args):
        """
        Background worker function to isolate folder parsing and matrix generation.
        Returns local variables to guarantee thread safety.
        """
        hd_path, scene, traj, gt_root = args

        traj_path = os.path.join(hd_path, scene, traj)
        img_dir = os.path.join(traj_path, "cam0", "data")
        depth_dir = os.path.join(traj_path, "depth0", "data")

        parts = traj.split('_')
        if len(parts) < 3:
            return None

        vel_name = f"velocity_angular_{parts[1]}_{parts[2]}"
        cam_file = os.path.join(gt_root, scene, vel_name, "cam0.ccam")

        if not os.path.exists(img_dir) or not os.path.exists(depth_dir) or not os.path.exists(cam_file):
            return None

        # Sort images by numerical value
        image_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith('.png')],
            key=lambda x: int(x.split('.')[0])
        )
        depth_files = sorted(
            [f for f in os.listdir(depth_dir) if f.endswith('.png')],
            key=lambda x: int(x.split('.')[0])
        )

        img_paths = [os.path.join(img_dir, f) for f in image_files]
        depth_paths = [os.path.join(depth_dir, f) for f in depth_files]

        extrinsics, intrinsics = self._parse_ccam_file(cam_file)

        min_len = min(len(img_paths), len(depth_paths), len(extrinsics))

        if min_len == 0:
            return None

        # Darkness filter
        img = torchvision.io.read_image(
            img_paths[0],
            mode=torchvision.io.image.ImageReadMode.RGB
        )
        # Calculate mean brightness (0 to 255 scale)
        mean_brightness = img.float().mean()

        # If the average pixel is less than ~6% brightness (pitch black / barely visible)
        if mean_brightness < 15.0:
            return None

        return (
            img_paths[:min_len],
            depth_paths[:min_len],
            extrinsics[:min_len],
            intrinsics[:min_len]
        )

    def _parse_ccam_file(self, filepath):
        """
        Helper function to parse the InteriorNet .ccam files.
        """
        extrinsics = []
        intrinsics = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()

            if len(parts) >= 15:
                f_val = float(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])

                w = float(parts[6])
                x = float(parts[7])
                y = float(parts[8])
                z = float(parts[9])

                tx = float(parts[10])
                ty = float(parts[11])
                tz = float(parts[12])

                # --- Construct Intrinsics (3x3) ---
                K = torch.eye(3, dtype=torch.float32)
                K[0, 0] = f_val
                K[1, 1] = f_val
                K[0, 2] = cx
                K[1, 2] = cy
                intrinsics.append(K)

                # --- Construct Extrinsics (C2W) ---
                R = Rotation.from_quat([x, y, z, w]).as_matrix()
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = R
                c2w[:3, 3] = [tx, ty, tz]

                # Convert OpenGL to OpenCV
                c2w[:, 1:3] *= -1

                extrinsics.append(torch.from_numpy(c2w[:3, :4]))

        return torch.stack(extrinsics), torch.stack(intrinsics)

    def depths_helper(self, sequence_index, frame_indices):
        """
        Overrides the base depth helper to process Euclidean -> Planar conversion.
        Reads strictly from the zero-copy flat arrays!
        """
        depths = []

        # Determine the absolute start index in the flattened array
        start_idx = self.starts[sequence_index].item()
        global_indices = [start_idx + i for i in frame_indices]

        for idx in global_indices:
            # 1. Read raw 16-bit depth and convert to meters
            depth_img = transforms.functional.to_dtype(
                torchvision.io.decode_image(self.depths[idx]),
                torch.float32,
                scale=False
            ) / 1000.0

            # 2. Get intrinsics to calculate ray vectors
            K = self.intrinsics[idx]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            _, H, W = depth_img.shape

            # 3. Create a pixel grid
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

            # 4. Calculate the normalized ray direction for each pixel
            dir_x = (x - cx) / fx
            dir_y = (y - cy) / fy

            # 5. Calculate the Euclidean norm
            ray_norm = torch.sqrt(dir_x ** 2 + dir_y ** 2 + 1.0)

            # 6. Flatten the fisheye distortion
            z_depth = depth_img / ray_norm

            depths.append(z_depth)

        return torch.cat(depths)