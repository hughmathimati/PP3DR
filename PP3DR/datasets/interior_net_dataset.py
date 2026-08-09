import os
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from datasets.dataset_base import DatasetBase


class interior_net_dataset(DatasetBase):
    def __init__(
            self,
            cache_path="/vulcanscratch/hughma/PP3DR/datasets/interior_net_dataset_cache.pth",
            dir="/fs/vulcan-datasets/InteriorNet"
    ):
        super().__init__(dir)
        if os.path.exists(cache_path):
            self.sequences = torch.load(cache_path)
            self.sequence_length = len(self.sequences)
            print(f"Loaded dataset from {cache_path}.")
            return

        print("{cache_path} not found. Initialising dataset from scratch...")
        # Call the base class to initialize patch sizes, grids, and dimensions.
        # This will create dummy self.sequences/sequence_names which we will immediately overwrite.

        self.sequence_names = []
        self.sequences = []

        gt_root = os.path.join(dir, "GroundTruth_HD1-HD6")

        # 1. Iterate over HD folders (HD1 through HD6, skipping HD7 as requested)
        hd_folders = [d for d in os.listdir(dir) if d.startswith("HD") and d != "HD7"]

        for hd in tqdm(hd_folders, desc="Parsing InteriorNet folders"):
            hd_path = os.path.join(dir, hd)
            if not os.path.isdir(hd_path):
                continue

            # 2. Iterate over the scenes within the HD folder
            for scene in os.listdir(hd_path):
                scene_path = os.path.join(hd_path, scene)
                if not os.path.isdir(scene_path):
                    continue

                # 3. Iterate over lighting/trajectory conditions (e.g., original_1_1, random_1_1)
                for traj in os.listdir(scene_path):
                    if not (traj.startswith("original_") or traj.startswith("random_")):
                        continue

                    traj_path = os.path.join(scene_path, traj)

                    # Directories containing the actual PNGs
                    img_dir = os.path.join(traj_path, "cam0", "data")
                    depth_dir = os.path.join(traj_path, "depth0", "data")

                    # Ground truth camera path
                    # InteriorNet maps both 'original_X_X' and 'random_X_X' to 'velocity_angular_X_X'
                    parts = traj.split('_')
                    if len(parts) >= 3:
                        vel_name = f"velocity_angular_{parts[1]}_{parts[2]}"
                    else:
                        continue  # Safety check for malformed folder names

                    cam_file = os.path.join(gt_root, scene, vel_name, "cam0.ccam")

                    # Skip if any of the required folders/files are missing
                    if not os.path.exists(img_dir) or not os.path.exists(depth_dir) or not os.path.exists(cam_file):
                        continue

                    # 4. Extract and safely sort the image and depth files
                    # We use integer sorting here to prevent the "0, 1, 10, 2" bug we caught earlier.
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

                    # 5. Parse the custom .ccam file into pre-computed tensors
                    extrinsics, intrinsics = self._parse_ccam_file(cam_file)

                    # 6. Safety check: ensure arrays align and sequence is long enough
                    min_len = min(len(img_paths), len(depth_paths), len(extrinsics))
                    if min_len < self.sequence_length:
                        continue

                    # Store the valid sequence
                    self.sequence_names.append(f"{hd}_{scene}_{traj}")
                    self.sequences.append({
                        'images': img_paths[:min_len],
                        'depths': depth_paths[:min_len],
                        'extrinsics': extrinsics[:min_len],
                        'intrinsics': intrinsics[:min_len]
                    })

        torch.save(self.sequences, cache_path)
        print(f"Saved to {cache_path}")

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
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            parts = line.split()

            # The InteriorNet format has 15 parameters per camera line:
            # f cx cy d0 d1 d2 w x y z X Y Z width height
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
                # Scipy expects quaternions in (x, y, z, w) format.
                # The .ccam file provides them in (w, x, y, z) format.
                R = Rotation.from_quat([x, y, z, w]).as_matrix()
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = R
                c2w[:3, 3] = [tx, ty, tz]

                # Convert OpenGL (X right, Y up, Z backward) to OpenCV (X right, Y down, Z forward)
                # The InteriorNet PDF explicitly states their system is "RHY UP" (Right-Handed, Y-Up)
                c2w[:, 1:3] *= -1

                extrinsics.append(torch.from_numpy(c2w[:3, :4]))

        return torch.stack(extrinsics), torch.stack(intrinsics)

    def depths_helper(self, sequence_index, frame_indices):
        """
        Overrides the base depth helper to:
        1. Read the raw 16-bit PNG without dividing by 65535 (scale=False).
        2. Divide by 1000.0 to convert millimeters to meters.
        3. Convert Euclidean (Radial) Depth into Planar Z-Depth.
        """
        depths = []
        for i in frame_indices:
            # 1. Read raw 16-bit depth and convert to meters
            depth_img = transforms.functional.to_dtype(
                torchvision.io.decode_image(
                    self.sequences[sequence_index]['depths'][i]
                ),
                torch.float32,
                scale=False
            ) / 1000.0

            # 2. Get intrinsics to calculate ray vectors
            K = self.sequences[sequence_index]['intrinsics'][i]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            _, H, W = depth_img.shape

            # 3. Create a pixel grid
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

            # 4. Calculate the normalized ray direction for each pixel
            dir_x = (x - cx) / fx
            dir_y = (y - cy) / fy

            # 5. Calculate the Euclidean norm (length) of each ray
            # D = Z * sqrt(x^2 + y^2 + 1) --> Z = D / sqrt(...)
            ray_norm = torch.sqrt(dir_x ** 2 + dir_y ** 2 + 1.0)

            # 6. Flatten the fisheye distortion!
            z_depth = depth_img / ray_norm

            depths.append(z_depth)

        # Concatenate into shape (L, H, W)
        return torch.cat(depths)

if __name__ == "__main__":
    dataset = interior_net_dataset()
    for k, v in dataset[0].items():
        print(k, v.shape)