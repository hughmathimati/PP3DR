import os

# CRITICAL: This MUST be set before cv2 is imported!
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets.dataset_base import DatasetBase


class rtmv_dataset(DatasetBase):
    def __init__(
            self,
            cache_path="/vulcanscratch/hughma/PP3DR/datasets/rtmv/rtmv_dataset_cache.pth",
            dir="/fs/vulcan-datasets/RTMV/all_in_separate_tar_files"
    ):
        super().__init__(dir)

        # Load from cache if available
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, weights_only=False)
            self.images = cache['images']
            self.depths = cache['depths']
            self.extrinsics = cache['extrinsics']
            self.intrinsics = cache['intrinsics']
            self.starts = cache['starts']
            self.lengths = cache['lengths']

            self.num_sequences = len(self.starts)
            print(f"Loaded RTMV dataset from {cache_path}.")
            return

        print(f"{cache_path} not found. Initialising RTMV dataset from scratch...")

        # 1. Grab top-level directories to distribute the crawling work
        try:
            top_level_dirs = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        except FileNotFoundError:
            top_level_dirs = []

        scene_dirs = []

        # 2. Worker function for parallel os.walk
        def crawl_directory(sub_dir):
            valid_scenes = []
            for root, dirs, files in os.walk(sub_dir):
                if "00000.exr" in files and "00000.json" in files:
                    valid_scenes.append(root)
            return valid_scenes

        # 3. Multithread the directory crawling to bypass NAS latency
        if top_level_dirs:
            with ThreadPoolExecutor(max_workers=32) as executor:
                results = list(tqdm(
                    executor.map(crawl_directory, top_level_dirs),
                    total=len(top_level_dirs),
                    desc="Crawling RTMV folder tree"
                ))
            for res in results:
                scene_dirs.extend(res)

        base_files = os.listdir(dir)
        if "00000.exr" in base_files and "00000.json" in base_files:
            scene_dirs.append(dir)

        print(f"Found {len(scene_dirs)} valid scene directories. Precomputing data...")

        temp_images = []
        temp_depths = []
        temp_extrinsics = []
        temp_intrinsics = []
        temp_starts = []
        temp_lengths = []

        current_idx = 0

        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(tqdm(
                executor.map(self._process_scene, scene_dirs),
                total=len(scene_dirs),
                desc="Parsing RTMV sequences and HDF5 files"
            ))

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

        self.extrinsics = torch.cat(temp_extrinsics, dim=0)
        self.intrinsics = torch.cat(temp_intrinsics, dim=0)

        self.starts = torch.tensor(temp_starts, dtype=torch.long)
        self.lengths = torch.tensor(temp_lengths, dtype=torch.long)

        self.num_sequences = len(self.starts)

        torch.save({
            'images': self.images,
            'depths': self.depths,
            'extrinsics': self.extrinsics,
            'intrinsics': self.intrinsics,
            'starts': self.starts,
            'lengths': self.lengths
        }, cache_path)

        print(f"Saved to {cache_path}")

    def _process_scene(self, scene_path):
        files = os.listdir(scene_path)

        frame_indices = []
        for f in files:
            if f.endswith('.json'):
                try:
                    frame_indices.append(int(f.split('.')[0]))
                except ValueError:
                    continue

        frame_indices = sorted(frame_indices)

        img_paths = []
        depth_paths = []
        extrinsics_list = []
        intrinsics_list = []

        sample_rgb = os.path.join(scene_path, f"{frame_indices[0]:05d}.exr")
        sample_img = cv2.imread(sample_rgb, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if sample_img is None:
            return None

        H, W = sample_img.shape[:2]

        for i in frame_indices:
            rgb_path = os.path.join(scene_path, f"{i:05d}.exr")
            depth_path = os.path.join(scene_path, f"{i:05d}.depth.exr")
            json_path = os.path.join(scene_path, f"{i:05d}.json")

            # STRICT EXCLUSION: Ignore known corrupted frame
            if rgb_path.endswith("abo/falling_amazon_berkeley_scenes/00065/00016.exr"):
                continue

            if not (os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(json_path)):
                continue

            with open(json_path, 'r') as f:
                meta = json.load(f)

            if 'transform_matrix' in meta:
                c2w = np.array(meta['transform_matrix'], dtype=np.float32)
            elif 'camera_data' in meta and 'cam2world' in meta['camera_data']:
                c2w = np.array(meta['camera_data']['cam2world'], dtype=np.float32)
            else:
                continue

            if np.allclose(c2w[:3, 3], 0.0) and not np.allclose(c2w[3, :3], 0.0):
                c2w = c2w.T

            c2w[:, 1:3] *= -1
            extrinsics_list.append(torch.from_numpy(c2w[:3, :4]))

            if 'camera_data' in meta and 'intrinsics' in meta['camera_data'] and 'fx' in meta['camera_data'][
                'intrinsics']:
                intr = meta['camera_data']['intrinsics']
                fx = intr['fx']
                fy = intr['fy']
                cx = intr.get('cx', W / 2.0)
                cy = intr.get('cy', H / 2.0)
            elif 'camera_angle_x' in meta:
                fov_x = meta['camera_angle_x']
                fx = 0.5 * W / np.tan(0.5 * fov_x)
                fy = fx
                cx, cy = W / 2.0, H / 2.0
            elif 'camera_data' in meta and 'intrinsics' in meta['camera_data']:
                intr = meta['camera_data']['intrinsics']
                if 'focal_length' in intr and 'sensor_width' in intr:
                    fx = (intr['focal_length'] / intr['sensor_width']) * W
                    fy = fx
                    cx, cy = W / 2.0, H / 2.0
                else:
                    fov_x = intr.get('fov', 0.85)
                    fx = 0.5 * W / np.tan(0.5 * fov_x)
                    fy = fx
                    cx, cy = W / 2.0, H / 2.0
            else:
                fx = 0.5 * W / np.tan(0.5 * 0.85)
                fy = fx
                cx, cy = W / 2.0, H / 2.0

            K = torch.eye(3, dtype=torch.float32)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            intrinsics_list.append(K)

            img_paths.append(rgb_path)
            depth_paths.append(depth_path)

        seq_len = len(img_paths)
        if seq_len == 0:
            return None

        return img_paths, depth_paths, torch.stack(extrinsics_list), torch.stack(intrinsics_list)

    def images_helper(self, sequence_index, frame_indices):
        start_idx = self.starts[sequence_index].item()
        global_indices_list = (start_idx + frame_indices).tolist()

        images = []
        for idx in global_indices_list:
            img_path = str(self.images[idx])
            img_bgr = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            # STRICT CRASH TRAP
            if img_bgr is None:
                raise RuntimeError(f"OpenCV failed to read RGB EXR: {img_path}")

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            images.append(img_tensor)

        return torch.stack(images)

    def depths_helper(self, sequence_index, frame_indices):
        start_idx = self.starts[sequence_index].item()
        global_indices_list = (start_idx + frame_indices).tolist()

        depths = []
        for idx in global_indices_list:
            depth_path = str(self.depths[idx])
            depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            # STRICT CRASH TRAP
            if depth_img is None:
                raise RuntimeError(f"OpenCV failed to read Depth EXR: {depth_path}")

            if depth_img.ndim == 3:
                depth_img = depth_img[:, :, 0]

            depth_tensor = torch.from_numpy(depth_img).float()

            depth_tensor = torch.nan_to_num(depth_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            depth_tensor[depth_tensor > 100.0] = 0.0

            K = self.intrinsics[idx]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            H, W = depth_tensor.shape

            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            dir_x = (x - cx) / fx
            dir_y = (y - cy) / fy
            ray_norm = torch.sqrt(dir_x ** 2 + dir_y ** 2 + 1.0)

            planar_z = depth_tensor / ray_norm
            depths.append(planar_z)

        return torch.stack(depths)


import traceback
def helper(dataset, i):
    d = dataset[i]
    return i, (~d['images'].isfinite()).sum() + (~d['depths'].isfinite()).sum() + (~d['extrinsics'].isfinite()).sum() + (~d['intrinsics'].isfinite()).sum()

if __name__ == "__main__":
    dataset = rtmv_dataset()
    invalid = []

    with ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(helper, dataset, i): i for i in range(len(dataset))}

        for future in tqdm(as_completed(future_to_idx)):
            i, bad = future.result()
            if bad > 0:
                invalid.append(i)

    print(len(invalid))
    print(invalid)