import torch
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from PIL import Image
from datasets.dataset_base import DatasetBase


class dynamic_replica_dataset(DatasetBase):
    """
    966 sequences, each with up to 300 frames.
    1280 x 720
    """

    def __init__(
            self,
            cache_path="/vulcanscratch/hughma/PP3DR/datasets/dynamic_replica/dynamic_replica_dataset_cache.pth",
            dir="/fs/vulcan-datasets/dynamic_replica/train",
            annotations_file_name="frame_annotations_train.jgz"
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
            print(f"Loaded dataset from {cache_path}.")
            return

        print(f"{cache_path} not found. Initialising Dynamic Replica dataset from scratch...")

        # 1. Load the annotations completely
        jgz_path = os.path.join(dir, annotations_file_name)
        data = self.load_jgz(jgz_path)

        # 2. Parse camera matrices into a fast lookup dictionary
        cam_dict = self._parse_cameras(data)

        temp_images = []
        temp_depths = []
        temp_extrinsics = []
        temp_intrinsics = []
        temp_starts = []
        temp_lengths = []

        # Filter valid sequences
        valid_sequences = [
            name for name in self.sequence_names
            if name != annotations_file_name and not name.endswith("right") and name in cam_dict
        ]

        tasks = [(seq_name, dir, cam_dict[seq_name]) for seq_name in valid_sequences]
        current_idx = 0

        # 3. Multithread the intersection
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self._process_sequence, tasks),
                total=len(tasks),
                desc="Aligning Dynamic Replica sequences and parsing paths"
            ))

        # 4. Sequentially flatten the thread results
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

    def load_jgz(self, path):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _parse_cameras(self, data):
        seq_cams = {}
        for frame in tqdm(data, desc="Reading camera matrices from JSON..."):
            if frame['camera_name'] == "right":
                continue

            target_name = frame['sequence_name'] + "_source_left"
            frame_index = frame['frame_number']
            viewpoint = frame["viewpoint"]

            # --- 1. Extrinsics (PyTorch3D W2C -> OpenCV C2W) ---
            R_pt3d = np.array(viewpoint["R"])
            T_pt3d = np.array(viewpoint["T"])

            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R_pt3d.T
            w2c[:3, 3] = T_pt3d

            # Convert PyTorch3D (+X Left, +Y Up) to OpenCV (+X Right, +Y Down)
            w2c[0, :] *= -1
            w2c[1, :] *= -1

            c2w = np.linalg.inv(w2c)
            extrinsic_tensor = torch.from_numpy(c2w[:3, :4])

            # --- 2. Intrinsics (NDC -> Pixels) ---
            focal_length = viewpoint["focal_length"]
            principal_point = viewpoint["principal_point"]

            H, W = 720, 1280
            half_w, half_h = W / 2.0, H / 2.0

            # DYNAMICALLY READ THE FORMAT TO GUARANTEE PERFECT ALIGNMENT
            format = viewpoint.get("intrinsics_format", "ndc_isotropic")

            if format.lower() == "ndc_norm_image_bounds":
                rescale_x = half_w
                rescale_y = half_h
            else:  # "ndc_isotropic"
                rescale_x = min(half_w, half_h)
                rescale_y = min(half_w, half_h)

            fx_px = focal_length[0] * rescale_x
            fy_px = focal_length[1] * rescale_y

            # Subtract from the true center using the correct format scalar!
            cx_px = half_w - (principal_point[0] * rescale_x)
            cy_px = half_h - (principal_point[1] * rescale_y)

            intrinsic_tensor = torch.eye(3, dtype=torch.float32)
            intrinsic_tensor[0, 0] = fx_px
            intrinsic_tensor[1, 1] = fy_px
            intrinsic_tensor[0, 2] = cx_px
            intrinsic_tensor[1, 2] = cy_px

            if target_name not in seq_cams:
                seq_cams[target_name] = {}
            seq_cams[target_name][frame_index] = (extrinsic_tensor, intrinsic_tensor)

        return seq_cams

    def _process_sequence(self, args):
        sequence_name, base_dir, frame_cams = args
        sequence_dir = os.path.join(base_dir, sequence_name)
        image_dir = os.path.join(sequence_dir, "images")
        depth_dir = os.path.join(sequence_dir, "depths")

        if not os.path.exists(image_dir) or not os.path.exists(depth_dir):
            return None

        # Handle delimiters flawlessly
        img_dict = {}
        for f in os.listdir(image_dir):
            if f == "done.ok": continue
            frame_idx = int(f.split('.')[0].replace('-', '_').split('_')[-1])
            img_dict[frame_idx] = os.path.join(image_dir, f)

        depth_dict = {}
        for f in os.listdir(depth_dir):
            if f == "done.ok": continue
            frame_idx = int(f.split('.')[0].replace('-', '_').split('_')[-1])
            depth_dict[frame_idx] = os.path.join(depth_dir, f)

        valid_frames = sorted(list(set(img_dict.keys()) & set(depth_dict.keys()) & set(frame_cams.keys())))

        if not valid_frames:
            return None

        img_paths = [img_dict[i] for i in valid_frames]
        depth_paths = [depth_dict[i] for i in valid_frames]
        extrinsics = [frame_cams[i][0] for i in valid_frames]
        intrinsics = [frame_cams[i][1] for i in valid_frames]

        return img_paths, depth_paths, torch.stack(extrinsics), torch.stack(intrinsics)

    def depths_helper(self, sequence_index, frame_indices):
        start_idx = self.starts[sequence_index].item()
        global_indices_list = (start_idx + frame_indices).tolist()

        depths = []
        for idx in global_indices_list:
            depth_png = str(self.depths[idx])

            # THE FIX: Cast the uint16 bits into float16 natively!
            with Image.open(depth_png) as depth_pil:
                depth_np = (
                    np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                    .astype(np.float32)
                    .reshape((depth_pil.size[1], depth_pil.size[0]))
                )

            depths.append(torch.from_numpy(depth_np))

        return torch.stack(depths)