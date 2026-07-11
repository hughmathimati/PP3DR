import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor
import cv2
import gzip
import json
from functools import partial
try:
    from .dataset_base import DatasetBase
except:
    from dataset_base import DatasetBase

class dynamic_replica_dataset(DatasetBase):
    """
    966 sequences, each with 300 frames.
    1280 x 720
    """
    # The shortest sequence only has 20 images.
    def __init__(self, dir="/fs/vulcan-datasets/dynamic_replica/train"):
        super().__init__(dir)
        for sequence in self.sequences:
            sequence['extrinsics'] = torch.empty(300, 3, 4)
            sequence['intrinsics'] = torch.zeros(300, 3, 3)
        for i, sequence in tqdm(enumerate(self.sequence_names), desc="Precomputing Dynamic Replica image file paths"):
            if sequence == "frame_annotations_train.jgz" or sequence[-5:] == "right":
                continue
            sequence_dir = os.path.join(dir, sequence)
            image_dir = os.path.join(sequence_dir, "images")
            depth_dir = os.path.join(sequence_dir, "depths")

            for image in os.listdir(image_dir):
                if image == "done.ok": continue
                self.sequences[i]['images'].append(os.path.join(image_dir, image))
            for depth in os.listdir(depth_dir):
                self.sequences[i]['depths'].append(os.path.join(depth_dir, depth))
        # Have to wait until the end to add this because I need self.sequences to be filled
        self.extract_sequence_cameras(os.path.join(dir, "frame_annotations_train.jgz"))

    def extract_sequence_cameras(self, jgz_path):
        """
        Extracts and standardizes camera matrices from a PyTorch3D .jgz file.
        """
        print(f"DynamicReplica: Loading {jgz_path}...")

        # Read the gzipped JSON file using standard libraries
        with gzip.open(jgz_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        for frame in tqdm(data, desc="Reading camera matrices..."):
            sequence_index = self.sequence_names.index(frame['sequence_name'] + "_source_left")
            frame_index = frame['frame_number']
            viewpoint = frame["viewpoint"]

            # --- 1. Extrinsics (PyTorch3D W2C -> OpenCV C2W) ---
            R_pt3d = np.array(viewpoint["R"])  # Shape (3, 3)
            T_pt3d = np.array(viewpoint["T"])  # Shape (3,)

            # Build standard column-major W2C matrix
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R_pt3d.T
            w2c[:3, 3] = T_pt3d

            # Convert PyTorch3D camera space (+X Left, +Y Up)
            # to OpenCV camera space (+X Right, +Y Down)
            w2c[0, :] *= -1
            w2c[1, :] *= -1

            # Invert to C2W and assign
            c2w = np.linalg.inv(w2c)
            self.sequences[sequence_index]['extrinsics'][frame_index][:3, :4] = torch.tensor(c2w[:3, :4])

            # --- 2. Intrinsics (NDC -> Pixels) ---
            focal_length = viewpoint["focal_length"]
            principal_point = viewpoint["principal_point"]

            # Get image dimensions (replace with your actual H and W variables)
            # Standard DynamicReplica is usually 480x640 or 360x640
            H, W = 480, 640
            half_w, half_h = W / 2.0, H / 2.0

            # Convert NDC to Pixel Coordinates
            fx_px = focal_length[0] * half_w
            fy_px = focal_length[1] * half_h

            # In PyTorch3D NDC, (0,0) is the center.
            # Subtracting moves it to the top-left pixel origin.
            cx_px = half_w - (principal_point[0] * half_w)
            cy_px = half_h - (principal_point[1] * half_h)

            # Construct 3x3 Intrinsic matrix
            self.sequences[sequence_index]['intrinsics'][frame_index][2, 2] = 1
            self.sequences[sequence_index]['intrinsics'][frame_index][0, 0] = fx_px
            self.sequences[sequence_index]['intrinsics'][frame_index][1, 1] = fy_px
            self.sequences[sequence_index]['intrinsics'][frame_index][0, 2] = cx_px
            self.sequences[sequence_index]['intrinsics'][frame_index][1, 2] = cy_px


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
    dataset = dynamic_replica_dataset()
    total_files = len(dataset)
    print(total_files)
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
    #     with open("invalid_dynamic_replica_files.txt", "w") as f:
    #         for path in invalid_files:
    #             f.write(f"{path}\n")