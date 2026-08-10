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
from datasets.dataset_base import DatasetBase

class dynamic_replica_dataset(DatasetBase):
    """
    966 sequences, each with 300 frames.
    1280 x 720
    """
    # The shortest sequence only has 20 images.
    def __init__(
            self,
            cache_path="/vulcanscratch/hughma/PP3DR/datasets/dynamic_replica/dynamic_replica_dataset_cache.pth",
            dir="/fs/vulcan-datasets/dynamic_replica/train",
            annotations_file_name="frame_annotations_train.jgz"
    ):
        super().__init__(dir)
        if os.path.exists(cache_path):
            self.sequences = torch.load(cache_path)
            self.sequence_length = len(self.sequences)
            print(f"Loaded dataset from {cache_path}.")
            return

        print(f"{cache_path} not found. Initialising Dynamic Replica dataset from scratch...")
        with ThreadPoolExecutor() as executor:
            # Extracting the camera matrices requires self.sequence_names to be filled. However, reading the jgz file
            # does not. So we'll do that first.
            a = executor.submit(self.load_jgz, os.path.join(dir, annotations_file_name))

            # Filter out the right-camera sequences (and the jgz file) ahead of time.
            # We need to do this since extract_sequence_cameras relies on self.sequence_names.index().
            # It will happen concurrently with self.load_jgz().
            valid_indices = [
                i for i, name in enumerate(self.sequence_names)
                if name != annotations_file_name and not name.endswith("right")
            ]
            self.sequence_names = [self.sequence_names[i] for i in valid_indices]
            self.sequences = [self.sequences[i] for i in valid_indices]
            self.name_to_idx = {name: i for i, name in enumerate(self.sequence_names)}
            for sequence in self.sequences:
                # It's okay for these to be longer than the image sequence length, because when we actually sample the
                # frame indices, we're only sampling from valid image indices.
                # The train sequence lengths seem to be at most 300, but the test ones seem to go up to 901.
                sequence['extrinsics'] = torch.empty(901, 3, 4)
                sequence['intrinsics'] = torch.zeros(901, 3, 3)

            # Now that self.sequence_names is filled, we can submit self.extract_sequence_cameras().
            b = executor.submit(self.extract_sequence_cameras, a.result())

            for i, sequence in tqdm(enumerate(self.sequence_names), desc="Precomputing Dynamic Replica image file paths"):
                sequence_dir = os.path.join(dir, sequence)
                image_dir = os.path.join(sequence_dir, "images")
                depth_dir = os.path.join(sequence_dir, "depths")

                for image in os.listdir(image_dir):
                    if image == "done.ok": continue
                    self.sequences[i]['images'].append(os.path.join(image_dir, image))
                for depth in os.listdir(depth_dir):
                    self.sequences[i]['depths'].append(os.path.join(depth_dir, depth))

            # Make sure we don't proceed until b is done running.
            b.result()

        torch.save(self.sequences, cache_path)
        print(f"Saved to {cache_path}")

    def load_jgz(self, path):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def extract_sequence_cameras(self, data):
        for frame in tqdm(data, desc="Reading camera matrices..."):
            if frame['camera_name'] == "right":
                continue
            target_name = frame['sequence_name'] + "_source_left"
            sequence_index = self.name_to_idx[target_name]
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


if __name__ == "__main__":
    dataset = dynamic_replica_dataset()
    empty = []
    for i, data in tqdm(enumerate(dataset), desc="Scanning sequences", mininterval=1):
        if len(data['images']) == 0:
            empty.append(i)
    print(empty)