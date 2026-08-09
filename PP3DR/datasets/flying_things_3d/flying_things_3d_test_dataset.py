import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re
from datasets.dataset_base import DatasetBase
from datasets.flying_things_3d.flying_things_3d_dataset import flying_things_3d_dataset


class flying_things_3d_test_dataset(flying_things_3d_dataset):
    """
    2239 sequences in total, each with 10 frames.
    However, we sometimes only have 9 extrinsics. In such cases, we must only take 9 of those frames.
    540x960
    """
    def __init__(self):
        super().__init__(
            cache_path="/vulcanscratch/hughma/PP3DR/datasets/flying_things_3d/flying_things_3d_test_dataset_cache.pth",
            images_dir="/fs/vulcan-datasets/FlyingThings3D/frames_cleanpass/TEST",
            disparities_dir="/fs/vulcan-datasets/FlyingThings3D/disparity/TEST",
            extrinsics_dir="/vulcanscratch/hughma/data/FlyingThings3D/camera_data/TEST"
        )