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
from datasets.dynamic_replica_dataset import dynamic_replica_dataset

class dynamic_replica_test_dataset(dynamic_replica_dataset):
    # The shortest sequence only has 20 images.
    def __init__(self):
        super().__init__(dir="/fs/vulcan-datasets/dynamic_replica/test")
