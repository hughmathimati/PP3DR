import torch
import torchvision.transforms.v2 as transforms
import torchvision
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datasets.dataset_base import DatasetBase
from datasets.sintel.sintel_io import depth_read, cam_read


class sintel_dataset(DatasetBase):
    """
    23 sequences, each with between 20 to 50 frames each.
    1024x436
    """

    def __init__(self, dir="/vulcanscratch/hughma/data/sintel/training/"):
        super().__init__(dir)
        # Because sintel_io.cam_read outputs both the extrinsic and intrinsic matrices for each frame's file path, I'm going to
        # pre-load all extrinsic/intrinsic matrices.
        images_dir = os.path.join(dir, "final")
        depths_dir = os.path.join(dir, "depth")
        cams_dir = os.path.join(dir, "camdata_left")
        # Redefine self.sequences because this dataset's folder structure is different from usual
        self.sequences = [dict(images=[], depths=[]) for _ in range(23)]
        with ThreadPoolExecutor() as executor:
            for i, sequence in tqdm(enumerate(os.listdir(images_dir))):
                images = os.path.join(images_dir, sequence)
                depths = os.path.join(depths_dir, sequence)
                cams = os.path.join(cams_dir, sequence)
                images_list = os.listdir(images)
                seq_length = len(images_list)
                self.sequences[i]['extrinsics'] = torch.empty(seq_length, 3, 4)
                self.sequences[i]['intrinsics'] = torch.empty(seq_length, 3, 3)

                executor.submit(self.cams_helper, cams, i)
                for image in images_list:
                    self.sequences[i]['images'].append(os.path.join(images, image))
                for depth in os.listdir(depths):
                    self.sequences[i]['depths'].append(os.path.join(depths, depth))

    def cams_helper(self, cams, sequence_index):
        for j, cam in enumerate(os.listdir(cams)):
            intrinsic, extrinsic = cam_read(os.path.join(cams, cam))
            self.sequences[sequence_index]['intrinsics'][j] = torch.from_numpy(intrinsic)
            self.sequences[sequence_index]['extrinsics'][j] = torch.from_numpy(extrinsic[:3])

    def depths_helper(self, sequence_index, frame_indices):
        """
        Sintel uses a binary format for its depth files, rather than a PNG image.

        Parameters
        ----------
        sequence_index
        frame_indices: Randomly-sampled indices of the frames in self.sequences[sequence_index] constituting this
                       randomly-sampled sequence.
        Returns
        -------
        (L, H, W) tensor of the depths for this randomly-sampled sequence.
        """
        return torch.stack([
            torch.from_numpy(depth_read(self.sequences[sequence_index]['depths'][frame_index]))
            for frame_index in frame_indices
        ])


if __name__ == "__main__":
    dataset = sintel_dataset()
    empty = []
    for i, data in tqdm(enumerate(dataset), desc="Scanning sequences..."):
        if len(data['images']) == 0:
            empty.append(i)
    print(empty)
