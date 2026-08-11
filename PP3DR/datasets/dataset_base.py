import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import torchvision
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
cv2.setNumThreads(0)


class DatasetBase(torch.utils.data.Dataset):
    """
    Base dataset class.

    You have to write __init__(), extrinsics_helper(), and intrinsics_helper().

    images_helper() and depths_helper() are optional overrides. Overriding depths_helper() is much more common.

    __getitem__() is intentionally implemented here, to be inherited by all children classes.
    """

    def __init__(self, dir):
        """
        MUST have:
        self.starts: Tensor containing the start index for each sequence
        self.lengths: Tensor containing the lengths of each sequence

        Default helper functions expect:
        self.images: Numpy array of all string paths to images
        self.depths: Numpy array of all string paths to depths
        self.extrinsics: Tensor containing all extrinsic matrices
        self.intrinsics: Tensor containing all intrinsic matrices
        """
        super().__init__()
        # This is the number of frames we'd like to sample from each sequence. If this is more than the number of frames
        # in the sequence, we will oversample.
        self.sequence_length = 10
        self.sequence_names = os.listdir(dir)
        # Start with a lower dimension for initial training and sanity-checking. You can fine-tune at a higher
        # resolution later.
        # print("WARNING: Patch size set to 14 for Pi3 sanity.")
        self.input_dim = 512
        self.patch_size = 16  # This should be set to the patch size of your feature extractor.
        assert self.input_dim % self.patch_size == 0, f"self.input_dim must be a multiple of the patch size ({self.patch_size})."
        # Set this to True to have __getittem__() return the raw images as well.
        # This should be False for training and only turned on for debugging/visualization purposes.
        self.raw = False

        # Create the base grid coordinates for RoPE (doing this once and saving it prevents recomputation later)
        grid_size = self.input_dim // self.patch_size
        # Create a grid of pixel coordinates for the center of each patch
        # Shape of y_grid and x_grid: (grid_size, grid_size)
        self.y_grid, self.x_grid = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32) + 0.5,
            torch.arange(grid_size, dtype=torch.float32) + 0.5,
            indexing='ij'
        )

    def __len__(self):
        return len(self.starts)

    def images_helper(self, sequence_index, frame_indices):
        """
        Parameters
        ----------
        sequence_index
        frame_indices: Randomly-sampled indices of the frames in self.sequences[sequence_index] constituting this
                       randomly-sampled sequence.
        Returns
        -------
        (L, 3, H, W) tensor of the images for this randomly-sampled sequence.
        """
        return torch.stack(
            [
                transforms.functional.to_dtype(
                    # torchvision.io.decode_image(self.images[i]),
                    # I'm using the below instead to be robust to 1-channel or 4-channel images.
                    torchvision.io.read_image(
                        self.images[i],
                        mode=torchvision.io.image.ImageReadMode.RGB
                    ),
                    torch.float32,
                    scale=True
                )[:3]  # I'm including this here for the RGBA datasets.
                for i in self.starts[sequence_index] + frame_indices
            ]
        )

    def depths_helper(self, sequence_index, frame_indices):
        """
        Parameters
        ----------
        sequence_index
        frame_indices: Randomly-sampled indices of the frames in self.sequences[sequence_index] constituting this
                       randomly-sampled sequence.
        Returns
        -------
        (L, H, W) tensor of the depths for this randomly-sampled sequence.
        """
        # We're using torch.cat here instead of torch.stack because we want to get rid of the singleton channel dimension.
        return torch.cat(
            [
                transforms.functional.to_dtype(
                    torchvision.io.decode_image(self.depths[i]),
                    torch.float32
                    # DON'T scale the depth.
                )
                for i in self.starts[sequence_index] + frame_indices
            ]
        )

    def extrinsics_helper(self, sequence_index, frame_indices):
        """
        If you choose not to override this function, it will assume you have an (L, 4, 3) tensor for each
        self.sequences[sequence_index]['extrinsic'].

        Parameters
        ----------
        sequence_index
        frame_indices: Randomly-sampled indices of the frames in self.sequences[sequence_index] constituting this
                       randomly-sampled sequence.
        Returns
        -------
        (L, 3, 4) tensor of the 3x4 camera extrinsic matrices for this randomly-sampled sequence.
        """
        return self.extrinsics[self.starts[sequence_index] + frame_indices]

    def intrinsics_helper(self, sequence_index, frame_indices):
        """
        If you choose not to override this function, it will assume you have an (L, 4, 3) tensor for each
        self.sequences[sequence_index]['intrinsic'].

        Parameters
        ----------
        sequence_index

        Returns
        -------
        (L, 3, 3) tensor of the 3x3 camera intrinsic matrix for this randomly-sampled sequence.
        """
        return self.intrinsics[self.starts[sequence_index] + frame_indices]

    def input_helper(self, raw_images, raw_depths, raw_intrinsics):
        """
        Applies replicate padding to ViT boundaries, and zero padding to reach input_dim.

        Depths are zero-padded only.

        Scales and shifts intrinsics to match the scaling and padding of the image.

        Keep in mind that the integer coordinates are located at the corners/edges of the pixels, with the pixels
        filling in unit squares between the integer coordinates. This means the center of N pixels is at coordinate N/2.

        Parameters
        ----------
        raw_images: (L, 3, H, W)
        raw_depths: (L, H, W)
        raw_intrinsics: (L, 3, 3)

        Returns
        -------
        final_images: (L, 3, input_dim, input_dim)
        final_depths: (L, input_dim, input_dim)
        new_intrinsics: (L, 3, 3)
        rope_x: (L, num_patches)
        rope_y: (L, num_patches)
        """
        L, H, W = raw_depths.shape

        # 1. Calculate Universal Scale (Works for both Portrait and Landscape)
        scale = self.input_dim / max(H, W)
        resized_H = int(round(H * scale))
        resized_W = int(round(W * scale))

        # 2. Resize Images (Bilinear with Antialiasing)
        resized_images = F.interpolate(
            raw_images,
            size=(resized_H, resized_W),
            mode='bilinear',
            align_corners=False,
            antialias=True
        )

        # 3. Resize Depths (Nearest-Neighbor to prevent flying pixels)
        # We must unsqueeze to (L, 1, H, W) for interpolate, then squeeze back
        resized_depths = F.interpolate(
            raw_depths.unsqueeze(1),
            size=(resized_H, resized_W),
            mode='nearest'
        ).squeeze(1)

        # 4. Calculate Sub-Patch Replicate Padding
        # Find the distance to the next highest multiple of self.patch_size
        # Final mod sets padding to zero if parentheses evaluate to self.patch_size.
        pad_H = (self.patch_size - (resized_H % self.patch_size)) % self.patch_size
        pad_W = (self.patch_size - (resized_W % self.patch_size)) % self.patch_size

        # Distribute the padding evenly to top/bottom and left/right
        top_pad = pad_H // 2
        bottom_pad = pad_H - top_pad
        left_pad = pad_W // 2
        right_pad = pad_W - left_pad
        # print(left_pad, right_pad, top_pad, bottom_pad) # DEBUG

        # F.pad format works backwards from the last dim: (pad_left, pad_right, pad_top, pad_bottom)
        replicate_pad_format = (left_pad, right_pad, top_pad, bottom_pad)

        # Apply Replicate Padding to the image edges
        padded_images = F.pad(resized_images, pad=replicate_pad_format, mode="replicate")

        # Calculate Remaining Zero Padding to reach self.input_dim
        current_H = resized_H + pad_H
        current_W = resized_W + pad_W
        rem_bottom = self.input_dim - current_H
        rem_right = self.input_dim - current_W

        # Apply Zero Padding to reach exactly 512x512
        zero_pad_format = (0, rem_right, 0, rem_bottom)
        final_images = F.pad(padded_images, pad=zero_pad_format, mode="constant", value=0)

        # Apply Zero Padding directly to Depths
        # (Combines the edge padding + remaining padding into one step, leaving boundaries as 0)
        combined_depth_pad = (left_pad, right_pad + rem_right, top_pad, bottom_pad + rem_bottom)
        final_depths = F.pad(resized_depths, pad=combined_depth_pad, mode="constant", value=0)

        # 5. Scale Intrinsics and Shift Principal Point
        # raw_intrinsics needs to be cloned so we don't affect the old one assigned to output['raw_intrinsics'].
        # raw_images and raw_depths do not suffer from this problem because F.interpolate etc. create new tensors.
        new_intrinsics = raw_intrinsics.clone()

        # Scale Focal Lengths
        new_intrinsics[:, 0, 0] *= scale  # fx
        new_intrinsics[:, 1, 1] *= scale  # fy

        # Scale Principal Points AND shift them by the left/top replicate padding
        new_intrinsics[:, 0, 2] = new_intrinsics[:, 0, 2] * scale + left_pad  # cx
        new_intrinsics[:, 1, 2] = new_intrinsics[:, 1, 2] * scale + top_pad  # cy

        # 6. Calculate Intrinsic-Centered Absolute RoPE Coordinates
        # Flatten and expand to match the batch dimension: Shape (L, N_patches)
        x_flat = self.x_grid.reshape(1, -1).expand(L, -1)
        y_flat = self.y_grid.reshape(1, -1).expand(L, -1)

        # Extract the batched principal points: Shape (L, 1)
        cx = new_intrinsics[:, 0, 2].unsqueeze(1)
        cy = new_intrinsics[:, 1, 2].unsqueeze(1)

        # Vectorized Center Shift!
        # Centers the RoPE coordinates perfectly on the camera's true optical axis
        rope_x = x_flat - cx / self.patch_size
        rope_y = y_flat - cy / self.patch_size

        # rope_x and rope_y are in patch coordinates (pixel coordinates divided by the patch size).
        return final_images, final_depths, new_intrinsics, rope_x, rope_y

    def __getitem__(self, index):
        """
        This function wraps the actual getitem() function to help make it robust against data corruption/invalid frames,
        etc.
        """
        max_retries = 10
        for _ in range(max_retries):
            try:
                return self.getitem(index)
            except Exception as e:
                latest_exception = e
        print(
            f"\n{'=' * 50}\n"
            f"CRASH: Could not obtain an error-free sample after {max_retries} retries!\n"
            f"Dataset: {type(self).__name__}\n"
            f"Index: {index}\n"
            f"{'=' * 50}",
            flush=True
        )
        raise latest_exception

    def getitem(self, index):
        """
        Pick a random starting frame before the last 10 * x frames, then randomly sample x frames from the subsequent
        10 * x frames.

        Returns
        -------
        {
            "images": The resized and padded images to be fed into the model (L, 3, self.input_dim, self.input_dim)
            "depths: The resized and padded depths to be fed into the model (L, self.input_dim, self.input_dim)
            "intrinsics": The adjusted intrinsics after resizing/padding of the image (L, 3, 3)
            "extrinsics": The 3x4 extrinsic camera matrix per frame (L, 3, 4)
            "rope_x": The 2D x-coordinates to be fed into RoPE ((self.input_dim / self.patch_size)**2,)
            "rope_y": The 2D y-coordinates to be fed into RoPE; same shape as above.
        }
        If self.raw is True, the following two elements will also be included in the returned dictionary:
        {
            "raw_images": The actual raw images constituting the sequence (L, 3, H, W)
            "raw_depths": The actual raw depths constituting the sequence (L, H, W)
            "raw_intrinsics": The actual raw 3x3 camera intrinsics constituting the sequence (L, 3, 3)
        }
        """
        true_length = self.lengths[index].item()
        assert true_length > 0, f"{type(self).__name__}, sequence {index} has 0 images!"

        window_size = 10 * self.sequence_length
        if true_length < window_size:
            start = 0
        else:
            max_start = max(0, true_length - window_size)
            start = torch.randint(low=0, high=max_start + 1, size=())

        frame_indices = torch.randint(low=start, high=min(start + window_size, true_length),
                                      size=(self.sequence_length,))
        frame_indices = frame_indices.sort().values
        # print(frame_indices)
        # Ensure no frames are more than 10 apart
        if true_length >= self.sequence_length + 20:
            for i in range(1, self.sequence_length):
                if frame_indices[i] - frame_indices[i - 1] > 10:
                    frame_indices[i] = frame_indices[i - 1] + 10
        output = {}
        # ProcessPoolExecutor -> Cannot re-initialize CUDA in forked subprocess.
        with ThreadPoolExecutor() as executor:
            a = executor.submit(self.images_helper, index, frame_indices)
            b = executor.submit(self.depths_helper, index, frame_indices)
            c = executor.submit(self.intrinsics_helper, index, frame_indices)
            d = executor.submit(self.extrinsics_helper, index, frame_indices)

            images, depths, intrinsics = a.result(), b.result(), c.result()
            if self.raw:
                output['raw_images'], output['raw_depths'], output['raw_intrinsics'] = images, depths, intrinsics

            # if depths is None:
            #     dataset_name = type(self).__name__
            #     raise ValueError(
            #         f"\n{'=' * 60}\n"
            #         f"CRASH DETECTED: depths_helper returned None!\n"
            #         f"Dataset: {dataset_name}\n"
            #         f"Sequence Index: {index}\n"
            #         f"{'=' * 60}"
            #     )
            # if images is None:
            #     dataset_name = type(self).__name__
            #     raise ValueError(f"CRASH: {dataset_name} images_helper returned None at seq {index}")

            output['images'], output['depths'], output['intrinsics'], output['rope_x'], output['rope_y'] = self.input_helper(images, depths, intrinsics)
            output['extrinsics'] = d.result()

        output['dataset_name'] = type(self).__name__ # DEBUG:
        return output


if __name__ == "__main__":
    torch.set_printoptions(profile="full", linewidth=200)
    dataset = DatasetBase("/")
    fake_image = torch.tensor([
        [1, 2, 3, 4],
        # [5, 6, 7, 8],
        # [9, 10, 11, 12]
    ], dtype = torch.float32).view(1, 1, 1, 4)
    fake_depths = torch.tensor([
        [1, 1, 2, 2],
        # [3, 3, 4, 4],
        # [5, 5, 6, 6]
    ], dtype = torch.float32).unsqueeze(0)
    fake_intrinsics = torch.tensor([
        [1, 0, 2],
        [0, 1, 0.5],
        [0, 0, 1]
    ], dtype = torch.float32).unsqueeze(0)
    a, b, c, d, e = dataset.input_helper(fake_image, fake_depths, fake_intrinsics)
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)