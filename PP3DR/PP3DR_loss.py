import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.linalg import vector_norm
import torch.cuda.amp as amp
from einops import rearrange
from typing import Callable

#torch.compile() # Uncomment once you've made sure it works. Do we need dynamic=True? Documentation says don't use it...
class PP3DR_loss(nn.Module):
    """
    Current losses are:
     - L1 loss for 3D world coordinates per frame
     - Some translation pose loss?
     - Cosine similarity on rotation pose loss?
    Right now, I'm just doing a sanity check to see if my model can at least overfit on NRGBD. I'll stick with:
     - Huber loss for 3D coordinates per frame
     - Huber loss for camera translation
     - Cosine similarity loss for camera rotation
    We're going to solve for a universal scale factor before we do all this. The scale factor will be averaged
    """
    def __init__(self):
        super().__init__()
        pass

    def calculate_scale(self, pred_pts, gt_pts, weights):
        """
        Parameters
        ----------
        pred_pts: Unprojected predicted points in each camera's own reference frame (B, N, H, W, 3)
        gt_pts: Unprojected ground-truth points in each camera's own reference frame (B, N, H, W, 3)
        weights: Per-point weights (e.g. for depth-weighted loss) (B, N, H, W, 1)

        Returns
        -------
        The scale factor minimising L1 3D point coordinate loss across all frames (B)
        """
        # Flatten all points, per batch
        pred = pred_pts.flatten(1) # New shape: (B, 3NHW)
        gt = gt_pts.flatten(1)  # New shape: (B, 3NHW)
        weights = weights.flatten(1)  # New shape: (B, NHW)

        # Mask out near-zero predictions to prevent division by zero
        valid_mask = pred.abs() > 1e-8
        pred = pred[valid_mask]
        gt = gt[valid_mask]
        weights = weights[valid_mask]
        ratios = gt / pred # (B, NHW)
        effective_weights = weights * pred.abs() # (B, NHW)

        # Sort the ratios (O(N log N))
        sorted_ratios, sort_indices = torch.sort(ratios)
        sorted_weights = effective_weights[sort_indices] # (B, NHW)

        # The median is the first ratio where the cumulative sum passes 50%
        cum_weights = torch.cumsum(sorted_weights, dim=1) # (B, NHW)
        half_total_weight = cum_weights[:, -1] / 2.0 # (B)
        # Binary search
        median_idx = torch.searchsorted(cum_weights, half_total_weight) # (B)

        # I don't think the below case is possible.
        # Handle the edge case where median_idx hits the end of the array
        # median_idx = torch.clamp(median_idx, max=len(sorted_ratios) - 1)

        return sorted_ratios[median_idx] # (B)

    def obtain_gt_relative_poses(self, extrinsics):
        """
        Parameters
        ----------
        extrinsics: gt['extrinsics']

        Returns
        -------
        Relative pose changes between frame i and frame i + 1 for i in [0, L - 1).
        Specifically, our extrinsics are c2w matrices, and we'd like to obtain the next extrinsic by left-multiplying
         the current extrinsic by the following relative extrinsic. Thus, the relative extrinsic from A to B is
         BA^-1.
        relative_rotations: (B, L - 1, 3, 3), relative_translations: (B, L - 1, 3)
        """
        B, L = extrinsics.shape[:2]
        relative_rotations = torch.empty(B, L - 1, 3, 3)
        relative_translations = torch.empty(B, L - 1, 3)
        # current_rotation: (B, 1, 3, 3), current_translation: (B, 1, 3, 1)
        current_rotation, current_translation = extrinsics[:, 0, :, :3], extrinsics [:, 0, :, 3]
        for i in range(0, L - 1):
            next_rotation, next_translation = extrinsics[:, i + 1, :, :3], extrinsics[:, i + 1, :, 3]
            relative_rotations[:, i] = next_rotation @ current_rotation.T
            relative_translations[:, i] = next_translation - current_translation
        return relative_rotations, relative_translations

    def obtain_pred_3D_points(self, pred):
        """
        Parameters
        ----------
        pred

        Returns
        -------
        (B, L, H, W, 3) of 3D-coordinates for each pixel's point, in the given frame's 3D coordinate system, UNSCALED.
        We won't scale yet, because we need the initial points to calculate the scale itself.
        """
        B, L, H, W = pred['log_depths'].shape
        output = torch.empty(B, L, H, W, 3)
        output[..., 2] = torch.exp(pred['log_depths'])
        output[..., :2] = pred['XY_rays'] * output[..., 2]
        return output

    def obtain_gt_3D_points(self, gt):
        """
        Parameters
        ----------
        gt

        Returns
        -------
        (B, L, H, W, 3) of world-frame 3D-coordinates, unprojected from gt, UN-NORMALISED.
        We won't normalize here; we'll normalize in the main function.
        """
        B, L, H, W = gt['depths'].shape
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        Z = gt['depths']
        fx, fy, cx, cy = gt['intrinsic'][0, 0], gt['intrinsic'][1, 1], gt['intrinsic'][0, 2], gt['intrinsic'][1, 2]
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy

        # Stack into a (3, H, W) point cloud tensor
        return torch.stack((X, Y, Z), dim=0)

    def forward(self, pred, gt):
        """
        Parameters
        ----------
        pred: {
            "XY_rays": (B, L, H, W, 2)
            "log_depths": (B, L, H, W)
            "relative_camera_translations": (B, L - 1, 3)
            "relative_camera_rotations": (B, L - 1, 3, 3)
        }
        gt: {
            "images": (B, L, H, W, 3)
            "depths": (B, L, H, W)
            "extrinsics": (B, L, 3, 4)
            "intrinsic": (B, 3, 3)
        }
        The `gt` parameter is just exactly what we receive from the DataLoader.

        Returns
        -------
        A single scalar, representing the loss.
        """
        final_loss = 0
        B, L, H, W = pred['log_depths'].shape
        # First, normalise the MEDIAN ground-truth depth to 1.
        gt['depths'] = gt['depths'] / torch.median(gt['depths'].flatten(1)).view(B, 1, 1, 1)
        weights = 2 / (1 + gt['depths']) # multiplied by 2 so the median weight is 1
        pred_points, gt_points = self.obtain_pred_3D_points(pred), self.obtain_gt_3D_points(gt)
        scale = self.calculate_scale(pred_points, gt_points, weights) # (B)
        # Weighted Huber loss for 3D point coordinates (per-frame, in camera coordinates)
        # Weird views ensure scale and view are broadcast correctly
        final_loss += torch.mean(
            F.huber_loss(
                pred_points * scale.view(B, 1, 1, 1, 1),
                gt_points,
                reduction='none'
            ) * weights.view(B, 1, 1, 1, 1)
        )
        # (Unweighted) Huber loss for relative camera translation
        gt_relative_rotations, gt_relative_translations = self.obtain_gt_relative_poses(gt['extrinsics'])
        final_loss += F.huber_loss(pred['relative_camera_translations'], gt_relative_translations)
        # Cosine similarity loss for relative camera rotation
        # cos = (Tr(R_1^TR_2) - 1)/2. Since Tr(R_1^TR_2) is equal to the inner product of R_1 and R_2, our loss is:
        # (3 - <R_1, R_2>)/2
        trace = (pred['relative_camera_rotations'] * gt_relative_rotations).sum(dim=(-2, -1)) # (B, L)
        final_loss += torch.clamp((3 - trace) / 2, min = 0).mean()