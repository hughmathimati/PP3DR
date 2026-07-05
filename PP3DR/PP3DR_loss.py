import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.linalg import vector_norm
import torch.cuda.amp as amp
from einops import rearrange
from typing import Callable

@torch.compile() # Uncomment once you've made sure it works. Do we need dynamic=True? Documentation says don't use it...
class PP3DR_loss(nn.Module):
    """
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
        weights: Per-point weights (e.g. for depth-weighted loss) (B, N, H, W)

        Returns
        -------
        The scale factor minimising L1 3D point coordinate loss across all frames (B)
        """
        # Flatten all points, per batch
        pred = pred_pts.flatten(1, -2) # New shape: (B, NHW, 3)
        gt = gt_pts.flatten(1, -2)  # New shape: (B, NHW, 3)
        # Repeat (reallocate), because we're going to perform manual per-coordinate masking later.
        weights = weights.flatten(1).unsqueeze(-1).repeat(1, 1, 3)  # New shape: (B, NHW, 3)

        # Force the weights of any near-zero coordinates to zero, so we don't end up dividing by them.
        valid_mask = pred.abs() > 1e-8
        weights = weights * valid_mask
        # Now simply clamp pred to a minimum of 1e-8, and we're all good.
        ratios = torch.flatten(gt / torch.clamp(pred, min = 1e-8), start_dim = 1) # (B, 3NHW)

        """
        pred is not going to be NaN. gt could be NaN, meaning certain elements of ratios could be NaN.
        The trick is that we never actually multiply by ratios; we pretty much only work with effective_weights, and
        since the effective weights of invalid gt points are zero, they won't affect our cumsum and this won't affect
        our computed median.
        """
        effective_weights = torch.flatten(weights * pred.abs(), start_dim = 1) # (B, 3NHW)

        # Sort the ratios (O(N log N))
        sorted_ratios, sort_indices = torch.sort(ratios)
        sorted_weights = torch.gather(effective_weights, dim=-1, index=sort_indices) # (B, 3NHW)

        # The median is the first ratio where the cumulative sum passes 50%
        cum_weights = torch.cumsum(sorted_weights, dim=-1) # (B, 3NHW)
        # We keep the last dimension as a singleton dimension because that's what torch.searchsorted requires.
        half_total_weight = cum_weights[:, -1:] / 2.0 # (B)
        # Binary search
        median_idx = torch.searchsorted(cum_weights, half_total_weight) # (B)

        # I don't think the below case is possible.
        # Handle the edge case where median_idx hits the end of the array
        # median_idx = torch.clamp(median_idx, max=len(sorted_ratios) - 1)

        return sorted_ratios[:, median_idx] # (B)

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
        relative_rotations = torch.empty(B, L - 1, 3, 3, device = "cuda")
        relative_translations = torch.empty(B, L - 1, 3, device = "cuda")
        # current_rotation: (B, 1, 3, 3), current_translation: (B, 1, 3, 1)
        current_rotation, current_translation = extrinsics[:, 0, :, :3], extrinsics [:, 0, :, 3]
        for i in range(0, L - 1):
            next_rotation, next_translation = extrinsics[:, i + 1, :, :3], extrinsics[:, i + 1, :, 3]
            # current_rotation.transpose(-1, -2), not current_rotation.T, because we need to keep the batch dimension.
            relative_rotations[:, i] = next_rotation @ current_rotation.transpose(-1, -2)
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
        depths = torch.exp(pred['log_depths']).unsqueeze(-1)
        xy = pred['XY_rays'] * depths
        return torch.cat((xy, depths), dim = -1)

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
        gt['intrinsic'] = gt['intrinsic'].to("cuda", non_blocking = True)
        fx, fy, cx, cy = gt['intrinsic'][:, 0, 0], gt['intrinsic'][:, 1, 1], gt['intrinsic'][:, 0, 2], gt['intrinsic'][:, 1, 2]
        X = (x.to("cuda", non_blocking = True) - cx) * Z / fx
        Y = (y.to("cuda", non_blocking = True) - cy) * Z / fy

        return torch.stack((X, Y, Z), dim=-1) # (B, N, H, W, 3)

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
            "images": (B, L, 3, H, W)
            "depths": (B, L, H, W)
            "extrinsics": (B, L, 3, 4)
            "intrinsic": (B, 3, 3)
        }
        The `gt` parameter is just exactly what we receive from the DataLoader.

        Returns
        -------
        A single scalar, representing the loss.
        """
        B, L, H, W = pred['log_depths'].shape
        final_loss = 0

        gt['depths'] = gt['depths'].to("cuda", non_blocking = True)
        gt_valid_depth_mask = torch.isfinite(gt['depths']) & (gt['depths'] != 0)

        # First, normalise the MEDIAN ground-truth depth to 1.
        median_depths, _ = torch.median(gt['depths'].flatten(1), -1) # (B)
        gt['depths'] = gt['depths'] / median_depths.view(B, 1, 1, 1)
        weights = 2 / (1 + gt['depths']) # multiplied by 2 so the median weight is 1

        # Multiply weights by gt_valid_depth_mask to zero the weights of any points with invalid gt depths.
        weights = weights * gt_valid_depth_mask

        pred_points, gt_points = self.obtain_pred_3D_points(pred), self.obtain_gt_3D_points(gt)
        scale = self.calculate_scale(pred_points, gt_points, weights) # (B)

        """
        Weighted Huber loss for 3D point coordinates (per-frame, in camera coordinates)
        Weird views ensure scale and view are broadcast correctly.
        It's okay for us to simply multiply by weights at the end, because whatever NaNs come up inside huber_loss
        are contained only in that element itself, because reduction='none'.
        """
        final_loss += torch.mean(
            F.huber_loss(
                pred_points * scale.view(B, 1, 1, 1, 1),
                gt_points,
                reduction='none'
            ) * weights.view(B, L, H, W, 1)
        )

        # (Unweighted) Huber loss for relative camera translation. Hopefully none of the gt camera poses are NaN...
        gt_relative_rotations, gt_relative_translations = self.obtain_gt_relative_poses(gt['extrinsics'].to("cuda", non_blocking = True))
        final_loss += F.huber_loss(scale.view(B, 1, 1) * pred['relative_camera_translations'], gt_relative_translations)

        # Cosine similarity loss for relative camera rotation
        # cos = (Tr(R_1^TR_2) - 1)/2. Since Tr(R_1^TR_2) is equal to the inner product of R_1 and R_2, our loss is:
        # (3 - <R_1, R_2>)/2
        trace = (pred['relative_camera_rotations'] * gt_relative_rotations).sum(dim=(-2, -1)) # (B, L)
        final_loss += torch.clamp((3 - trace) / 2, min = 0).mean()

        return final_loss