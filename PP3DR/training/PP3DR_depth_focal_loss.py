import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.linalg import vector_norm
import torch.cuda.amp as amp
from einops import rearrange
from typing import Callable
from PP3DR_loss import PP3DR_loss


@torch.compile()
class PP3DR_loss(PP3DR_loss):
    """
    Current losses are:
     - Huber loss for 3D coordinates per frame
     - Huber loss for camera translation
     - Cosine similarity loss for camera rotation
    We're going to solve for a universal scale factor before we do all this. The scale factor will be averaged across
    all depths for each frame in a given sequence. Thus, our scale will have shape (B,).
    """
    def obtain_pred_3D_points(self, pred):
        """
        Parameters
        ----------
        pred

        Returns
        -------
        (B, L, H, W, 3) of world-frame 3D-coordinates, unprojected from pred, unscaled.
        """
        B, L, H, W = pred['log_depths'].shape
        device = pred['log_depths'].device
        y, x = torch.meshgrid(
            torch.arange(H, device = device) + 0.5,
            torch.arange(W, device = device) + 0.5,
            indexing='ij'
        ) # (H, W)
        y, x = y.view(1, 1, H, W), x.view(1, 1, H, W)
        Z = torch.exp(pred['log_depths']) # (B, L, H, W)
        fx, fy = pred['fx'].view(B, L, 1, 1), pred['fy'].view(B, L, 1, 1)
        cx, cy = pred['cx'].view(B, L, 1, 1), pred['cy'].view(B, L, 1, 1)
        X = (x - cx) * Z / fx # (B, L, H, W)
        Y = (y - cy) * Z / fy # (B, L, H, W)

        return torch.stack((X, Y, Z), dim=-1) # (B, N, H, W, 3)

    def forward(self, pred, gt):
        """
        Parameters
        ----------
        pred: {
            "log_depths": (B, L, H, W)
            "fx": (B, L)
            "fy": (B, L)
            "cx": (B, L)
            "cy": (B, L)
            "relative_camera_translations": (B, L - 1, 3)
            "relative_camera_rotations": (B, L - 1, 3, 3)
        }
        gt: {
            "inputs": (B, L, 3, H, W)
            "depths": (B, L, H, W)
            "extrinsics": (B, L, 3, 4)
            "intrinsics": (B, L, 3, 3)
        }
        The `gt` parameter is just exactly what we receive from the DataLoader. Both `pred` and `gt` are already on the
        correct GPU.

        Returns
        -------
        A single scalar, representing the loss.
        """
        # First of all, check the model predictions for infs and nans.
        torch._assert(
            (~pred['log_depths'].isfinite()).sum() + (~pred['fx'].isfinite()).sum()
            + (~pred['fy'].isfinite()).sum() + (~pred['cx'].isfinite()).sum() + (~pred['cy'].isfinite()).sum()
            + (~pred['relative_camera_translations'].isfinite()).sum()
            + (~pred['relative_camera_rotations'].isfinite()).sum() == 0,
            f"Pred has invalid values. log_depths: {(~pred['log_depths'].isfinite()).sum()}, "
            f"fx: {(~pred['fx'].isfinite()).sum()}, "
            f"fy: {(~pred['fy'].isfinite()).sum()}, "
            f"cx: {(~pred['cx'].isfinite()).sum()}, "
            f"cy: {(~pred['cy'].isfinite()).sum()}, "
            f"relative_camera_translations: {(~pred['relative_camera_translations'].isfinite()).sum()}, "
            f"relative_camera_rotations: {(~pred['relative_camera_rotations'].isfinite()).sum()}"
        )

        B, L, H, W = pred['log_depths'].shape

        gt_valid_depth_mask = torch.isfinite(gt['depths']) & (gt['depths'] != 0)
        gt_invalid_depth_mask = ~gt_valid_depth_mask

        # First, normalise the MEDIAN ground-truth depth to 1.
        # Calculate the median over only valid elements by filling all invalid with NaN and using torch.nanmedian().
        median_depths = torch.nanmedian(
            gt['depths'].masked_fill(gt_invalid_depth_mask, float('nan')).flatten(1),
            dim=-1
        )[0] # (B)
        torch._assert(
            median_depths.isnan().sum() == 0,
            f"{median_depths.isnan().sum()} batch's gt depths are completely invalid."
        )
        gt['depths'] = gt['depths'] / median_depths.view(B, 1, 1, 1)
        # Sanitize GT depths before they touch the predictions by setting all invalid values to the median depth, 1.
        gt['depths'] = gt['depths'].masked_fill(gt_invalid_depth_mask, 1)
        weights = 2 / (1 + gt['depths']) # multiplied by 2 so the median weight is 1
        # Multiply weights by gt_valid_depth_mask to zero the weights of any points with invalid gt depths.
        weights = weights.masked_fill(gt_invalid_depth_mask, 0)

        pred_points, gt_points = self.obtain_pred_3D_points(pred), self.obtain_gt_3D_points(gt)
        if self.scale:
            scale = self.calculate_scale(pred_points, gt_points, weights) # (B, 1)
        else:
            scale = torch.tensor([1], device = "cuda").expand(B)

        """
        Weighted Huber loss for 3D point coordinates (per-frame, in camera coordinates)
        """
        total_point_loss = F.huber_loss(
            (pred_points * scale.view(B, 1, 1, 1, 1)),
            gt_points,
            reduction='none'
        ) * weights.view(B, L, H, W, 1) # Broadcasts against all 3 coordinates of each point.
        # We shouldn't have to masked_fill() total_point_loss here, as pred and gt should all be valid by this point.
        point_loss = total_point_loss.sum() / (gt_valid_depth_mask.sum() * 3) # *3 for x, y, and z
        torch._assert(
            point_loss.isfinite(),
            f"Point loss invalid ({point_loss})\ttotal_point_loss = {total_point_loss}"
        )
        """
        Normal direction loss
        """
        normal_loss = self.normal_loss(
            points=pred_points,
            gt_points=gt_points,
            mask=gt_valid_depth_mask,
            gt_depths=gt['depths']
        )

        gt_relative_rotations, gt_relative_translations = self.obtain_gt_relative_poses(gt['extrinsics'])
        """
        Huber loss for relative camera translation.
        """
        # If any of the 3 coordinates for a GT camera translation are invalid, we want to ignore that entire translation.
        gt_valid_translation_mask = gt_relative_translations.isfinite().all(dim=-1, keepdim=True) # (B, L - 1, 1)
        gt_invalid_translation_mask = ~gt_valid_translation_mask
        gt_relative_translations = gt_relative_translations.masked_fill(gt_invalid_translation_mask, 0)
        total_translation_loss = F.huber_loss(
            scale.view(B, 1, 1) * pred['relative_camera_translations'],
            gt_relative_translations / median_depths.view(B, 1, 1),
            reduction='none'
        )
        total_translation_loss = total_translation_loss.masked_fill(gt_invalid_translation_mask, 0)
        translation_loss = total_translation_loss.sum() / ((gt_valid_translation_mask).sum() * 3) # *3 for x, y, and z
        torch._assert(translation_loss.isfinite(), f"Translation loss invalid ({translation_loss})")

        """
        Cosine similarity loss for relative camera rotation
        cos = (Tr(R_1^TR_2) - 1)/2. Since Tr(R_1^TR_2) is equal to the inner product of R_1 and R_2, our loss is:
        (3 - <R_1, R_2>)/2
        """
        # (B, L - 1)
        gt_rotation_valid_mask = (
                gt_relative_rotations.isfinite().all(dim=-1).all(dim=-1)
                                  & (gt_relative_rotations.flatten(start_dim=2).abs().max(dim=-1).values <= 1.05)
        )
        gt_rotation_invalid_mask = ~gt_rotation_valid_mask
        identity_matrix = torch.eye(
            3,
            device=gt_relative_rotations.device,
            dtype=gt_relative_rotations.dtype
        ).view(1, 1, 3, 3)
        gt_relative_rotations = torch.where(
            gt_rotation_valid_mask.view(B, L - 1, 1, 1),
            gt_relative_rotations,
            identity_matrix
        )
        trace = (pred['relative_camera_rotations'] * gt_relative_rotations).sum(dim=(-2, -1)) # (B, L - 1)
        # assert (~trace.isfinite()).sum() == 0, f"trace has {(~trace.isfinite()).sum()} invalid elements"

        raw_rotation_loss = torch.clamp((3 - trace) / 2, min = 0)
        rotation_loss = raw_rotation_loss.masked_fill(gt_rotation_invalid_mask, 0).sum() / gt_rotation_valid_mask.sum()
        torch._assert(rotation_loss.isfinite(), f"Rotation loss invalid ({rotation_loss})")

        total_loss = 10 * point_loss + 0.1 * normal_loss + 10 * translation_loss + rotation_loss
        # total_loss = 10 * point_loss + 0.1 * normal_loss
        return total_loss, dict(
            total_loss=total_loss,
            point_loss=point_loss,
            normal_loss=normal_loss,
            translation_loss=translation_loss,
            rotation_loss=rotation_loss
        )