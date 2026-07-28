import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.linalg import vector_norm
import torch.cuda.amp as amp
from einops import rearrange
from typing import Callable
import math


def _smooth(err: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8):
    """Safely computes the angular difference, immune to the norm(0) singularity."""
    v1, v2 = v1.to(torch.float32), v2.to(torch.float32)
    # Normalize both vectors first!
    # Otherwise, the tiny unnormalized cross-product magnitudes get completely overpowered by `eps`.
    v1 = F.normalize(v1, dim=-1, eps=1e-6)
    v2 = F.normalize(v2, dim=-1, eps=1e-6)
    cross = torch.linalg.cross(v1, v2, dim=-1)
    cross_norm = torch.sqrt(torch.sum(cross.square(), dim=-1) + eps)
    dot = (v1 * v2).sum(dim=-1)
    return torch.atan2(cross_norm, dot).to(v1.dtype)


def depth_edge(depth: torch.Tensor, rtol: float = 0.03) -> torch.Tensor:
    """
    Rewritten to avoid in-place bitwise mutations (|=) on zeros_like tensors.
    Dynamo/Triton loses track of the torch.bool dtype during in-place mutations
    and attempts to execute bitwise operations on float32s, causing a crash.
    F.pad is safer, compiles perfectly, and executes faster on the GPU.
    """
    dy = torch.abs(depth[..., 1:, :] - depth[..., :-1, :]) / depth[..., :-1, :].clamp_min(1e-6)
    dx = torch.abs(depth[..., :, 1:] - depth[..., :, :-1]) / depth[..., :, :-1].clamp_min(1e-6)

    dy_mask = dy > rtol
    dx_mask = dx > rtol

    # Pad the masks to align back to the original depth shape.
    # F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
    dy_up = F.pad(dy_mask, (0, 0, 0, 1))
    dy_down = F.pad(dy_mask, (0, 0, 1, 0))
    dx_left = F.pad(dx_mask, (0, 1, 0, 0))
    dx_right = F.pad(dx_mask, (1, 0, 0, 0))

    # Pure boolean combination entirely bypasses in-place mutation bugs
    return dy_up | dy_down | dx_left | dx_right

@torch.compile()
class PP3DR_loss(nn.Module):
    """
    Current losses are:
     - Huber loss for 3D coordinates per frame
     - Huber loss for camera translation
     - Cosine similarity loss for camera rotation
    We're going to solve for a universal scale factor before we do all this. The scale factor will be averaged across
    all depths for each frame in a given sequence. Thus, our scale will have shape (B,).
    """
    def __init__(self,
        scale = False,
        # point_loss = True,
        # normal_loss = True,
        # translation_loss = True,
        # rotation_loss = True,
    ):
        super().__init__()
        self.scale = scale
        # self.point_loss = point_loss
        # self.normal_loss = normal_loss
        # self.translation_loss = translation_loss
        # self.rotation_loss = rotation_loss
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
        # Flatten all points, per batch. Detach pred first!
        pred = pred_pts.detach().flatten(1, -2) # New shape: (B, NHW, 3)
        gt = gt_pts.flatten(1, -2)  # New shape: (B, NHW, 3)
        # Repeat (reallocate), because we're going to perform manual per-coordinate masking later.
        weights = weights.flatten(1).unsqueeze(-1).repeat(1, 1, 3)  # New shape: (B, NHW, 3)

        # Force the weights of any near-zero coordinates to zero, so we don't end up dividing by them.
        valid_mask = pred.abs() > 1e-8
        weights = weights * valid_mask
        assert weights.isnan().sum() == 0, f"{weights.isnan().sum()} NaNs in weights"
        ratios = gt / pred # (B, NHW, 3)
        # Fill all invalid ratios with 0.
        ratios = ratios.masked_fill(~valid_mask, 0)
        effective_weights = torch.flatten(weights * pred.abs(), start_dim = 1) # (B, NHW * 3)
        # assert effective_weights.isnan().sum() == 0, f"{effective_weights.isnan().sum()} NaNs in effective_weights"

        # Sort the ratios (O(N log N))
        sorted_ratios, sort_indices = torch.sort(ratios.flatten(start_dim=1))
        sorted_weights = torch.gather(effective_weights, dim=-1, index=sort_indices) # (B, NHW * 3)

        # The median is the first ratio where the cumulative sum passes 50%
        # Casting to float32 because we're performing additions here
        cum_weights = torch.cumsum(sorted_weights.to(torch.float32), dim=-1) # (B, NHW * 3)
        # We keep the last dimension as a singleton dimension because that's what torch.searchsorted requires.
        half_total_weight = cum_weights[:, -1:] / 2.0 # (B)
        # Binary search
        median_idx = torch.searchsorted(cum_weights, half_total_weight) # (B)

        # I don't think the below case is possible.
        # Handle the edge case where median_idx hits the end of the array
        # median_idx = torch.clamp(median_idx, max=len(sorted_ratios) - 1)

        # sorted_ratios has shape (B, 3LHW). median_idx has shape (B, 1).
        assert median_idx.max() < sorted_ratios.shape[1] and median_idx.min() > 0, \
            f"median_idx min/max = {median_idx.min().item()} / {median_idx.max().item()}\n{median_idx}"
        return torch.gather(sorted_ratios, dim=1, index=median_idx) # (B, 1)

    def obtain_gt_relative_poses(self, extrinsics):
        """
        Parameters
        ----------
        extrinsics: gt['extrinsics']

        Returns
        -------
        Relative pose changes between frame i and frame i + 1 for i in [0, L - 1).
        Our extrinsics are c2w matrices, and we'd like to obtain the next extrinsic rotation matrix by left-multiplying
         the current rotation matrix by the following relative rotation matrix. Thus, the relative rotation from A to B
         is BA^-1.
        As for the translation, we'll just add on the next relative translation to the current one. Thus, the relative
         translation from A to B is B - A.
        relative_rotations: (B, L - 1, 3, 3), relative_translations: (B, L - 1, 3)
        """
        current_rotations = extrinsics[:, :-1, :, :3]  # (B, L-1, 3, 3)
        next_rotations = extrinsics[:, 1:, :, :3]  # (B, L-1, 3, 3)
        relative_rotations = next_rotations @ current_rotations.transpose(-1, -2)

        current_translations = extrinsics[:, :-1, :, 3]  # (B, L-1, 3)
        next_translations = extrinsics[:, 1:, :, 3]  # (B, L-1, 3)
        relative_translations = next_translations - current_translations

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
        device = gt['depths'].device
        y, x = torch.meshgrid(
            torch.arange(H, device = device) + 0.5,
            torch.arange(W, device = device) + 0.5,
            indexing='ij'
        ) # (H, W)
        y, x = y.view(1, 1, H, W), x.view(1, 1, H, W)
        Z = gt['depths'] # (B, L, H, W)
        fx, fy = gt['intrinsics'][..., 0, 0].view(B, L, 1, 1), gt['intrinsics'][..., 1, 1].view(B, L, 1, 1)
        cx, cy = gt['intrinsics'][..., 0, 2].view(B, L, 1, 1), gt['intrinsics'][..., 1, 2].view(B, L, 1, 1)
        X = (x - cx) * Z / fx # (B, L, H, W)
        Y = (y - cy) * Z / fy # (B, L, H, W)

        return torch.stack((X, Y, Z), dim=-1) # (B, N, H, W, 3)

    def normal_loss(self, points, gt_points, mask, gt_depths):
        not_edge = ~depth_edge(gt_depths, rtol=0.03)
        mask = mask & not_edge

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[
            ..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = rightup - rightdown
        leftxdown = leftup - rightup
        downxright = leftdown - leftup
        rightxup = rightdown - leftdown

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[
            ..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], \
        gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]

        # Explicitly cast the final boolean masks to float32 before multiplication to prevent Dynamo type inference bugs
        loss = mask_upxleft.to(torch.float32) * _smooth(
            angle_diff_vec3(torch.cross(upxleft, leftdown - rightdown, dim=-1),
                            torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)).clamp(MIN_ANGLE,
                                                                                                              MAX_ANGLE),
            beta=BETA_RAD) \
               + mask_leftxdown.to(torch.float32) * _smooth(
            angle_diff_vec3(torch.cross(leftxdown, rightdown - rightup, dim=-1),
                            torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)).clamp(MIN_ANGLE,
                                                                                                          MAX_ANGLE),
            beta=BETA_RAD) \
               + mask_downxright.to(torch.float32) * _smooth(
            angle_diff_vec3(torch.cross(downxright, rightup - leftup, dim=-1),
                            torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)).clamp(MIN_ANGLE,
                                                                                                        MAX_ANGLE),
            beta=BETA_RAD) \
               + mask_rightxup.to(torch.float32) * _smooth(
            angle_diff_vec3(torch.cross(rightxup, leftup - leftdown, dim=-1),
                            torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)).clamp(MIN_ANGLE,
                                                                                                            MAX_ANGLE),
            beta=BETA_RAD)

        # Added + 1e-6 to prevent division by zero if an entire patch/batch is masked out!
        return loss.sum() / (mask.sum() * 4 + 1e-6)

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
            (~pred['XY_rays'].isfinite()).sum() + (~pred['log_depths'].isfinite()).sum()
            + (~pred['relative_camera_translations'].isfinite()).sum()
            + (~pred['relative_camera_rotations'].isfinite()).sum() == 0,
            f"Pred has invalid values. XY_rays: {(~pred['XY_rays'].isfinite()).sum()}, "
            f"log_depths: {(~pred['log_depths'].isfinite()).sum()}, "
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
        Huber Loss on the angle for relative camera rotation
        cos = (Tr(R_1^TR_2) - 1)/2. Since Tr(R_1^TR_2) is equal to the inner product of R_1 and R_2, our loss is:
        Huber( arccos( (<R_1, R_2> - 1) / 2 ) )
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
        cosine = torch.clamp(((trace - 1.0) / 2.0).to(torch.float32), min=-1.0 + 1e-6, max=1.0 - 1e-6)
        raw_rotation_loss = F.huber_loss(torch.acos(cosine).to(trace.dtype), torch.zeros_like(cosine, device="cuda"))
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

"""
L1 or Huber loss for rotation loss:
# Calculate the trace (Inner product of the two matrices)
trace = (pred['relative_camera_rotations'] * gt_relative_rotations).sum(dim=(-2, -1)) # (B, L - 1)

# 1. Convert to float32 to protect the clamp from bfloat16 rounding
cosine = ((trace - 1.0) / 2.0).to(torch.float32)

# 2. Safely clamp strictly inside the valid domain to prevent the acos(-Inf) gradient explosion
cosine = torch.clamp(cosine, min=-1.0 + 1e-6, max=1.0 - 1e-6)

# 3. Calculate the exact angular error in radians, then cast back to network precision
rot_ang_err = torch.acos(cosine).to(trace.dtype)

# 4. Mask and average (Added the 1e-6 safety net to the denominator!)
rotation_loss = rot_ang_err.masked_fill(gt_rotation_invalid_mask, 0.0).sum() / (gt_rotation_valid_mask.sum() + 1e-6)

torch._assert(rotation_loss.isfinite(), f"Rotation loss invalid ({rotation_loss})")
"""