import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.linalg import vector_norm
import torch.cuda.amp as amp
from einops import rearrange
from typing import Callable
import math

class HughberLoss(nn.Module):
    def __init__(self, beta = 1e-2, mse_weight=0.5):
        super().__init__()
        self.beta = beta
        self.mse_weight = mse_weight

    def _forward2(self, x, gt):
        """
        NOTE: Performs reduction='none'.
        """
        return F.smooth_l1_loss(x, gt, reduction="none", beta=self.beta) + self.mse_weight * F.mse_loss(x, gt, reduction="none")

    def _forward1(self, error):
        """
        NOTE: Performs reduction='none'.
        Expects a single ABSOLUTE `error` tensor (e.g., predictions - gt).
        """

        # Calculates the exact formula from your screenshot seamlessly
        square = torch.square(error)
        loss = torch.where(
            error < self.beta,
            0.5 * square / self.beta,
            error - 0.5 * self.beta
        ) + self.mse_weight * square

        return loss

    def forward(self, *args):
        if len(args) == 1:
            return self._forward1(*args)
        else:
            return self._forward2(*args)

def _smooth(err: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8):
    """Safely computes the angular difference, immune to the norm(0) singularity."""
    v1, v2 = v1.to(torch.float32), v2.to(torch.float32)
    # Normalize both vectors first!
    v1 = F.normalize(v1, dim=-1, eps=1e-6)
    v2 = F.normalize(v2, dim=-1, eps=1e-6)
    cross = torch.linalg.cross(v1, v2, dim=-1)
    cross_norm = torch.sqrt(torch.sum(cross.square(), dim=-1) + eps)
    dot = (v1 * v2).sum(dim=-1)
    return torch.atan2(cross_norm, dot).to(v1.dtype)

def depth_edge(depth: torch.Tensor, rtol: float = 0.03) -> torch.Tensor:
    dy = torch.abs(depth[..., 1:, :] - depth[..., :-1, :]) / depth[..., :-1, :].clamp_min(1e-6)
    dx = torch.abs(depth[..., :, 1:] - depth[..., :, :-1]) / depth[..., :, :-1].clamp_min(1e-6)

    dy_mask = dy > rtol
    dx_mask = dx > rtol

    # F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
    dy_up = F.pad(dy_mask, (0, 0, 0, 1))
    dy_down = F.pad(dy_mask, (0, 0, 1, 0))
    dx_left = F.pad(dx_mask, (0, 1, 0, 0))
    dx_right = F.pad(dx_mask, (1, 0, 0, 0))

    return dy_up | dy_down | dx_left | dx_right

@torch.compile()
class PP3DR_loss(nn.Module):
    def __init__(self, scale=False, median_depth=10):
        super().__init__()
        self.scale = scale
        self.median_depth = median_depth
        self.hughber_loss = HughberLoss()

    def calculate_scale(self, pred_pts, gt_pts, weights):
        pred = pred_pts.detach().flatten(1, -2)
        gt = gt_pts.flatten(1, -2)
        weights = weights.flatten(1).unsqueeze(-1).expand(-1, -1, 3)

        valid_mask = pred.abs() > 1e-8
        weights = weights * valid_mask
        ratios = gt / pred
        ratios = ratios.masked_fill(~valid_mask, 0)
        effective_weights = torch.flatten(weights * pred.abs(), start_dim=1)

        sorted_ratios, sort_indices = torch.sort(ratios.flatten(start_dim=1))
        sorted_weights = torch.gather(effective_weights, dim=-1, index=sort_indices)

        cum_weights = torch.cumsum(sorted_weights.to(torch.float32), dim=-1)
        half_total_weight = cum_weights[:, -1:] / 2.0
        median_idx = torch.searchsorted(cum_weights, half_total_weight)

        return torch.gather(sorted_ratios, dim=1, index=median_idx)

    def obtain_gt_relative_poses(self, extrinsics):
        current_rotations = extrinsics[:, :-1, :, :3]
        next_rotations = extrinsics[:, 1:, :, :3]
        relative_rotations = next_rotations @ current_rotations.transpose(-1, -2)

        current_translations = extrinsics[:, :-1, :, 3]
        next_translations = extrinsics[:, 1:, :, 3]
        relative_translations = next_translations - current_translations

        return relative_rotations, relative_translations

    def obtain_pred_3D_points(self, pred, gt):
        B, L, H, W = pred['log_depths'].shape
        device = pred['log_depths'].device
        y, x = torch.meshgrid(
            torch.arange(H, device=device) + 0.5,
            torch.arange(W, device=device) + 0.5,
            indexing='ij'
        )
        y, x = y.view(1, 1, H, W), x.view(1, 1, H, W)
        Z = torch.exp(pred['log_depths'])
        focal_length = pred['focal_length'].view(B, 1, 1, 1).expand(-1, L, -1, -1)
        cx, cy = gt['intrinsics'][..., 0, 2].view(B, L, 1, 1), gt['intrinsics'][..., 1, 2].view(B, L, 1, 1)
        X = (x - cx) * Z / focal_length
        Y = (y - cy) * Z / focal_length

        return torch.stack((X, Y, Z), dim=-1)

    def obtain_gt_3D_points(self, gt):
        B, L, H, W = gt['depths'].shape
        device = gt['depths'].device
        y, x = torch.meshgrid(
            torch.arange(H, device=device) + 0.5,
            torch.arange(W, device=device) + 0.5,
            indexing='ij'
        )
        y, x = y.view(1, 1, H, W), x.view(1, 1, H, W)
        Z = gt['depths']
        fx, fy = gt['intrinsics'][..., 0, 0].view(B, L, 1, 1), gt['intrinsics'][..., 1, 1].view(B, L, 1, 1)
        cx, cy = gt['intrinsics'][..., 0, 2].view(B, L, 1, 1), gt['intrinsics'][..., 1, 2].view(B, L, 1, 1)
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy

        return torch.stack((X, Y, Z), dim=-1)

    def initialize(self, pred, gt):
        gt['depths'], gt['extrinsics'], gt['intrinsics'] = gt['depths'].cuda(), gt['extrinsics'].cuda(), gt[
            'intrinsics'].cuda()
        B = gt['depths'].shape[0]

        gt_valid_depth_mask = torch.isfinite(gt['depths']) & (gt['depths'] > 0)
        gt_invalid_depth_mask = ~gt_valid_depth_mask

        median_depths = torch.nanmedian(
            gt['depths'].masked_fill(gt_invalid_depth_mask, float('nan')).flatten(start_dim=1),
            dim=-1
        )[0]
        torch._assert(
            median_depths.isnan().sum() == 0,
            f"{median_depths.isnan().sum()} batch's gt depths are completely invalid."
        )
        gt['depths'] = self.median_depth * gt['depths'] / median_depths.view(B, 1, 1, 1)
        gt['depths'] = gt['depths'].masked_fill(gt_invalid_depth_mask, self.median_depth)

        weights = 2 * self.median_depth / (self.median_depth + gt['depths'])
        weights = weights.masked_fill(gt_invalid_depth_mask, 0)

        pred_points, gt_points = self.obtain_pred_3D_points(pred, gt), self.obtain_gt_3D_points(gt)
        if self.scale:
            scale = self.calculate_scale(pred_points, gt_points, weights)
        else:
            scale = torch.tensor([1], device="cuda").expand(B)

        return pred_points, gt_points, scale, weights, median_depths, gt_valid_depth_mask

    def point_loss(self, pred_points, gt_points, scale, weights, gt_valid_depth_mask, s):
        B, L, H, W = pred_points.shape[:4]

        raw_error = F.smooth_l1_loss(
            pred_points * scale.view(B, 1, 1, 1, 1),
            gt_points,
            reduction='none',
            beta=1e-2
        ) * weights.view(B, L, H, W, 1)
        # raw_error = self.hughber_loss(
        #     pred_points * scale.view(B, 1, 1, 1, 1),
        #     gt_points
        # ) * weights.view(B, L, H, W, 1)

        s_expanded = s.view(B, L, H, W, 1)
        loss = raw_error * torch.exp(-s_expanded) + s_expanded

        loss = loss.masked_fill(~gt_valid_depth_mask.unsqueeze(-1), 0.0)
        return loss.sum(dtype=torch.float32) / (gt_valid_depth_mask.sum() * 3 + 1e-6)

    def normal_loss(self, points, gt_points, mask, gt_depths, s):
        not_edge = ~depth_edge(gt_depths, rtol=0.03)
        mask = mask & not_edge

        B, L, H, W = s.shape
        s_2d = s.view(B * L, 1, H, W)

        # Max pooling works perfectly on log(sigma). Max s = Max uncertainty.
        s_norm_2d = F.max_pool2d(s_2d, kernel_size=2, stride=1)
        s_normal = s_norm_2d.view(B, L, H - 1, W - 1)

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

        upxleft_angle = angle_diff_vec3(torch.cross(upxleft, leftdown - rightdown, dim=-1),
                                              torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown,
                                                          dim=-1)).clamp(MIN_ANGLE, MAX_ANGLE)
        raw_upxleft = _smooth(upxleft_angle, beta=1e-2)
        # raw_upxleft = self.hughber_loss(upxleft_angle)

        leftxdown_angle = angle_diff_vec3(torch.cross(leftxdown, rightdown - rightup, dim=-1),
                                torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)).clamp(MIN_ANGLE,
                                                                                                              MAX_ANGLE)
        raw_leftxdown = _smooth(leftxdown_angle, beta=1e-2)
        # raw_leftxdown = self.hughber_loss(leftxdown_angle)

        downxright_angle = angle_diff_vec3(torch.cross(downxright, rightup - leftup, dim=-1),
                                torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)).clamp(MIN_ANGLE,
                                                                                                            MAX_ANGLE)
        raw_downxright = _smooth(downxright_angle, beta=1e-2)
        # raw_downxright = self.hughber_loss(downxright_angle)

        rightxup_angle = angle_diff_vec3(torch.cross(rightxup, leftup - leftdown, dim=-1),
                                torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)).clamp(
            MIN_ANGLE, MAX_ANGLE)
        raw_rightxup = _smooth(rightxup_angle, beta=1e-2)
        # raw_rightxup = self.hughber_loss(rightxup_angle)

        loss_upxleft = raw_upxleft * torch.exp(-s_normal) + s_normal
        loss_leftxdown = raw_leftxdown * torch.exp(-s_normal) + s_normal
        loss_downxright = raw_downxright * torch.exp(-s_normal) + s_normal
        loss_rightxup = raw_rightxup * torch.exp(-s_normal) + s_normal

        loss = mask_upxleft.to(torch.float32) * loss_upxleft \
               + mask_leftxdown.to(torch.float32) * loss_leftxdown \
               + mask_downxright.to(torch.float32) * loss_downxright \
               + mask_rightxup.to(torch.float32) * loss_rightxup

        total_valid_triangles = mask_upxleft.sum() + mask_leftxdown.sum() + mask_downxright.sum() + mask_rightxup.sum()
        return loss.sum(dtype=torch.float32) / (total_valid_triangles + 1e-6)

    def gradient_matching_loss(self, pred_depth, gt_depth, valid_mask, s):
        B, L, H, W = pred_depth.shape
        s_2d = s.view(B * L, 1, H, W)

        s_dy = F.max_pool2d(s_2d, kernel_size=(2, 1), stride=1).view(B, L, H - 1, W)
        s_dx = F.max_pool2d(s_2d, kernel_size=(1, 2), stride=1).view(B, L, H, W - 1)

        pred_dy = pred_depth[..., 1:, :] - pred_depth[..., :-1, :]
        gt_dy = gt_depth[..., 1:, :] - gt_depth[..., :-1, :]

        pred_dx = pred_depth[..., :, 1:] - pred_depth[..., :, :-1]
        gt_dx = gt_depth[..., :, 1:] - gt_depth[..., :, :-1]

        mask_dy = valid_mask[..., 1:, :] & valid_mask[..., :-1, :]
        mask_dx = valid_mask[..., :, 1:] & valid_mask[..., :, :-1]

        raw_dy = F.smooth_l1_loss(pred_dy, gt_dy, reduction='none', beta=1e-2)
        raw_dx = F.smooth_l1_loss(pred_dx, gt_dx, reduction='none', beta=1e-2)
        # raw_dy = self.hughber_loss(pred_dy, gt_dy)
        # raw_dx = self.hughber_loss(pred_dx, gt_dx)

        loss_dy = raw_dy * torch.exp(-s_dy) + s_dy
        loss_dx = raw_dx * torch.exp(-s_dx) + s_dx

        loss_dy = loss_dy.masked_fill(~mask_dy, 0.0)
        loss_dx = loss_dx.masked_fill(~mask_dx, 0.0)

        return (loss_dy.sum(dtype=torch.float32) / (mask_dy.sum() + 1e-6)) + (loss_dx.sum(dtype=torch.float32) / (mask_dx.sum() + 1e-6))

    def depth_loss(self, pred_depth, gt_depth, valid_mask, s):
        """
        Depth is the only loss which uses hughber_loss because I keep having random pixels inside tables predicted as
        being very far away.
        """
        # raw_error = F.smooth_l1_loss(pred_depth, gt_depth, reduction='none', beta=1e-2)
        raw_error = self.hughber_loss(pred_depth, gt_depth)

        loss = raw_error * torch.exp(-s) + s
        loss = loss.masked_fill(~valid_mask, 0.0)

        return loss.sum(dtype=torch.float32) / (valid_mask.sum() + 1e-6)

    def translation_loss(self, pred, gt_relative_translations, scale, median_depths):
        B = scale.shape[0]
        gt_valid_translation_mask = gt_relative_translations.isfinite().all(dim=-1, keepdim=True)
        gt_invalid_translation_mask = ~gt_valid_translation_mask
        gt_relative_translations = gt_relative_translations.masked_fill(gt_invalid_translation_mask, 0)

        total_translation_loss = F.smooth_l1_loss(
            scale.view(B, 1, 1) * pred['relative_camera_translations'],
            self.median_depth * gt_relative_translations / median_depths.view(B, 1, 1),
            reduction='none',
            beta=1e-2
        )
        # total_translation_loss = self.hughber_loss(
        #     scale.view(B, 1, 1) * pred['relative_camera_translations'],
        #     self.median_depth * gt_relative_translations / median_depths.view(B, 1, 1)
        # )

        total_translation_loss = total_translation_loss.masked_fill(gt_invalid_translation_mask, 0)
        return total_translation_loss.sum(dtype=torch.float32) / ((gt_valid_translation_mask).sum() * 3 + 1e-6)

    def rotation_loss(self, pred, gt_relative_rotations):
        B, Lm1 = gt_relative_rotations.shape[:2]
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
            gt_rotation_valid_mask.view(B, Lm1, 1, 1),
            gt_relative_rotations,
            identity_matrix
        )

        # Calculate trace of (R_pred @ R_gt^T) using element-wise sum
        trace = (pred['relative_camera_rotations'] * gt_relative_rotations).sum(dim=(-2, -1))

        # trace = 1 + 2*cos(theta). Isolate cos(theta) and safely clamp to prevent NaN in arccos.
        cos_theta = torch.clamp((trace - 1.0) / 2.0, min=-1.0 + 1e-6, max=1.0 - 1e-6)

        # L1 absolute error of the rotation angle from 0 is just the angle itself
        angle_error = _smooth(torch.acos(cos_theta), beta=1e-2)
        # angle_error = self.hughber_loss(torch.acos(cos_theta))

        return angle_error.masked_fill(gt_rotation_invalid_mask, 0).sum(dtype=torch.float32) / (gt_rotation_valid_mask.sum() + 1e-6)

    def forward(self, pred, gt):
        invalid_dict = {k: 0 if v is None else (~v.isfinite()).sum() for k, v in pred.items()}
        torch._assert(
            sum(invalid_dict.values()) == 0,
            "Pred has invalid values:" + "".join([f"\n{k}: {v}" for k, v in invalid_dict.items()])
        )

        pred_points, gt_points, scale, weights, median_depths, gt_valid_depth_mask = self.initialize(pred, gt)

        # Log-parameterized uncertainty (s) replaces the divided sigma
        s = pred['raw_uncertainty']

        point_loss = self.point_loss(pred_points, gt_points, scale, weights, gt_valid_depth_mask, s)
        torch._assert(point_loss.isfinite(), f"Point loss invalid ({point_loss})")

        gt_log_depths = torch.log(gt['depths'])
        depth_loss = self.depth_loss(pred['log_depths'], gt_log_depths, gt_valid_depth_mask, s)
        torch._assert(depth_loss.isfinite(), f"Depth loss invalid ({depth_loss})")

        normal_loss = self.normal_loss(
            points=pred_points,
            gt_points=gt_points,
            mask=gt_valid_depth_mask,
            gt_depths=gt['depths'],
            s=s
        )
        torch._assert(normal_loss.isfinite(), f"Normal loss invalid ({normal_loss})")

        gradient_matching_loss = self.gradient_matching_loss(pred['log_depths'], gt_log_depths, gt_valid_depth_mask, s)
        torch._assert(gradient_matching_loss.isfinite(), f"Gradient matching loss invalid ({gradient_matching_loss})")

        gt_relative_rotations, gt_relative_translations = self.obtain_gt_relative_poses(gt['extrinsics'])

        translation_loss = self.translation_loss(pred, gt_relative_translations, scale, median_depths)
        torch._assert(translation_loss.isfinite(), f"Translation loss invalid ({translation_loss})")

        rotation_loss = self.rotation_loss(pred, gt_relative_rotations)
        torch._assert(rotation_loss.isfinite(), f"Rotation loss invalid ({rotation_loss})")

        total_loss = point_loss + depth_loss + gradient_matching_loss + normal_loss + translation_loss + rotation_loss
        return total_loss, dict(
            total_loss=total_loss,
            point_loss=point_loss,
            depth_loss=depth_loss,
            normal_loss=normal_loss,
            gradient_matching_loss=gradient_matching_loss,
            translation_loss=translation_loss,
            rotation_loss=rotation_loss
        )