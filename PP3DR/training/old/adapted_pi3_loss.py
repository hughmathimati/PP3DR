import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim=None, keepdim: bool = False,
                  eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).sum(dim=dim, keepdim=keepdim) / w.sum(dim=dim, keepdim=keepdim).add(eps)


def _smooth(err: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8):
    """Safely computes the angular difference, immune to the norm(0) singularity."""
    v1, v2 = v1.to(torch.float32), v2.to(torch.float32)
    cross = torch.linalg.cross(v1, v2, dim=-1)
    cross_norm = torch.sqrt(torch.sum(cross.square(), dim=-1) + eps)
    dot = (v1 * v2).sum(dim=-1)
    return torch.atan2(cross_norm, dot).to(v1.dtype)


def depth_edge(depth: torch.Tensor, rtol: float = 0.03) -> torch.Tensor:
    dy = torch.abs(depth[..., 1:, :] - depth[..., :-1, :]) / depth[..., :-1, :].clamp_min(1e-6)
    dx = torch.abs(depth[..., :, 1:] - depth[..., :, :-1]) / depth[..., :, :-1].clamp_min(1e-6)
    edge = torch.zeros_like(depth, dtype=torch.bool)
    edge[..., 1:, :] |= (dy > rtol)
    edge[..., :-1, :] |= (dy > rtol)
    edge[..., :, 1:] |= (dx > rtol)
    edge[..., :, :-1] |= (dx > rtol)
    return edge


@torch.compile()
class Adapted_Pi3_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_point = 10.0
        self.w_normal = 1.0
        self.w_trans = 20.0
        self.w_rot = 1.0
        self.w_anchor = 0.01

    def calculate_scale(self, pred_pts, gt_pts, weights):
        pred_z = pred_pts[..., 2].flatten(1)
        gt_z = gt_pts[..., 2].flatten(1)
        weights = weights.flatten(1)

        valid_mask = pred_z.abs() > 1e-8
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

        ratios = gt_z / torch.clamp(pred_z, min=1e-8)
        effective_weights = weights * pred_z.abs()

        effective_weights = torch.nan_to_num(effective_weights, nan=0.0, posinf=0.0, neginf=0.0)
        ratios = torch.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)

        sorted_ratios, sort_indices = torch.sort(ratios)
        sorted_weights = torch.gather(effective_weights, dim=-1, index=sort_indices)

        sorted_weights = sorted_weights.to(torch.float32)
        cum_weights = torch.cumsum(sorted_weights, dim=-1)

        half_total_weight = cum_weights[:, -1:] / 2.0
        median_idx = torch.searchsorted(cum_weights, half_total_weight)

        max_valid_idx = sorted_ratios.shape[1] - 1
        median_idx = median_idx.clamp(0, max_valid_idx)

        return torch.gather(sorted_ratios, dim=1, index=median_idx)

    def obtain_gt_relative_poses(self, extrinsics):
        current_rotation = extrinsics[:, :-1, :, :3]
        next_rotation = extrinsics[:, 1:, :, :3]
        current_translation = extrinsics[:, :-1, :, 3]
        next_translation = extrinsics[:, 1:, :, 3]

        relative_rotations = next_rotation @ current_rotation.transpose(-1, -2)
        relative_translations = next_translation - current_translation
        return relative_rotations, relative_translations

    def obtain_pred_3D_points(self, pred):
        safe_log_depths = torch.clamp(pred['log_depths'], min=-20.0, max=80.0)
        depths = torch.exp(safe_log_depths).unsqueeze(-1)
        xy = pred['XY_rays'] * depths
        return torch.cat((xy, depths), dim=-1)

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

    def normal_loss(self, points, gt_points, mask, gt_depths):
        not_edge = ~depth_edge(gt_depths, rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

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

        loss = mask_upxleft * _smooth(angle_diff_vec3(torch.cross(upxleft, leftdown - rightdown, dim=-1),
                                                      torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown,
                                                                  dim=-1)).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
               + mask_leftxdown * _smooth(angle_diff_vec3(torch.cross(leftxdown, rightdown - rightup, dim=-1),
                                                          torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup,
                                                                      dim=-1)).clamp(MIN_ANGLE, MAX_ANGLE),
                                          beta=BETA_RAD) \
               + mask_downxright * _smooth(angle_diff_vec3(torch.cross(downxright, rightup - leftup, dim=-1),
                                                           torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup,
                                                                       dim=-1)).clamp(MIN_ANGLE, MAX_ANGLE),
                                           beta=BETA_RAD) \
               + mask_rightxup * _smooth(angle_diff_vec3(torch.cross(rightxup, leftup - leftdown, dim=-1),
                                                         torch.cross(gt_rightdown - gt_leftdown,
                                                                     gt_leftup - gt_leftdown, dim=-1)).clamp(MIN_ANGLE,
                                                                                                             MAX_ANGLE),
                                         beta=BETA_RAD)

        return loss.sum() / (mask.sum() * 4 * max(points.shape[-3:-1]) + 1e-6)

    def forward(self, pred, gt):
        B, L, H, W = pred['log_depths'].shape

        # ==========================================================
        # 1. PHYSICAL SANITIZATION: DEPTHS
        # ==========================================================
        gt_valid_depth_mask = torch.isfinite(gt['depths']) & (gt['depths'] != 0)

        depths_for_median = gt['depths'].masked_fill(~gt_valid_depth_mask, float('nan'))
        median_depths = torch.nanmedian(depths_for_median.flatten(1), dim=-1).values
        median_depths = torch.nan_to_num(median_depths, nan=1.0)

        # Overwrite all NaNs/Infs in the GT array with the median depth.
        # This permanently protects the backward pass of Huber Loss!
        gt['depths'] = torch.where(gt_valid_depth_mask, gt['depths'], median_depths.view(B, 1, 1, 1))
        gt['depths'] = gt['depths'] / median_depths.view(B, 1, 1, 1)

        weights = gt['depths'].clone()
        min_weights = 0.1 * weighted_mean(weights, gt_valid_depth_mask, dim=(-2, -1), keepdim=True)
        weights = weights.clamp_min(min_weights)
        weights = 1.0 / (weights + 1e-6)
        weights = torch.where(gt_valid_depth_mask, weights, torch.zeros_like(weights))

        pred_points, gt_points = self.obtain_pred_3D_points(pred), self.obtain_gt_3D_points(gt)
        scale = self.calculate_scale(pred_points.detach(), gt_points, weights)

        # ------------------- Point & Normal Loss -------------------
        total_point_loss = F.huber_loss(
            pred_points,
            gt_points / scale.view(B, 1, 1, 1, 1),
            reduction='none'
        ) * weights.view(B, L, H, W, 1)

        total_point_loss = torch.where(gt_valid_depth_mask.unsqueeze(-1), total_point_loss,
                                       torch.zeros_like(total_point_loss))
        point_loss = total_point_loss.sum() / (gt_valid_depth_mask.sum() * 3 + 1e-6)

        normal_loss = self.normal_loss(
            pred_points,
            gt_points / scale.view(B, 1, 1, 1, 1),
            gt_valid_depth_mask,
            gt['depths']
        )

        # ==========================================================
        # 2. PHYSICAL SANITIZATION: POSES
        # ==========================================================
        gt_relative_rotations, gt_relative_translations = self.obtain_gt_relative_poses(gt['extrinsics'])

        gt_valid_translation_mask = torch.isfinite(gt_relative_translations).all(dim=-1)
        gt_rot_valid = gt_relative_rotations.isfinite().all(dim=-1).all(dim=-1) & (
                    gt_relative_rotations.flatten(start_dim=2).abs().max(dim=-1).values <= 1.05)

        # Overwrite all NaN translations with 0.0
        safe_gt_translations = torch.where(
            gt_valid_translation_mask.unsqueeze(-1),
            gt_relative_translations,
            torch.zeros_like(gt_relative_translations)
        )

        # Overwrite all NaN rotations with a valid Identity matrix [1, 0, 0; 0, 1, 0...]
        identity_matrix = torch.eye(3, device=gt_relative_rotations.device, dtype=gt_relative_rotations.dtype).view(1,
                                                                                                                    1,
                                                                                                                    3,
                                                                                                                    3)
        safe_gt_rotations = torch.where(
            gt_rot_valid.view(B, L - 1, 1, 1),
            gt_relative_rotations,
            identity_matrix
        )

        # ------------------- Translation Loss -------------------
        total_translation_loss = F.huber_loss(
            pred['relative_camera_translations'],
            safe_gt_translations / (median_depths.view(B, 1, 1) * scale.view(B, 1, 1)),
            reduction='none'
        )
        total_translation_loss = torch.where(gt_valid_translation_mask.unsqueeze(-1), total_translation_loss,
                                             torch.zeros_like(total_translation_loss))
        translation_loss = total_translation_loss.sum() / (gt_valid_translation_mask.sum() + 1e-6)

        # ------------------- Rotation Loss -------------------
        trace = (pred['relative_camera_rotations'] * safe_gt_rotations).sum(dim=(-2, -1))
        cosine = ((trace - 1.0) / 2.0).to(torch.float32)
        rot_ang_err = torch.acos(torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6))
        rot_ang_err = rot_ang_err.to(trace.dtype)

        rot_ang_err = torch.where(gt_rot_valid, rot_ang_err, torch.zeros_like(rot_ang_err))
        rotation_loss = rot_ang_err.sum() / (gt_rot_valid.sum() + 1e-6)

        # ------------------- Anchor Loss -------------------
        anchor_loss = self.w_anchor * pred['log_depths'].mean().square()

        total_loss = (self.w_point * point_loss) + \
                     (self.w_normal * normal_loss) + \
                     (self.w_trans * translation_loss) + \
                     (self.w_rot * rotation_loss) + \
                     anchor_loss

        return total_loss, dict(
            total_loss=total_loss,
            point_loss=point_loss,
            normal_loss=normal_loss,
            translation_loss=translation_loss,
            rotation_loss=rotation_loss
        )