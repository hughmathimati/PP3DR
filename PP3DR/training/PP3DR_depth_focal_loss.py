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


# Old depth-focal-specific NaN checking:
# torch._assert(
#     (~pred['log_depths'].isfinite()).sum() + (~pred['fx'].isfinite()).sum()
#     + (~pred['fy'].isfinite()).sum() + (~pred['cx'].isfinite()).sum() + (~pred['cy'].isfinite()).sum()
#     + (~pred['relative_camera_translations'].isfinite()).sum()
#     + (~pred['relative_camera_rotations'].isfinite()).sum() == 0,
#     f"Pred has invalid values. log_depths: {(~pred['log_depths'].isfinite()).sum()}, "
#     f"fx: {(~pred['fx'].isfinite()).sum()}, "
#     f"fy: {(~pred['fy'].isfinite()).sum()}, "
#     f"cx: {(~pred['cx'].isfinite()).sum()}, "
#     f"cy: {(~pred['cy'].isfinite()).sum()}, "
#     f"relative_camera_translations: {(~pred['relative_camera_translations'].isfinite()).sum()}, "
#     f"relative_camera_rotations: {(~pred['relative_camera_rotations'].isfinite()).sum()}"
# )