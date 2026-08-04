import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from models.PP3DR import GlobalBlock, LocalBlock, PP3DR

# --------------------------------------------------------
# 1. DPT Sub-Modules (Highly Optimized FPN)
# --------------------------------------------------------

class PreActResidualConvUnit(nn.Module):
    """ResNet-style block used to smooth and refine features during fusion."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + res


class FeatureFusionBlock(nn.Module):
    """Zips together a deeper FPN layer with a shallower FPN layer."""

    def __init__(self, channels):
        super().__init__()
        self.res_conv_unit1 = PreActResidualConvUnit(channels)
        self.res_conv_unit2 = PreActResidualConvUnit(channels)
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x, res=None):
        if res is not None:
            # Upsample the deeper layer to match the current layer's spatial size
            if x.shape[-2:] != res.shape[-2:]:
                res = F.interpolate(res, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.res_conv_unit1(res)

        x = self.res_conv_unit2(x)
        # Upsample the fused result by 2x for the next FPN stage
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        return self.project(x)


class DPT(nn.Module):
    """
    The core Dense Prediction Transformer logic.
    Reassembles the 1D sequences into 2D feature maps, injects global context,
    builds the FPN hierarchy, and fuses them into a high-res depth map.
    """

    def __init__(self, dim=1280, num_registers=5, fpn_channels=[128, 256, 512, 1024], fusion_dim=256):
        super().__init__()
        self.num_registers = num_registers
        self.dim = dim

        # 1. Readout Projects: Merges the global register tokens into every spatial pixel
        self.readout_projects = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU()) for _ in range(4)
        ])

        # 2. Channel Projects: Forces the 4 layers into FPN channel sizes (shallower = smaller)
        self.channel_projs = nn.ModuleList([
            nn.Conv2d(dim, c, kernel_size=1) for c in fpn_channels
        ])

        # 3. Rescalers: Physically builds the FPN spatial pyramid
        self.resizers = nn.ModuleList([
            nn.ConvTranspose2d(fpn_channels[0], fpn_channels[0], kernel_size=4, stride=4, padding=0),  # 4x Scale
            nn.ConvTranspose2d(fpn_channels[1], fpn_channels[1], kernel_size=2, stride=2, padding=0),  # 2x Scale
            nn.Identity(),  # 1x Scale
            nn.Conv2d(fpn_channels[3], fpn_channels[3], kernel_size=3, stride=2, padding=1)  # 0.5x Scale
        ])

        # 4. Fusion Blocks: The "Zipper"
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(fusion_dim) for _ in range(4)
        ])
        # The deepest layer doesn't have a previous layer to merge with
        self.fusion_blocks[0].res_conv_unit1 = None

        # 5. Pre-Fusion Refinement (Anti-aliasing)
        self.refinements = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fusion_dim, kernel_size=3, padding=1, bias=False),
                nn.GELU()
            ) for c in fpn_channels
        ])

        # 6. Final UpConv Head: Projects to 1 channel and scales to original image size
        self.upconv = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fusion_dim // 2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1)  # 1 channel for depth
        )

    def forward(self, features, H, W, patch_size=16):
        """
        features: List of 4 Tensors of shape (B*L, num_registers + H_patch*W_patch, dim)
        """
        # Calculate patch grid dimensions
        patch_h, patch_w = H // patch_size, W // patch_size

        fpn_features = []

        # --- Step 1 & 2: Reassembly & Rescaling ---
        for i, feat in enumerate(features):
            # Split registers (global context) from spatial tokens
            registers = feat[:, :self.num_registers, :]
            spatial = feat[:, self.num_registers:, :]

            # Average registers to create a single global scene token
            global_token = registers.mean(dim=1, keepdim=True)  # (B*L, 1, dim)

            # Inject global awareness into every local pixel
            spatial = torch.cat([spatial, global_token.expand_as(spatial)], dim=-1)
            spatial = self.readout_projects[i](spatial)

            # Reshape 1D sequence back into 2D image map
            spatial_2d = rearrange(spatial, "B (h w) C -> B C h w", h=patch_h, w=patch_w)

            # Project channels and artificially scale to create FPN hierarchy
            spatial_2d = self.channel_projs[i](spatial_2d)
            spatial_2d = self.resizers[i](spatial_2d)
            fpn_features.append(spatial_2d)

        # --- Step 3: Pre-Fusion Smoothing ---
        fpn_features = [self.refinements[i](f) for i, f in enumerate(fpn_features)]

        # --- Step 4: FPN Fusion (Deepest to Shallowest) ---
        out = self.fusion_blocks[0](fpn_features[3])  # Start with deepest (Layer 3)
        out = self.fusion_blocks[1](fpn_features[2], out)
        out = self.fusion_blocks[2](fpn_features[1], out)
        out = self.fusion_blocks[3](fpn_features[0], out)

        # --- Step 5: Final UpConv & Interpolation ---
        depth = self.upconv(out)
        # Force exact original image dimensions (resolves any rounding issues from strides)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)

        # Return flattened depth: (B*L, H*W)
        return depth.squeeze(1)


# --------------------------------------------------------
# 2. Main Architecture Blocks
# --------------------------------------------------------

class DPTHead(nn.Module):
    def __init__(
            self,
            input_dim=2 * 1280,
            hidden_dim=1280,
            num_registers=5,
            num_heads=20,
            blocks=4,  # Pi3 has 5 transformer blocks per decoder
            ffn_ratio=4,
    ):
        super().__init__()
        assert blocks % 2 == 0, f"Number of decoder blocks ({blocks}) must be even"
        self.blocks_each = blocks // 2
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim

        self.global_blocks = nn.ModuleList(
            [GlobalBlock(hidden_dim, num_heads, ffn_ratio) for _ in range(self.blocks_each)])
        self.local_blocks = nn.ModuleList(
            [LocalBlock(hidden_dim, num_heads, ffn_ratio) for _ in range(self.blocks_each)])

        self.registers_layer_norm = nn.LayerNorm(hidden_dim)
        self.focal_proj = nn.Linear(num_registers * hidden_dim, 4)  # fx, fy, cx, cy
        self.num_registers = num_registers
        self.num_blocks = blocks

        self.final_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dpt = DPT(dim=hidden_dim, num_registers=num_registers)

        # Handle checkpointing property if not inherited
        self.start_checkpointing = 0

    def forward(self, x: torch.Tensor, dpt_features, rope2d, rope3d, L, H, W) -> tuple:
        B_L = x.shape[0]
        B = B_L // L

        x = self.input_proj(x)
        for i in range(self.blocks_each - 1):
            if self.training and i >= self.start_checkpointing:
                x = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant=False)
                x = checkpoint(self.local_blocks[i], x, rope2d, L, use_reentrant=False)
            else:
                x = self.global_blocks[i](x, rope3d, L)
                x = self.local_blocks[i](x, rope2d, L)

        last_index = self.blocks_each - 1
        if self.training and self.start_checkpointing <= last_index:
            a = checkpoint(self.global_blocks[last_index], x, rope3d, L, use_reentrant=False)
            x = torch.cat([
                a.view(B * L, -1, self.hidden_dim),
                checkpoint(self.local_blocks[last_index], a, rope2d, L, use_reentrant=False)
            ], dim=-1)
        else:
            a = self.global_blocks[last_index](x, rope3d, L)
            x = torch.cat([
                a.view(B * L, -1, self.hidden_dim),
                self.local_blocks[last_index](a, rope2d, L)
            ], dim=-1)

        x = self.final_proj(x)
        dpt_features.append(x)  # 4th and final feature map

        # --- Depth Pathway ---
        # Returns flattened depth map of shape (B*L, H*W)
        depths = self.dpt(dpt_features, H, W)
        depths = depths.view(B, L, H, W)

        # --- Intrinsics Pathway ---
        x = rearrange(x, "(B L) X dim -> B L X dim", L=L)
        registers = self.registers_layer_norm(x[:, :, :self.num_registers, :])
        fx, fy, cx, cy = self.focal_proj(registers.flatten(start_dim=2)).unbind(-1)

        fx, fy = F.softplus(fx), F.softplus(fy)
        cx, cy = cx + W / 2, cy + H / 2

        return depths, fx, fy, cx, cy


@torch.compile()
class PP3DR_DPT(PP3DR):
    def __init__(self, *args, **kwargs):
        # NOTE: Assuming you have a dummy point_head_class arg in super or you patch it
        super().__init__(*args, **kwargs, point_head_class=DPTHead)
        # Using self.dim assuming it's initialized in super()
        self.half_proj = nn.Linear(2 * self.dim, self.dim)
        self.end_proj = nn.Linear(2 * self.dim, self.dim)

    def forward(self, x: torch.Tensor, rope_x, rope_y) -> dict:
        B, L, C, H, W = x.shape

        # Step 1: Feature Extraction
        x = self.ViT(x.flatten(0, 1))
        dpt_features = [x]  # 1st Feature

        rope2d, rope3d = self.rope2d(rope_x, rope_y), self.rope3d(rope_x, rope_y)

        # First half of the 3D blocks
        half = self.blocks_each // 2
        for i in range(half - 1):
            if self.training and i >= self.start_checkpointing:
                x = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant=False)
                x = checkpoint(self.local_blocks[i], x, rope2d, L, use_reentrant=False)
            else:
                x = self.global_blocks[i](x, rope3d, L)
                x = self.local_blocks[i](x, rope2d, L)

        # Midpoint Extraction
        if self.training and half - 1 >= self.start_checkpointing:
            a = checkpoint(self.global_blocks[half - 1], x, rope3d, L, use_reentrant=False)
            x = checkpoint(self.local_blocks[half - 1], a, rope2d, L, use_reentrant=False)
        else:
            a = self.global_blocks[half - 1](x, rope3d, L)
            x = self.local_blocks[half - 1](a, rope2d, L)

        # 2nd Feature (Projected back down to self.dim to match the others)
        dpt_features.append(self.half_proj(torch.cat([a.view(B * L, -1, self.dim), x], dim=-1)))

        # Second half of the 3D blocks
        last_index = self.blocks_each - 1
        for i in range(half, last_index):
            if self.training and i >= self.start_checkpointing:
                x = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant=False)
                x = checkpoint(self.local_blocks[i], x, rope2d, L, use_reentrant=False)
            else:
                x = self.global_blocks[i](x, rope3d, L)
                x = self.local_blocks[i](x, rope2d, L)

        # End of 3D Blocks Extraction
        if self.training and last_index >= self.start_checkpointing:
            a = checkpoint(self.global_blocks[last_index], x, rope3d, L, use_reentrant=False)
            x = torch.cat([
                a.view(B * L, -1, self.dim),
                checkpoint(self.local_blocks[last_index], a, rope2d, L, use_reentrant=False)
            ], dim=-1)
        else:
            a = self.global_blocks[last_index](x, rope3d, L)
            x = torch.cat([
                a.view(B * L, -1, self.dim),
                self.local_blocks[last_index](a, rope2d, L)
            ], dim=-1)

        # 3rd Feature (Projected back down to self.dim)
        dpt_features.append(self.end_proj(x))

        # Step 2: Per-task decoders
        log_depths, fx, fy, cx, cy = self.point_head(x, dpt_features, rope2d, rope3d, L, H, W)

        poses = self.pose_head(x, rope2d, rope3d, L)[:, :-1]

        a = F.normalize(poses[:, :, 3:6], dim=-1, eps=1e-6)
        b = poses[:, :, 6:]
        b = F.normalize(b - torch.linalg.vecdot(a, b, dim=-1).unsqueeze(-1) * a, dim=-1, eps=1e-6)
        c = torch.linalg.cross(a, b, dim=-1)

        return {
            "log_depths": torch.clamp(log_depths, min=-80, max=80),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "relative_camera_translations": poses[:, :, :3],
            "relative_camera_rotations": torch.stack([a, b, c], dim=-1)
        }

"""My old code below:"""
# import os
# import torch.nn.functional as F
# import torch
# import torch.nn as nn
# from torch.linalg import vector_norm
# import torch.cuda.amp as amp
# from einops import rearrange
# from typing import Callable
# from torch.utils.checkpoint import checkpoint
# import torchvision.transforms.v2 as transforms
#
# from models.BidirectionalLaCT import GlobalLaCT, LocalLaCT
# from models.pos_embed import RopePositionEmbedding, Rope3D, Rope2D
# from models.ViTTT import ViTTT
# from models.PP3DR import GlobalBlock, LocalBlock, PP3DR
# from xformers.ops import SwiGLU
# from timm.layers import DropPath
#
# class DPT(nn.Module):
#     def forward(self, dpt_features):
#
#
# class DPTHead(nn.Module):
#     def __init__(
#             self,
#             input_dim=2 * 1280,
#             hidden_dim=1280,
#             num_registers=5,
#             num_heads=20,
#             blocks=4,  # Pi3 has 5 transformer blocks per decoder
#             ffn_ratio=4,
#     ):
#         super().__init__()
#         assert blocks % 2 == 0, f"Number of decoder blocks ({blocks}) must be even for alternating global and frame-wise attention"
#         self.blocks_each = blocks // 2
#         self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
#         self.hidden_dim = hidden_dim
#
#         self.global_blocks = nn.ModuleList(
#             [GlobalBlock(hidden_dim, num_heads, ffn_ratio) for _ in range(self.blocks_each)])
#         self.local_blocks = nn.ModuleList(
#             [LocalBlock(hidden_dim, num_heads, ffn_ratio) for _ in range(self.blocks_each)])
#
#         self.registers_layer_norm = nn.LayerNorm(hidden_dim)  # Pre-projection layer norm.
#         self.focal_proj = nn.Linear(num_registers * hidden_dim, 4)  # fx, fy, cx, cy
#         self.num_registers = num_registers
#         self.num_blocks = blocks
#
#         self.final_proj = nn.Linear(2 * hidden_dim, hidden_dim)
#         self.dpt = DPT()
#
#     def forward(self, x: torch.Tensor, dpt_features, rope2d, rope3d, L, H, W) -> torch.Tensor:
#         """
#         Input to the model consists of the outputs of the last global block AND the last local block, concatenated
#         along the last dimension. Thus, the input dimension is 2 * 1280.
#
#         Parameters
#         ----------
#         x: Input tensor
#         rope2d: 2D RoPE positional embedding
#         rope3d: 3D RoPE positional embedding
#         L: Number of frames. Necessary for the rearrange operations.
#         """
#         x = self.input_proj(x)
#         for i in range(self.blocks_each - 1):
#             if self.training:
#                 # Global attention: absorb frame-length into patch-length dimension.
#                 x = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant=False)
#                 # Local attention: absorb frame-length into batch dimension. Sequence length is now patch-length.
#                 x = checkpoint(self.local_blocks[i], x, rope2d, L, use_reentrant=False)
#             else:
#                 x = self.global_blocks[i](x, rope3d, L)
#                 x = self.local_blocks[i](x, rope2d, L)
#
#         last_index = self.blocks_each - 1
#         if self.training and self.start_checkpointing == self.blocks_each:
#             x = checkpoint(self.global_blocks[last_index], x, rope3d, L, use_reentrant=False)
#             x = torch.cat([
#                 x.view(B * L, -1, self.dim),
#                 checkpoint(self.local_blocks[last_index], x, rope2d, L, use_reentrant=False)
#             ], dim=-1)
#         else:
#             x = self.global_blocks[last_index](x, rope3d, L)
#             x = torch.cat([
#                 x.view(B * L, -1, self.dim),
#                 self.local_blocks[last_index](x, rope2d, L)
#             ], dim=-1)
#         # x.shape == (B * L, num_registers + HW // 256, 2 * dim)
#         x = self.final_proj(x)
#         dpt_features.append(x)
#
#         x = rearrange(x, "(B L) X dim -> B L X dim", L=L)  # (B, L, num_registers + HW // 256, dim)
#         registers = self.registers_layer_norm(x[:, :, :self.num_registers, :])  # (B, L, num_registers, dim)
#         # (B, L, num_registers * dim) -> (B, L, 4) -> 4 x (B, L)
#         fx, fy, cx, cy = self.focal_proj(registers.flatten(start_dim=2)).unbind(-1)
#         # fx and fy will be passed into softplus to ensure they're always positive.
#         fx, fy = F.softplus(fx), F.softplus(fy)
#         # cx and cy will actually be the residuals to the image center point.
#         cx, cy = cx + W / 2, cy + H / 2
#
#         # depths has shape (B, L, HW). fx, fy, cx, and cy have shape (B, L).
#         return self.dpt(dpt_features), fx, fy, cx, cy
#
#
# @torch.compile()
# class PP3DR_DPT(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs, point_head_class=DPTHead)
#         self.half_proj = nn.Linear(2 * self.dim, self.dim)
#         self.end_proj = nn.Linear(2 * self.dim, self.dim)
#
#     def forward(self, x: torch.Tensor, rope_x, rope_y) -> dict:
#         """
#         The input must be pre-processed so its height and width are multiples of the patch size.
#
#         rope_x and rope_y come from the dataloader.
#
#         Parameters
#         ----------
#         images: (batch size, RGB, height, width)
#         rope_x: The 2D x-coordinates to be fed into RoPE (B, L, HW)
#         rope_y: The 2D y-coordinates to be fed into RoPE; same shape as above.
#
#         Returns
#         -------
#         "XY_ray": Normalized XY ray direction for each pixel's point
#
#         "log_depth": Log depth for each pixel's point
#
#         "relative_camera_translation": (x, y, z) translation between this camera pose and the next one
#
#         "relative_camera_rotation": quaternion rotation between this camera pose and the next one
#         """
#         B, L, C, H, W = x.shape
#
#         # Step 1: ViT
#         # ViT is frame-wise. We need to collapse the batch dimension into the sequence dimension before passing it into Dino.
#         x = self.ViT(x.flatten(0, 1))
#         # x.shape == (B * L, num_registers + HW // 256, dim)
#         dpt_features = [x] # First feature comes from feature extractor
#
#         """
#         We will now perform alternating-attention, starting with global attention and ending with frame-wise attention.
#
#         The RoPE coordinates have been pre-calculated and are provided by the dataloader. This is because they're
#         dependent on the dimensions of the raw images.
#
#         rope2d has shape (B * L, HW, head_dim), and rope3d has shape (B, L, HW, head_dim).
#         """
#         rope2d, rope3d = self.rope2d(rope_x, rope_y), self.rope3d(rope_x, rope_y)
#
#         # First half of the blocks
#         half = self.blocks_each // 2
#         for i in range(half - 1):
#             if self.training and i >= self.start_checkpointing:
#                 # Global attention: absorb frame-length into patch-length dimension.
#                 x = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant=False)
#                 # Local attention: absorb frame-length into batch dimension. Sequence length is now patch-length.
#                 x = checkpoint(self.local_blocks[i], x, rope2d, L, use_reentrant=False)
#             else:
#                 x = self.global_blocks[i](x, rope3d, L)
#                 x = self.local_blocks[i](x, rope2d, L)
#         if self.training and half >= self.start_checkpointing:
#             # Global attention: absorb frame-length into patch-length dimension.
#             a = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant=False)
#             # Local attention: absorb frame-length into batch dimension. Sequence length is now patch-length.
#             x = checkpoint(self.local_blocks[i], a, rope2d, L, use_reentrant=False)
#         else:
#             a = self.global_blocks[i](x, rope3d, L)
#             x = self.local_blocks[i](a, rope2d, L)
#         # Second feature comes from halfway
#         dpt_features.append(self.half_proj(torch.cat(a, x, dim=-1)))
#
#         # Save one block from each of the last two layers
#         last_index = self.blocks_each - 1
#         if self.training and self.start_checkpointing == self.blocks_each:
#             x = checkpoint(self.global_blocks[last_index], x, rope3d, L, use_reentrant=False)
#             x = torch.cat([
#                 x.view(B * L, -1, self.dim),
#                 checkpoint(self.local_blocks[last_index], x, rope2d, L, use_reentrant=False)
#             ], dim=-1)
#         else:
#             x = self.global_blocks[last_index](x, rope3d, L)
#             x = torch.cat([
#                 x.view(B * L, -1, self.dim),
#                 self.local_blocks[last_index](x, rope2d, L)
#             ], dim=-1)
#         # x.shape == (B * L, num_registers + HW // 256, 2 * dim)
#         dpt_features.append(x)
#
#         # Step 2: Per-task decoders
#         log_depths, fx, fy, cx, cy = self.DPTHead(x, dpt_features, rope2d, rope3d, L, H, W)
#         # We're predicting the relative pose from this frame to the next one, which is why our sequence length is L - 1.
#         poses = self.pose_decoder(x, rope2d, rope3d, L)[:, :-1]  # (B, L - 1, 9)
#
#         # Get the rotation matrix by orthogonalizing the first two 3D vectors, then taking the cross product for the third.
#         # We're going to construct the actual columns as rows of the current matrix, then take the transpose at the end.
#         # eps is increased from 1e-12 to 1e-6 in an attempt to increase stability.
#         a = F.normalize(poses[:, :, 3:6], dim=-1, eps=1e-6)
#         b = poses[:, :, 6:]
#         # Unsqueeze after the dot product so the resulting scalars broadcast correctly against the vector a.
#         b = F.normalize(b - torch.linalg.vecdot(a, b, dim=-1).unsqueeze(-1) * a, dim=-1, eps=1e-6)
#         c = torch.linalg.cross(a, b, dim=-1)
#
#         return {
#             # e^-80 to e^80 is safely within the range of bfloat16.
#             "log_depths": torch.clamp(log_depths, min=-80, max=80),
#             "fx": fx,
#             "fy": fy,
#             "cx": cx,
#             "cy": cy,
#             "relative_camera_translations": poses[:, :, :3],  # (B, L - 1, 3) -> 3 scalars, (x, y, z)
#             "relative_camera_rotations": torch.stack([a, b, c], dim=-1)  # (B, L - 1, 3, 3)
#         }