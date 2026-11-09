import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.linalg import vector_norm
import torch.cuda.amp as amp
from einops import rearrange
from typing import Callable
from torch.utils.checkpoint import checkpoint
import torchvision.transforms.v2 as transforms
from xformers.ops import SwiGLU
from timm.layers import DropPath

from models.pos_embed import RopePositionEmbedding, Rope3D, Rope2D
from models.ViTTT import ViTTT, LayerScale, SpatialDepthwiseConv
from models.BidirectionalLaCT import GlobalLaCT, LocalLaCT
from models.utils import apply_pos_embed


class GlobalBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio, drop_path=0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.TTT = GlobalLaCT(dim, num_heads)
        self.layer_scale_1 = LayerScale(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(in_features=dim, hidden_features=dim * ffn_ratio)
        self.layer_scale_2 = LayerScale(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope3d, L) -> torch.Tensor:
        """
        Parameters
        ----------
        x: [(B L) X dim]
        rope3d: 2 * [LHW, D_head]
        L

        Returns
        -------
        [B (L X) dim]
        """
        # First addition needs a rearrange, since x gets rearranged in the TTT.
        x = rearrange(x, "(B L) X dim -> B (L X) dim", L=L) + self.drop_path(
            self.layer_scale_1(self.TTT(self.layer_norm_1(x), rope3d, L))
        )
        x = x + self.drop_path(self.layer_scale_2(self.ffn(self.layer_norm_2(x))))
        return x

class LocalBlock(nn.Module):
    """
    NOTE: LocalLaCT has been updated so that the input is reshaped to ((B L), X, dim) *before* being passed into
    forward().
    This also means we don't need to pass L into forward() anymore.
    """
    def __init__(self, dim, num_heads, ffn_ratio, drop_path=0):
        super().__init__()
        self.conv1 = SpatialDepthwiseConv(dim)
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.TTT = LocalLaCT(dim, num_heads)
        self.layer_scale_1 = LayerScale(dim)
        self.conv2 = SpatialDepthwiseConv(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(in_features=dim, hidden_features=dim * ffn_ratio)
        self.layer_scale_2 = LayerScale(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope, L, H_p: int, W_p: int) -> torch.Tensor:
        x = rearrange(x, "B (L X) dim -> (B L) X dim", L=L)
        x = x + self.drop_path(self.layer_scale_1(self.TTT(
            self.conv1(self.layer_norm_1(x), H_p, W_p),
            rope
        )))
        x = x + self.drop_path(self.layer_scale_2(self.ffn(
            self.conv2(self.layer_norm_2(x), H_p, W_p)
        )))
        return x

class PoseHead(nn.Module):
    """
    Predict the relative camera pose per frame via the register (special) tokens.
    """

    def __init__(
            self,
            dim,
            num_registers=5,
            output_dim=9,  # (x, y, z) + 6D continuous rotation representation. This is per-frame.
    ):
        super().__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(num_registers * dim)  # Pre-projection layer norm.
        self.dim_proj = nn.Linear(2 * num_registers * dim, output_dim)
        self.num_registers = num_registers
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, L) -> torch.Tensor:
        """
        Parameters
        ----------
        x: Input tensor
        rope2d: 2D RoPE positional embedding
        rope3d: 3D RoPE positional embedding
        L: Number of frames. Necessary for the rearrange operations.
        """
        # x.shape == (B * L, num_registers + HW // 256, dim)
        # Keep only the register tokens (recall that they're concatenated along the sequence dimension, not the embedding dimension).
        x = x[:, :self.num_registers, :]
        # x.shape == (B * L, num_registers, dim)
        # Flatten the register tokens (and separate the batch size and frame dimensions):
        x = rearrange(x, "(B L) reg dim -> B L (reg dim)", L=L)
        x = self.layer_norm(x)

        """
        Concatenate each frame's register tokens with those of the next frame. This also means we don't need RoPE3D anymore.
        """
        x = torch.cat([x[:, :-1], x[:, 1:]], dim=-1) # (B, L - 1, 2 * (reg dim))

        # Project so we end up with the correct number of dimensions at the end
        return self.dim_proj(x) # (B, L - 1, output_dim)

class FocalHead(nn.Module):
    def __init__(self, dim, num_registers=5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.focal_proj = nn.Linear(num_registers * dim, 1)

    def forward(self, x, L, H, W):
        """
        Parameters
        ----------
        (B * L, num_registers, dim)

        Returns
        -------
        (B,)
        """
        registers = self.layer_norm(x)
        f_mult = self.focal_proj(registers.flatten(start_dim=-2)) # (B * L, 1)
        f_mult = rearrange(f_mult.squeeze(-1), "(B L) -> B L", L=L).mean(dim=-1) # (B,)
        f_mult = F.softplus(f_mult)
        # H and W are the padded + resized dimensions. The
        return f_mult * max(H, W)

class ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x

class DepthHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.initial_proj = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(dim, 256, 4, 4),
                # NOTE: I'm somewhat doubtful of the usefulness of these post conv-transpose convolutions.
                nn.Conv2d(256, 256, 3, 1, 1)
            ), # 4x
            nn.Sequential(
                nn.ConvTranspose2d(dim, 256, kernel_size=2, stride=2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ), # 2x
            # A single linear convolution works here because dim is so much larger than 256.
            nn.Conv2d(dim, 256, kernel_size=3, stride=1, padding=1), # 1x
            # No smoothing needed here because it immediately feeds into a resconv
            nn.Conv2d(dim, 256, kernel_size=2, stride=2, padding=0)
        ])
        self.fusion_resconv = nn.ModuleList([ResidualConvUnit(256) for _ in range(6)])
        self.final_upscale = nn.Sequential(
            nn.Conv2d(256, 32, 1, 1), # 32 = 2 * 4^2
            nn.PixelShuffle(upscale_factor=4)
        )
        mid_channels = 64
        self.rgb_proj = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d(3 * 16, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, 256, kernel_size=3, stride=1, padding=1),
        )
        self.rgb_layerscale = LayerScale([1, 256, 1, 1])

    def upscale(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, original_rgb, L, original_height, original_width):
        """
        Parameters
        ----------
        4 x (B * L, dim, H // patch_size, W // patch_size)

        Returns
        -------
        log_depths (B, L, H, W)
        raw_uncertainty (B, L, H, W)
        """
        x = [self.initial_proj[i](y) for i, y in enumerate(x)]
        a, b, c, d = [apply_pos_embed(y, original_width, original_height, L) for y in x]

        x = self.fusion_resconv[1](c + self.upscale(self.fusion_resconv[0](d)))
        x = self.fusion_resconv[3](b + self.upscale(self.fusion_resconv[2](x)))
        x = self.fusion_resconv[5](
            a + self.rgb_layerscale(self.rgb_proj(original_rgb)) + self.upscale(self.fusion_resconv[4](x))
        )

        x = apply_pos_embed(x, original_width, original_height, L)
        # (B * L, 2, H, W)
        return rearrange(self.final_upscale(x), "(B L) two H W -> B L two H W", L=L).unbind(2)

class PointHead(nn.Module):
    def __init__(
            self,
            dim,
            num_registers=5,
    ):
        """
        dim is the (last) dimension of `outputs` in forward(). These are the intermediate blocks fed to the DPT head.
        focal_dim is the (last) dimension of `x` in forward(). This is the tensor fed to the focal head.
        """
        super().__init__()
        self.dim = dim
        self.num_registers = num_registers
        self.focal_head = FocalHead(dim)
        # These LayerNorms are for the depth head only. The FocalHead has its own layer norm.
        self.token_layer_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.depth_head = DepthHead(dim)

    def forward(
            self, outputs: torch.Tensor, original_rgb, L, H, W, original_height, original_width
    ) -> torch.Tensor:
        """
        `outputs` has four tensors of shape (B * L, X, dim).
        `original_rgc` has shape (B * L, 3, H, W)
        """
        # 4 x (B * L, HW // 256, dim)
        tokens = [self.token_layer_norms[i](y[:, self.num_registers:, :]) for i, y in enumerate(outputs)]
        tokens = [rearrange(y, "BL (H_p W_p) dim -> BL dim H_p W_p", H_p=H // 16) for y in tokens]
        focal = self.focal_head(outputs[-1][:, :self.num_registers, :], L, H, W)
        depths, confidence = self.depth_head(
            tokens, original_rgb,
            L, original_height, original_width
        )
        return focal, depths, confidence


@torch.compile()
class PP3DR(nn.Module):
    def __init__(
            self,
            dim: int = 1024,
            num_heads: int = 16,
            # VGGT-Omega uses 24 encoder blocks and 48 decoder blocks.
            encoder_blocks: int = 16,
            decoder_blocks: int = 36,
            ffn_ratio: int = 4,
            num_registers: int = 5,
            # How many blocks EACH not to checkpoint (total # is twice as many).
            start_checkpointing=14,
            point_head_class=PointHead,
            pose_head_class=PoseHead,
    ):
        super().__init__()
        assert decoder_blocks % 2 == 0,\
            f"Number of decoder blocks ({decoder_blocks}) must be even for alternating global and frame-wise attention."
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}."

        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.blocks_each = decoder_blocks // 2
        self.start_checkpointing = start_checkpointing

        # Drop rates
        rates = torch.linspace(0, 0.1, encoder_blocks + decoder_blocks)
        ViTTT_drop_rates = rates[:encoder_blocks]
        PP3DR_drop_rates = rates[encoder_blocks:]

        # General decoder
        self.dim = dim
        drop_rates = [x.item() for x in PP3DR_drop_rates]
        self.global_blocks = nn.ModuleList([
            GlobalBlock(dim, num_heads, ffn_ratio, drop_rates[2 * i])
            for i in range(self.blocks_each)
        ])
        self.local_blocks = nn.ModuleList([
            LocalBlock(dim, num_heads, ffn_ratio, drop_rates[2 * i + 1])
            for i in range(self.blocks_each)
        ])

        # We don't store num_registers here. It gets stored in the decoder heads.
        self.rope2d = Rope2D(embed_dim=dim, num_heads=num_heads, device="cuda")
        # self.rope3d = Rope3D(embed_dim=dim, num_heads=num_heads, device="cuda")

        # Don't checkpoint the ViTTT blocks. Checkpointing the PP3DR blocks saves more time.
        self.ViTTT = ViTTT(
            dim,
            num_heads,
            encoder_blocks,
            ffn_ratio,
            drop_rates=ViTTT_drop_rates,
            start_checkpointing=0,
            final_layer_norm = False,
            # Don't take any ViTTT blocks. Pull all intermediate blocks from PP3DR.
            output_blocks = []
        )
        self.num_registers = num_registers

        # Per-task decoders
        # We're concatenating the global and local feature maps before passing them in, meaning out input dimension is
        # actually twice this model's hidden dimension.
        self.point_head = point_head_class(dim=2 * dim)
        """
        The camera decoder will predict the relative SE3 transformation to the next frame.
        We are parameterizing our camera with three scalars for the translation and six scalars for the rotation.
        Details of how the rotation prediction works are in the forward() method.
        """
        self.pose_head = pose_head_class(dim=2 * dim)

        # Take all four intermediate feature maps from PP3DR. We have 36 // 2 == 18 blocks each
        self.output_blocks = [3, 7, 12, 17]

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # 1. Standard Linear projection init (Transformers / TTT)
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # 2. Standard 2D Convolution base init
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # 3. Dedicated zero-init for SpatialDepthwiseConv (Type-based, name-agnostic!)
        elif isinstance(m, SpatialDepthwiseConv):
            nn.init.zeros_(m.depthwise_convolution.weight)
            if m.depthwise_convolution.bias is not None:
                nn.init.zeros_(m.depthwise_convolution.bias)

        # 4. Dedicated zero-init for ResidualConvUnit final conv (Type-based!)
        elif isinstance(m, ResidualConvUnit):
            # conv1 keeps its Kaiming init from step 2; only conv2 is zeroed:
            nn.init.zeros_(m.conv2.weight)
            if m.conv2.bias is not None:
                nn.init.zeros_(m.conv2.bias)

        self._init_head_overrides()

    def _init_head_overrides(self):
        # Final output projection layers that output log_depths / raw_uncertainty
        # should use small normal init so predictions start near 0.0 (exp(-s) = 1.0)
        # Find the very last conv in the sequence
        for mod in reversed(list(self.point_head.depth_head.final_upscale.modules())):
            if isinstance(mod, nn.Conv2d):
                nn.init.normal_(mod.weight, std=1e-3)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
                break

    def forward(self, x: torch.Tensor, rope_x, rope_y, original_height, original_width) -> dict:
        """
        The input must be pre-processed so its height and width are multiples of the patch size.

        rope_x and rope_y come from the dataloader.

        Parameters
        ----------
        images: (batch size, sequence length, RGB, height, width)
        rope_x: The 2D x-coordinates to be fed into RoPE (B, L, HW)
        rope_y: The 2D y-coordinates to be fed into RoPE; same shape as above.

        Returns
        -------
        "log_depth": Log depth for each pixel's point (B, L, H, W)

        "focal_length": Unified focal length per sequence (B,)

        "relative_camera_translation": (x, y, z) translation between this camera pose and the next one (B, L - 1, 3)

        "relative_camera_rotation": 3x3 rotation matrix between this camera pose and the next one (B, L - 1, 3, 3)
        """
        B, L, C, H, W = x.shape
        H_p, W_p = H // 16, W // 16

        x = x.flatten(0, 1)
        x = transforms.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)

        # Step 1: ViT
        # ViT is frame-wise. We need to collapse the batch dimension into the sequence dimension before passing it into Dino.
        local_map, outputs = self.ViTTT(x)
        # x.shape == (B * L, num_registers + HW // 256, dim)

        """
        We will now perform alternating-attention, starting with global attention and ending with frame-wise attention.

        The RoPE coordinates have been pre-calculated and are provided by the dataloader. This is because they're
        dependent on the dimensions of the raw images.

        rope2d has shape (B * L, HW, head_dim), and rope3d has shape (B, L, HW, head_dim).
        """
        # rope2d, rope3d = self.rope2d(rope_x, rope_y), self.rope3d(rope_x, rope_y)
        rope = self.rope2d(rope_x, rope_y)
        for i in range(self.blocks_each):
            if self.training and i >= self.start_checkpointing:
                # Global attention: absorb frame-length into patch-length dimension.
                global_map = checkpoint(self.global_blocks[i], local_map, rope, L, use_reentrant=False)
                # Local attention: absorb frame-length into batch dimension. Sequence length is now patch-length.
                local_map = checkpoint(self.local_blocks[i], global_map, rope, L, H_p, W_p, use_reentrant=False)
            else:
                global_map = self.global_blocks[i](local_map, rope, L)
                local_map = self.local_blocks[i](global_map, rope, L, H_p, W_p)
            if i in self.output_blocks:
                outputs.append(torch.cat([global_map.reshape(*local_map.shape), local_map], dim=-1))

        # Step 2: Per-task decoders
        focal_length, log_depths, raw_uncertainty = self.point_head(
            outputs, x,
            L, H, W, original_height, original_width
        )
        # We're predicting the relative pose from this frame to the next one, which is why our sequence length is L - 1.
        poses = self.pose_head(outputs[-1], L) # (B, L - 1, 9)

        # Get the rotation matrix by orthogonalizing the first two 3D vectors, then taking the cross product for the third.
        # We're going to construct the actual columns as rows of the current matrix, then take the transpose at the end.
        # eps is increased from 1e-12 to 1e-6 in an attempt to increase stability.
        a = F.normalize(poses[:, :, 3:6], dim=-1, eps=1e-6)
        b = poses[:, :, 6:]
        # Unsqueeze after the dot product so the resulting scalars broadcast correctly against the vector a.
        b = F.normalize(b - torch.linalg.vecdot(a, b, dim=-1).unsqueeze(-1) * a, dim=-1, eps=1e-6)
        c = torch.linalg.cross(a, b, dim=-1)

        return {
            # e^-80 to e^80 is safely within the range of bfloat16.
            "log_depths": torch.clamp(log_depths, min=-80, max=80).to(torch.float32),
            "raw_uncertainty": torch.clamp(raw_uncertainty, min=-80, max=80).to(torch.float32),
            "focal_length": focal_length.to(torch.float32),
            "relative_camera_translations": poses[:, :, :3].to(torch.float32),  # (B, L - 1, 3) -> 3 scalars, (x, y, z)
            "relative_camera_rotations": torch.stack([a, b, c], dim=-1).to(torch.float32)  # (B, L - 1, 3, 3)
        }