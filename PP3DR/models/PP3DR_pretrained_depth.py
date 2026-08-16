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

from models.BidirectionalLaCT import GlobalLaCT, LocalLaCT
from models.pos_embed import RopePositionEmbedding, Rope3D, Rope2D
from models.PP3DR import PP3DR, GlobalBlock, LocalBlock, PoseHead, FocalHead
from models.ViTTT import ViTTT
from models.vggt_omega_depth_head import DenseHead


class PointHead(nn.Module):
    def __init__(
            self,
            dim=2048,
            num_registers=5,
    ):
        super().__init__()
        self.dim = dim
        self.num_registers = num_registers
        self.focal_head = FocalHead(dim=dim)
        self.depth_head = DenseHead()
        self.depth_head.load_state_dict(torch.load("/vulcanscratch/hughma/PP3DR/models/vggt_dense_head.pth",
                                              weights_only=True,
                                              map_location="cpu"))
        for parameter in self.depth_head.parameters():
            parameter.requires_grad = False
        self.depth_head.eval()

    def train(self, mode=True):
        """
        Keep the pretrained depth head in eval mode.
        """
        super().train(mode)
        self.depth_head.eval()
        return self

    def forward(self, x: torch.Tensor, L, H, W) -> torch.Tensor:
        """
        Inputs consist of:
        1. The output of block 6/12 (index 5) from ViTTT
        2. The output at the end of ViTTT
        3. The output of block 18/36 (index 17) from PP3DR
        4. The output at the end of PP3DR

        Parameters
        ----------
        x: Input tensor
        rope2d: 2D RoPE positional embedding
        rope3d: 3D RoPE positional embedding
        L: Number of frames. Necessary for the rearrange operations.
        """
        # 4 x (B * L, HW // 256, dim)
        four_dim = [rearrange(y, "(B L) X dim -> B L X dim", L=L) for y in x]
        focal = self.focal_head(x[-1][:, :self.num_registers, :], L, H, W)
        log_depths, raw_depth_confidence = self.depth_head(four_dim, four_dim[0].shape[0], L, H, W, self.num_registers)
        return focal, log_depths, raw_depth_confidence


@torch.compile()
class PP3DR_pretrained_depth(PP3DR):
    def __init__(
            self,
            dim: int = 1024,
            # dim must be divisible by num_heads. VGGT-Omega uses 16 heads.
            # Rope3D was written to be flexible with input dims, but a dim of 1024 with 16 heads gives us a head_dim of
            # 64, which is actually the exact same as a dim of 1280 with 20 heads. Thus, there's nothing to worry about.
            num_heads: int = 16,
            encoder_blocks: int = 12,  # Pi3 is 36 ViT blocks
            decoder_blocks: int = 36,  # Pi3 is 36 decoder blocks.
            ffn_ratio: int = 4,
            num_registers: int = 5,
            # How many blocks EACH not to checkpoint (total # is twice as many).
            # Keep in mind we're already not checkpointing all 12 ViTTT blocks.z
            start_checkpointing=11,
            # freeze_feature_extractor is kept for compatibility with the BaseTrainer but is not used.
            freeze_feature_extractor=False,
            point_head_class=PointHead,
            pose_head_class=PoseHead,
            output_blocks = [3, 8, 13, 17] # These are block_each indices. The highest block_each index is 17.
    ):
        nn.Module.__init__(self) # Don't call the PP3DR init.
        assert decoder_blocks % 2 == 0, f"Number of decoder blocks ({decoder_blocks}) must be even for alternating global and frame-wise attention"
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

        self.rope2d = Rope2D(embed_dim=dim, num_heads=num_heads, device="cuda")
        self.rope3d = Rope3D(embed_dim=dim, num_heads=num_heads, device="cuda")

        self.ViTTT = ViTTT(dim, num_heads, encoder_blocks, ffn_ratio, drop_rates=ViTTT_drop_rates, output_blocks=[])
        # Don't load the pretrained ViTTT, as its dimension is 1280 instead of 1024 as required here.
        # ViTTT puts in the registers for me.
        self.num_registers = num_registers

        # Per-task decoders
        self.point_head = point_head_class()
        """
        The camera decoder will predict the relative SE3 transformation to the next frame.
        We are parameterizing our camera with three scalars for the translation and six scalars for the rotation.
        Details of how the rotation prediction works are in the forward() method.
        """
        self.pose_head = pose_head_class(dim=dim)

        self.output_blocks = output_blocks

    def train(self, mode=True):
        """
        Bypass the PP3DR train override to just use the original default.
        """
        nn.Module.train(self, mode)
        return self

    def forward(self, x: torch.Tensor, rope_x, rope_y) -> dict:
        """
        The input must be pre-processed so its height and width are multiples of the patch size.

        rope_x and rope_y come from the dataloader.

        Parameters
        ----------
        images: (batch size, RGB, height, width)
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

        # Step 1: ViT
        # ViT is frame-wise. We need to collapse the batch dimension into the sequence dimension before passing it into Dino.
        x, outputs = self.ViT(x.flatten(0, 1))
        # x.shape == (B * L, num_registers + HW // 256, dim)

        """
        We will now perform alternating-attention, starting with global attention and ending with frame-wise attention.

        The RoPE coordinates have been pre-calculated and are provided by the dataloader. This is because they're
        dependent on the dimensions of the raw images.

        rope2d has shape (B * L, HW, head_dim), and rope3d has shape (B, L, HW, head_dim).
        """
        rope2d, rope3d = self.rope2d(rope_x, rope_y), self.rope3d(rope_x, rope_y)
        for i in range(self.blocks_each):
            if self.training and i >= self.start_checkpointing:
                # Global attention: absorb frame-length into patch-length dimension.
                global_output = checkpoint(self.global_blocks[i], x, rope3d, L, use_reentrant=False)
                # Local attention: absorb frame-length into batch dimension. Sequence length is now patch-length.
                x = checkpoint(self.local_blocks[i], global_output, rope2d, L, use_reentrant=False)
            else:
                global_output = self.global_blocks[i](x, rope3d, L)
                x = self.local_blocks[i](global_output, rope2d, L)
            if i in self.output_blocks:
                outputs.append(torch.cat([x, global_output.view(B * L, -1, self.dim)], dim=-1))
        # x.shape == (B * L, num_registers + HW // 256, 2 * dim)

        # Step 2: Per-task decoders
        """
        If you're observant, you'll notice that every tensor added to `outputs` was the output of a LocalBlock.
        This is fine, because we're doing dense depth prediction. This would be harder to justify for the PoseHead.
        """
        focal_length, log_depths, raw_uncertainty = self.point_head(outputs, L, H, W)
        # We're predicting the relative pose from this frame to the next one, which is why our sequence length is L - 1.
        poses = self.pose_head(x, L)[:, :-1]  # (B, L - 1, 9)

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