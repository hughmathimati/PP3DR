import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange
from models.PP3DR import PoseHead, FocalHead
from models.vggt_omega_depth_head import DenseHead
from zipmap.models.aggregator_ttt import Aggregator


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
        self.depth_head = DenseHead(dim)

    def forward(
            self, outputs: torch.Tensor, L, H, W, original_height, original_width
    ) -> torch.Tensor:
        """
        `x` has shape (B * L, X, dim).
        `outputs` has four tensors of shape (B * L, X, dim).
        """
        # 4 x (B * L, HW // 256, dim)
        focal = self.focal_head(outputs[-1][:, :self.num_registers, :], L, H, W)
        four_dim = [rearrange(y, "(B L) X dim -> B L X dim", L=L) for y in outputs]
        depths, confidence = self.depth_head(
            four_dim, four_dim[0].shape[0], L,
            H // 16, W // 16,
            original_height, original_width,
            self.num_registers
        )
        return focal, depths, confidence

class ZipMapPP3DRAblation(nn.Module):
    def __init__(
            self,
            dim: int = 1024,
            num_heads: int = 16,
            point_head_class=PointHead,
            pose_head_class=PoseHead,
    ):
        super().__init__()

        # Parameters needed for the base_trainer
        self.encoder_blocks = 24  # 24 blocks in DINOv2 / DINOv3 ViT-Large
        self.decoder_blocks = 48  # 24 frame_blocks + 24 global_blocks

        # 1. Instantiate the ZipMap Backbone
        # Default ZipMap uses 4 register tokens + 1 camera token = 5 special tokens (patch_start_idx = 5)
        self.aggregator = Aggregator(
            embed_dim=dim,
            num_heads=num_heads,
            depth=24,  # ZipMap uses 24 blocks
            patch_embed="dinov2_vitl14_reg",
            num_register_tokens=4,
            aa_order=["frame", "global"],
            ttt_config={"ttt_mode": True, "params": {
                    "head_dim": dim // num_heads,  # 1024 // 16 = 64
                    "muon_update_steps": 5,        # Matches zeropower_via_newtonschulz5
                }
            },  # Enable TTT for global blocks
        )
        # CRITICAL for DDP: mark all patch_embed parameters as requires_grad=False
        self.aggregator.patch_embed.requires_grad_(False)
        self.aggregator.patch_embed.eval()

        # 3. Custom Heads
        # We pass 5 to num_registers because ZipMap has 1 camera_token + 4 register_tokens
        self.point_head = point_head_class(dim=2 * dim, num_registers=5)
        self.pose_head = pose_head_class(dim=2 * dim, num_registers=5, output_dim=9)

    # Two unused parameters just so the forward() has the same function signature as PP3DR (for the base trainer to call).
    def forward(self, images: torch.Tensor, rope_x, rope_y, original_height, original_width) -> dict:
        """
        images: (B, L, 3, H, W)
        ZipMap natively handles RoPE via its own PositionGetter, so rope_x and rope_y are no longer needed here.
        """
        B, L, C_in, H, W = images.shape

        # 1. Run ZipMap Aggregator
        # outputs is a list of 4 tensors of shape (B, L, P_total, 2 * dim)
        """
        This interpolation is needed since the input images + depth head are for patch size 16, but the Aggregator's
        patch embed (Dinov2) has patch size 14.
        """
        # Compute the targetpatch grid (e.g., 32x32 for 512x512, or 16x16 for 256x256)
        H_p, W_p = H // 16, W // 16
        # Resize to (H_p * 14, W_p * 14) -> (448, 448) for 512x512, or (224, 224) for 256x256
        images_14 = F.interpolate(
            images.flatten(0, 1), size=(H_p * 14, W_p * 14), mode="bilinear", align_corners=False
        ).view(B, L, 3, H_p * 14, W_p * 14)
        raw_outputs, patch_start_idx, _ = self.aggregator(images_14)

        # 2. Reshape from (B, L, P_total, 2048) -> (B * L, P_total, 2048)
        outputs = [rearrange(out, "B L P dim -> (B L) P dim") for out in raw_outputs]

        # 3. Extract the final hidden state to serve as `x` for the Pose and Focal heads
        final_x = outputs[-1]  # (B * L, P_total, 1024)

        # 4. Route through Custom Heads
        focal_length, log_depths, raw_uncertainty = self.point_head(
            outputs=outputs,
            L=L,
            H=H,
            W=W,
            original_height=original_height,
            original_width=original_width
        )

        poses = self.pose_head(outputs[-1], L) # (B, L - 1, 9)

        # 5. Rotation Orthogonalization (Same as your original code)
        a = F.normalize(poses[:, :, 3:6], dim=-1, eps=1e-6)
        b = poses[:, :, 6:]
        b = F.normalize(b - torch.linalg.vecdot(a, b, dim=-1).unsqueeze(-1) * a, dim=-1, eps=1e-6)
        c = torch.linalg.cross(a, b, dim=-1)

        return {
            "log_depths": torch.clamp(log_depths, min=-80, max=80).to(torch.float32),
            "raw_uncertainty": torch.clamp(raw_uncertainty, min=-80, max=80).to(torch.float32),
            "focal_length": focal_length.to(torch.float32),
            "relative_camera_translations": poses[:, :, :3].to(torch.float32),
            "relative_camera_rotations": torch.stack([a, b, c], dim=-1).to(torch.float32)
        }