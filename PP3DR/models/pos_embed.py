import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
@torch.compile()
class RopePositionEmbedding(nn.Module):
    """
    2D RoPE with centered, normalized coordinates.
    """
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This software may be used and distributed in accordance with
    # the terms of the DINOv3 License Agreement.
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        if self.training:
            # Shift coords by adding a uniform value in [-shift, shift]
            if self.shift_coords is not None:
                shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
                coords += shift_hw[None, :]

            # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
            if self.jitter_coords is not None:
                jitter_max = np.log(self.jitter_coords)
                jitter_min = -jitter_max
                jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
                coords *= jitter_hw[None, :]

            # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
            if self.rescale_coords is not None:
                rescale_max = np.log(self.rescale_coords)
                rescale_min = -rescale_max
                rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
                coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                # Construct an integer range from [0, self.D_head // 4)
                # assert embed_dim % (4 * num_heads) == 0 -> self.d_head is divisible by 4. So this expression is just:
                # 4 * [0, self.D_head // 4) / self.D_head -> self.D_head // 4 steps from [0, 1)
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods

@torch.compile()
class Rope2D(nn.Module):
    """
    2D RoPE with absolute 2D coordinates, centered per image.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        base: float | None = 10000.0,
        min_period: float | None = None,
        max_period: float | None = None,
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, rope_x, rope_y) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        rope_x: (B, L, HW)
        rope_y: (B, L HW)
        """
        B, L, HW = rope_x.shape
        dd = {"device": self.periods.device, "dtype": self.dtype}

        coords = torch.stack([rope_y, rope_x], dim=-1)  # (B, L, HW, 2)

        if self.training:
            # Shift coords: Shape (B, 1, 2) - Independent random shift per sequence
            if self.shift_coords is not None:
                shift = torch.empty(B, 1, 1, 2, **dd).uniform_(-self.shift_coords, self.shift_coords)
                coords += shift

            # Jitter coords: Independent stretch/squash for X and Y per sequence
            if self.jitter_coords is not None:
                jitter_max = math.log(self.jitter_coords)
                jitter = torch.empty(B, 1, 1, 2, **dd).uniform_(-jitter_max, jitter_max).exp()
                coords *= jitter

            # Rescale coords: Uniform zoom in/out per sequence (same scalar for X and Y to preserve aspect ratio)
            if self.rescale_coords is not None:
                rescale_max = math.log(self.rescale_coords)
                rescale = torch.empty(B, 1, 1, 1, **dd).uniform_(-rescale_max, rescale_max).exp()
                coords *= rescale

        # Prepare angles and sin/cos
        coords = coords.view(B * L, HW, 2)
        angles = 2 * math.pi * coords.unsqueeze(-1) / self.periods.view(1, 1, 1, -1)  # (B * L, HW, 2, D//4)
        angles = angles.flatten(2, 3)  # (B * L, HW, D//2)
        # Explicitly tile the last dimension only (1x on B * L, 1x on HW, 2x on D_head)
        angles = angles.tile(1, 1, 2)  # (B * L, HW, D)

        return torch.sin(angles), torch.cos(angles) # 2 x (B * L, HW, D)

    def _init_weights(self):
        dd = {"device": self.periods.device, "dtype": self.dtype}
        periods = self.base ** (
            # Construct an integer range from [0, self.D_head // 4)
            # assert embed_dim % (4 * num_heads) == 0 -> self.d_head is divisible by 4. So this expression is just:
            # 4 * [0, self.D_head // 4) / self.D_head -> self.D_head // 4 steps from [0, 1)
            2 * torch.arange(self.D_head // 4, **dd) / (self.D_head // 2)
        )  # [D//4]
        self.periods.data = periods

@torch.compile()
class Rope3D(nn.Module):
    """
    Hard-coded for dim=1280 and heads=20. Gives us 2 feature dimensions per trig function (4 per head) without RoPE.
    NOTE: These are FEATURE DIMENSIONS (per token) which don't get rope encodings, not entire tokens.
    Base frequency 100 for spatial components, and 1e4 for temporal components.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        # A base of 1e4 allows us to process a sequence length of a little under 4000 without aliasing (and no
        # coordinate normalization). That's a little under 4000 16x16 patches, and a little under 4000 frames.
        spatial_base: float = 10000.0,
        temporal_base: float= 10000.0,
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = torch.float32,
        device: torch.device | None = None,
    ):
        super().__init__()

        D_head = embed_dim // num_heads
        """
        Divide by 2 for sin/cos. Divide by 3 for the H, W, and temporal dimensions.
        For dim = 1280 and num_heads = 20, there are 4 remainder dimensions left over; 2 for sin and 2 for cos.
        We just won't apply RoPE to those dimensions.
        """
        self.rope_dim = D_head // 6
        self.remainder_per_half = (D_head % 6) // 2
        self.spatial_base = spatial_base
        self.temporal_base = temporal_base
        self.D_head = D_head
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        self.register_buffer(
            "periods",
            torch.empty(3, self.rope_dim, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, rope_x, rope_y) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        rope_x: (B, L, HW)
        rope_y: (B, L HW)

        Returns
        -------
        2 x (B, LHW, D): A sin/cos embedding for every token in the global sequence (of which there are LHW).
        """
        B, L, HW = rope_x.shape
        dd = {"device": self.periods.device, "dtype": self.dtype}

        # rope_t is torch.arange(L, *dd) (un-normalized from 0 to L)
        # We want to "duplicate" it across B and HW, since the frame index depends *only* on the sequence dimension.
        rope_t = torch.arange(L, **dd).view(1, L, 1).expand(B, L, HW) # (B, L, HW)
        # Before flattening, the shape is (B, L, HW, 3), where the last dimension consists of a patch's t-coordinate,
        # y-coordinate, and x-coordinate, in that order.
        coords = torch.stack([rope_t, rope_y, rope_x], dim=-1).flatten(1, 2) # (B, LHW, 3)

        if self.training:
            # Slice out the spatial coordinates to apply augmentations ONLY to Y and X
            # Shape: (B, LHW, 2)
            spatial_coords = coords[..., 1:]

            if self.shift_coords is not None:
                shift = torch.empty(B, 1, 2, **dd).uniform_(-self.shift_coords, self.shift_coords)
                spatial_coords += shift

            if self.jitter_coords is not None:
                jitter_max = math.log(self.jitter_coords)
                jitter = torch.empty(B, 1, 2, **dd).uniform_(-jitter_max, jitter_max).exp()
                spatial_coords *= jitter

            if self.rescale_coords is not None:
                rescale_max = math.log(self.rescale_coords)
                rescale = torch.empty(B, 1, 1, **dd).uniform_(-rescale_max, rescale_max).exp()
                spatial_coords *= rescale

            # Assign the augmented spatial coordinates back
            coords[..., 1:] = spatial_coords

        # Prepare angles and sin/cos
        # Why does self.periods.view() have a 3 in there? Because we decided to separate the temporal and spatial
        # periods in case we ever wanted to make their bases different. Rope2D just has a 1 there, because the x and
        # y-coordinates use the same base period.
        # (B, LHW, 3, self.rope_dim)
        angles_scaled = (2 * math.pi * coords.unsqueeze(-1)) / self.periods.view(1, 1, 3, self.rope_dim)
        angles_scaled = angles_scaled.flatten(2, 3)  # (B, LHW, 3 * self.rope_dim) = (B, LHW, self.D_head // 2)
        angles = torch.zeros(B, L * HW, self.D_head // 2, **dd)
        angles[:, :, self.remainder_per_half:] = angles_scaled
        # Explicitly tile the last dimension only (1x on B, 1x on LHW, 2x on D_head)
        angles = angles.tile(1, 1, 2)  # (B, LHW, self.D_head)
        angles = angles.view(B, L, HW, -1)

        return torch.sin(angles), torch.cos(angles)

    def _init_weights(self):
        dd = {"device": self.periods.device, "dtype": self.dtype}
        # Create self.rope_dim steps from [0, 1)
        exponents = torch.arange(self.rope_dim, **dd) / self.rope_dim # (self.rope_dim,)
        self.periods.data[0] = self.temporal_base ** exponents
        self.periods.data[1:] = self.spatial_base ** exponents

if __name__ == "__main__":
    rope2d = RopePositionEmbedding(embed_dim = 1280, num_heads = 20, dtype = torch.float32)
    rope3d = Rope3D(1280, 20, dtype = torch.float32)
    print("RoPE2D:")
    print(rope2d(2, 3)[0].shape)
    print(rope2d(2, 3)[0][0])
    print("RoPE3D:")
    print(rope3d(5, 2, 3)[0].shape)
    print(rope3d(5, 2, 3)[0][0])
