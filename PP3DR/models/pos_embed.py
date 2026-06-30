# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
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
        # A base of 1e4 allows us to process up to a little under 4000 frames without aliasing (no normalization).
        spatial_base: float = 100,
        temporal_base: float= 10000,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
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
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(self.rope_dim, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, L: int, H: int, W: int) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        L: Number of frames in the sequence
        H: Number of 16x16 patches constituting a column of the image
        W: Number of 16x16 patches constituting a row of the image

        Returns
        -------
        2 * [LHW, D]: A sin/cos embedding for every token in the global sequence (of which there are LHW).
        """
        dd = {"device": self.periods.device, "dtype": self.dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = 2 * (torch.arange(0.5, H, **dd) / max_HW) - 1  # [H]
            coords_w = 2 * (torch.arange(0.5, W, **dd) / max_HW) - 1  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = 2 * (torch.arange(0.5, H, **dd) / min_HW) - 1  # [H]
            coords_w = 2 * (torch.arange(0.5, W, **dd) / min_HW) - 1  # [W]
        elif self.normalize_coords == "separate":
            coords_h = 2 * (torch.arange(0.5, H, **dd) / H) - 1  # [H]
            coords_w = 2 * (torch.arange(0.5, W, **dd) / W) - 1  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        # coords_l is torch.arange(L, *dd) (un-normalized from 0 to L)
        # DEBUG
        coords_l = torch.arange(L, **dd)
        coords = torch.stack(torch.meshgrid(coords_l, coords_h, coords_w, indexing="ij"), dim=-1)  # [L, H, W, 3]
        coords = coords.flatten(0, 2)  # [LHW, 3]
        # The re-centering below got moved into the normalization branches, because I don't want to normalize the L dim.
        # coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

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
        # coords is [LHW, 3]. periods is [3, self.rope_dim]
        angles = torch.empty(coords.shape[0], self.D_head // 2, **dd)
        # [LHW, self.D_head // 2]
        angles[:, self.remainder_per_half:] = (2 * math.pi * coords[:, :, None] / self.periods[None, :, :]).flatten(1, 2)
        angles[:, :self.remainder_per_half] = 0
        angles = angles.tile(2)  # [LHW, self.D_head]
        cos = torch.cos(angles)  # [LHW, self.D_head]
        sin = torch.sin(angles)  # [LHW, self.D_head]

        return sin, cos  # 2 * [LHW, self.D_head]

    def _init_weights(self):
        dd = {"device": self.periods.device, "dtype": self.dtype}
        # Create self.rope_dim steps from [0, 1)
        exponents = torch.arange(self.rope_dim, **dd) / self.rope_dim # [self.rope_dim]
        self.periods.data = torch.empty(3, self.rope_dim, **dd)
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
