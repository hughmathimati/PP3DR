import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.linalg import vector_norm
import torch.cuda.amp as amp
from einops import rearrange
from typing import Callable, Tuple
from torch import Tensor

def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (vector_norm(x, dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    # X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    X = X / (vector_norm(X, dim=(1, 2), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = (
                b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X

# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    # print("rope_apply shapes:", x.shape, cos.shape) # DEBUG
    return (x * cos) + (rope_rotate_half(x) * sin)

def apply_rope(q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    """
    If this function is getting called from BidirectionalLaCT, q and k will have shape (B L nh) X hd, and rope will have
    shape (HW, hd). The block itself instead uses (b h) l d for q and k, but this is only because the feature extractor
    absorbs L into B (X is then renamed to l).
    """
    # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
    q_dtype = q.dtype
    k_dtype = k.dtype
    sin, cos = rope
    rope_dtype = sin.dtype
    q = q.to(dtype=rope_dtype)
    k = k.to(dtype=rope_dtype)
    N = q.shape[-2]
    prefix = N - sin.shape[-2] # prefix is the number of special (register) tokens. # special = N - HW.
    assert prefix >= 0
    """
    The original DINO RoPE had [B * head, hw, D//head] and q/k[:, :, :prefix, :]. Since we've merged the first two dimensions
    (batch size and # heads), we simply remove one of the first two dimensions from our array accesses.
    """
    q_prefix = q[:, :prefix, :]
    q = rope_apply(q[:, prefix:, :], sin, cos)  # [B * head, hw, D//head]
    q = torch.cat((q_prefix, q), dim=-2)  # [B * head, N, D//head]
    k_prefix = k[:, :prefix, :]
    k = rope_apply(k[:, prefix:, :], sin, cos)  # [B * head, hw, D//head]
    k = torch.cat((k_prefix, k), dim=-2)  # [B * head, N, D//head]
    q = q.to(dtype=q_dtype)
    k = k.to(dtype=k_dtype)
    return q, k

def bidirectional_lact_swiglu(
        w0: torch.Tensor,  # [b, dh, dk]
        w1: torch.Tensor,  # [b, dv, dh]
        w2: torch.Tensor,  # [b, dh, dk]
        q: torch.Tensor,  # [b, l, dk]
        k: torch.Tensor,  # [b, l, dk]
        v: torch.Tensor,  # [b, l, dv]
) -> torch.Tensor:
    """
    Bidirectional LaCT with SwiGLU fast weight function.
    Modified so W1 is the only fast weight, with W2 and W0 as slow weights.
    Constant learning rate of 1. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are mostly likely fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.
    Outputs:
        o: [b, l, dv]
    """

    # adding detach here sometimes improves stability.
    w1_norm = vector_norm(w1.detach(), dim=2, keepdim=True)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    ######### update the fast weight w0, w1, w2 with test-time training #########

    #### Forward pass with key
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    gate_before_act = torch.bmm(w0, k.transpose(1, 2))
    hidden_before_mul = torch.bmm(w2, k.transpose(1, 2))
    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

    #### Backward pass to compute fast weight gradients
    # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
    # dhidden = torch.bmm(w1.transpose(1, 2), v)

    # dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
    # dgate = dhidden * hidden_before_mul
    # dgate_before_act = silu_backprop(dgate, gate_before_act)

    # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
    dw1 = torch.bmm(v, (hidden.transpose(1, 2)).type_as(v))  # [b, d, d]
    dw1 = zeropower_via_newtonschulz5(dw1)

    w1 = w1 + dw1

    w1 = w1 / (vector_norm(w1, dim=2, keepdim=True) + 1e-5) * w1_norm

    ######### apply the updated fast weights to the query #########

    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, q)
    gate = F.silu(torch.bmm(w0, q), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    o = torch.bmm(w1, gate * h).transpose(1, 2)

    return o


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


class BidirectionalLaCT(torch.nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            inter_multi: float = 1, # Hidden dimension = head dimension * inter_multi.
            use_o_norm: bool = True,  # recommended to be True
            qk_l2_norm: bool = True,  # recommended to be True
            use_muon: bool = True,  # if your seq len > head_dim * 2, recommended to be True
            layer_norm: Callable[..., nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inter_multi = inter_multi
        self.use_o_norm = use_o_norm
        self.qk_l2_norm = qk_l2_norm

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # create initial fast weights
        d_in, d_out = self.head_dim, self.head_dim
        d_h = int(self.head_dim * self.inter_multi)

        self.w0 = nn.Parameter(torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(self.num_heads, d_out, d_h) / math.sqrt(d_h))
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in)
        )

        self.use_muon = use_muon

        self.use_o_norm = use_o_norm
        if self.use_o_norm:
            self.o_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        else:
            self.o_norm = nn.Identity()

        self.layer_norm = layer_norm(dim, bias=False) # New

    def forward(self, x: torch.Tensor, rope) -> torch.Tensor:
        """
        x: [b, l, d]
        """
        x = self.layer_norm(x) # New

        qkv = F.silu(self.to_qkv(x), inplace=True)  # SiLU - Linear

        # [b * num_heads, l, head_dim]
        q, k, v = rearrange(
            qkv,
            "b l (qkv h d) -> qkv (b h) l d",
            qkv=3,
            h=self.num_heads,
            d=self.head_dim,
        )

        if self.qk_l2_norm:
            q = l2_norm(q)
            k = l2_norm(k)

        # Original DINO rope expects q, k to be (B, # heads, sequence length, dim per head).
        # Here we instead use (B * # heads, sequence length, dim per head), which simply entails merging the first two dimensions.
        q, k = apply_rope(q, k, rope)

        # [nh, d, d] -> [b * nh, d, d]
        w0 = self.w0.repeat(x.shape[0], 1, 1)
        w1 = self.w1.repeat(x.shape[0], 1, 1)
        w2 = self.w2.repeat(x.shape[0], 1, 1)

        # [b * num_heads, l, head_dim]
        output = bidirectional_lact_swiglu(w0, w1, w2, q, k, v)

        output = self.o_norm(output)
        output = rearrange(output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0])
        output = self.o_proj(output)

        # [b, l, d]
        return output


class GlobalLaCT(torch.nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_registers: int = 5,
            inter_multi: float = 1, # Hidden dimension = head dimension * inter_multi.
            use_o_norm: bool = True,  # recommended to be True
            qk_l2_norm: bool = True,  # recommended to be True
            use_muon: bool = True,  # if your seq len > head_dim * 2, recommended to be True
            layer_norm: Callable[..., nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_registers = num_registers
        self.inter_multi = inter_multi
        self.use_o_norm = use_o_norm
        self.qk_l2_norm = qk_l2_norm

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # create initial fast weights
        d_in, d_out = self.head_dim, self.head_dim
        d_h = int(self.head_dim * self.inter_multi)

        self.w0 = nn.Parameter(torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(self.num_heads, d_out, d_h) / math.sqrt(d_h))
        self.w2 = nn.Parameter(torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in))

        self.use_muon = use_muon

        self.use_o_norm = use_o_norm
        if self.use_o_norm:
            self.o_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        else:
            self.o_norm = nn.Identity()

        self.layer_norm = layer_norm(dim, bias=False) # New

    def forward(self, x: torch.Tensor, rope3d, L) -> torch.Tensor:
        """
        GlobalBlock forward.
        Input shape: ((B L), X, dim)
        Output shape: (B, (L X), dim)
        rope3d: (B, L, HW, head_dim)
        """
        B, X = x.shape[0] // L, x.shape[1]

        x = self.layer_norm(x) # New
        # ((B L), X, 3 * dim)
        qkv = F.silu(self.to_qkv(x), inplace=True)  # SiLU - Linear

        # ((B L), X, 3 * dim) -> (B, L, X, 3, nh, hd) -> 3 * (B, L, X, nh, hd)
        q, k, v = qkv.view(B, L, X, 3, self.num_heads, self.head_dim).unbind(3)

        if self.qk_l2_norm:
            q = l2_norm(q)
            k = l2_norm(k)

        """"
        We need to:
        1. Remove the register tokens
        2. Reshape q and k
        3. Apply rope3d
        4. Reshape back
        5. Add back the register tokens
        6. Reshape for the global TTT
        
        The order of operations below does not strictly follow the order above, because I use view instead of rearrange
        to minimise memory reallocations.
        """
        # Step 1.
        q_registers, k_registers = q[:, :, :self.num_registers, :, :], k[:, :, :self.num_registers, :, :] # (B, L, self.num_registers, nh, hd)
        q_tokens, k_tokens = q[:, :, self.num_registers:, :, :], k[:, :, self.num_registers:, :, :] # (B, L, HW, nh, hd)

        sin, cos = rope3d # 2 x (B, L, HW, hd)
        sin, cos = sin.unsqueeze(-2), cos.unsqueeze(-2) # 2 x (B, L, HW, 1, hd)
        rope_dtype = sin.dtype

        # This combines/replaces steps 2 and 3. Notice that instead of absorbing nh into B and creating a new tensor
        # with LHW after removing the register tokens, we instead separate L from HW and broadcast the RoPE with a view.
        q_tokens = rope_apply(q_tokens.to(dtype=rope_dtype), sin, cos).to(dtype=q.dtype)
        k_tokens = rope_apply(k_tokens.to(dtype=rope_dtype), sin, cos).to(dtype=k.dtype)
        # Step 4 is now no longer needed, as we maintained the original shape from before!

        # Step 5.
        q, k = torch.cat((q_registers, q_tokens), dim = 2), torch.cat((k_registers, k_tokens), dim = 2)
        # q, k, and v now all have the shape (B, L, X, nh, hd)

        # Step 6.
        q = rearrange(q, "B L X h d -> (B h) (L X) d", h = self.num_heads, L = L)
        k = rearrange(k, "B L X h d -> (B h) (L X) d", h = self.num_heads, L = L)
        v = rearrange(v, "B L X h d -> (B h) (L X) d", h = self.num_heads, L = L)

        # All done! TTT time.
        # [nh, d, d] -> [B * nh, d, d]
        w0 = self.w0.repeat(B, 1, 1)
        w1 = self.w1.repeat(B, 1, 1)
        w2 = self.w2.repeat(B, 1, 1)

        # [b * num_heads, l, head_dim]
        output = bidirectional_lact_swiglu(w0, w1, w2, q, k, v)

        output = self.o_norm(output)
        # Einops complains about not being able to identify L if I write (L X).
        output = rearrange(output, "(B nh) LX hd -> B LX (nh hd)", nh=self.num_heads)
        output = self.o_proj(output)
        return output # Output shape: (B, (L X), dim)


class LocalLaCT(torch.nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_registers: int = 5,
            inter_multi: float = 1, # Hidden dimension = head dimension * inter_multi.
            use_o_norm: bool = True,  # recommended to be True
            qk_l2_norm: bool = True,  # recommended to be True
            use_muon: bool = True,  # if your seq len > head_dim * 2, recommended to be True
            layer_norm: Callable[..., nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_registers = num_registers
        self.inter_multi = inter_multi
        self.use_o_norm = use_o_norm
        self.qk_l2_norm = qk_l2_norm

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # create initial fast weights
        d_in, d_out = self.head_dim, self.head_dim
        d_h = int(self.head_dim * self.inter_multi)

        self.w0 = nn.Parameter(torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(self.num_heads, d_out, d_h) / math.sqrt(d_h))
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in)
        )

        self.use_muon = use_muon

        self.use_o_norm = use_o_norm
        if self.use_o_norm:
            self.o_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        else:
            self.o_norm = nn.Identity()

        self.layer_norm = layer_norm(dim, bias=False) # New

    def forward(self, x: torch.Tensor, rope2d, L) -> torch.Tensor:
        """
        LocalBlock forward.
        Input shape: (B, (L X), dim)
        Output shape: ((B L), X, dim)
        rope2d: (B * L, HW, head_dim)
        """
        B, X = x.shape[0], x.shape[1] // L
        BL = B * L
        x = self.layer_norm(x) # New

        qkv = F.silu(self.to_qkv(x), inplace=True)  # SiLU - Linear

        # (B, L * X, 3 * dim) -> (B * L, X, 3, nh, hd) -> 3 * (B * L, X, nh, hd)
        q, k, v = qkv.view(BL, X, 3, self.num_heads, self.head_dim).unbind(2)

        if self.qk_l2_norm:
            q = l2_norm(q)
            k = l2_norm(k)

        q_registers, k_registers = q[:, :self.num_registers, :, :], k[:, :self.num_registers, :, :] # (B * L, num_registers, nh, hd)
        q_tokens, k_tokens = q[:, self.num_registers:, :, :], k[:, self.num_registers:, :, :] # (B * L, HW, nh, hd)

        sin, cos = rope2d # 2 x (B * L, HW, hd)
        sin, cos = sin.unsqueeze(2), cos.unsqueeze(2) # 2 x (B * L, HW, 1, hd)
        rope_dtype = sin.dtype
        q_tokens = rope_apply(q_tokens.to(dtype=rope_dtype), sin, cos).to(dtype=q.dtype)
        k_tokens = rope_apply(k_tokens.to(dtype=rope_dtype), sin, cos).to(dtype=k.dtype)
        q, k = torch.cat((q_registers, q_tokens), dim = 1), torch.cat((k_registers, k_tokens), dim = 1)

        # [nh, d, d] -> [B * L * nh, d, d]
        # x gets rearranged at the beginning, so we can just do x.shape[0] here.

        w0 = self.w0.repeat(BL, 1, 1)
        w1 = self.w1.repeat(BL, 1, 1)
        w2 = self.w2.repeat(BL, 1, 1)

        # [b * num_heads, l, head_dim]
        # bidirectional_lact_swiglu() expects q, k, and v to have shape (B, L, hd).
        # In our case, that's (B L nh) X hd.
        q = rearrange(q, "BL X nh hd -> (BL nh) X hd")
        k = rearrange(k, "BL X nh hd -> (BL nh) X hd")
        v = rearrange(v, "BL X nh hd -> (BL nh) X hd")
        output = bidirectional_lact_swiglu(w0, w1, w2, q, k, v)

        output = self.o_norm(output)
        output = rearrange(output, "(BL nh) X hd -> BL X (nh hd)", nh=self.num_heads)
        output = self.o_proj(output)
        return output # Output shape: ((B L), X, dim)


def _test_layer():
    B, L, D, HeadDim = 4, 32768, 2048, 512

    layer = BidirectionalLaCT(D, HeadDim, use_muon=True)

    layer = layer.to("cuda")

    x = torch.randn(B, L, D).to("cuda")

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = layer(x)
    print(output.shape, output.dtype)
    print("Input norm", vector_norm(x), "Output norm", vector_norm(output))


if __name__ == "__main__":
    # _test_layer()
    from pos_embed import Rope3D
    # torch.set_printoptions(profile="full")

    rope3d = Rope3D(embed_dim = 1280, num_heads = 20)
    rope = rope3d(3, 4, 5)
    global_block = GlobalLaCT(1280, 20, 5)
    # (2, 3, 5 + 4*5, 1280)
    fake_input = torch.randn(6, 25, 1280)
    output = global_block(fake_input, rope, 3)
    # print("output:")
    # print(output.shape)
    # print(output)

"""
Separated into two groups of 2, 10, 10, 10. 2 and first 10 should be unrotated for first token.

pre q (first token):
[-0.0238, -0.0503,
  0.2705, -0.0352, -0.0478,  0.1530,  0.2707,  0.0686, -0.0118, -0.0978, -0.0395,  0.2509,
 -0.0315, -0.0813,  0.1007,  0.0574, -0.0907,  0.1557, -0.0987, -0.0290, -0.0244, -0.0369,
  0.0010,  0.3094, -0.0762, -0.0083,  0.0580,  0.0260, -0.1080,  0.0419, -0.0733,  0.2371,
  0.0949, -0.0013,
  0.1501, -0.0719,  0.0650, -0.0993, -0.0750, -0.0734,  0.0278,  0.2389, -0.0061,  0.0634,
  0.0464, -0.0721, -0.0943,  0.0525, -0.0362, -0.0326,  0.2524, -0.0417,  0.0423,  0.1780,
  0.1890,  0.2028,  0.1586,  0.3616, -0.0300, -0.0500,  0.1061,  0.0619,  0.0013, -0.0509]
         
post q (first token):
[-0.0238, -0.0503,
  0.2705, -0.0352, -0.0478,  0.1530,  0.2707,  0.0686, -0.0118, -0.0978, -0.0395,  0.2509,
 -0.0464,  0.0681, -0.1202,  0.0703, -0.0912,  0.1239, -0.0205, -0.0363, -0.0193, -0.0235,
 -0.1794, -0.3153,  0.1759,  0.3421,  0.0191, -0.0013, -0.0695,  0.0534, -0.0725,  0.2323,
  0.0949, -0.0013,
  0.1501, -0.0719,  0.0650, -0.0993, -0.0750,  -0.0734, 0.0278,  0.2389, -0.0061,  0.0634,
 -0.0315,  0.0847, -0.0677, -0.0333,  0.0351, -0.0997,  0.2703, -0.0356,  0.0449,  0.1803,
  0.0594, -0.1934,  0.0031,  0.1176, -0.0625, -0.0563,  0.1345,  0.0524,  0.0105, -0.0696]
  
With views instead of rearrange:
Pre q (first token):
>> print(q[0, 0, 5, 0])
[ 8.9055e-02,  1.3856e-01,
 -1.0229e-01,  6.8262e-02,  8.5791e-02, -4.7469e-02, -5.1419e-02, -7.4939e-02,  1.7939e-01,  1.5336e-01, -5.2171e-02,  1.0980e-02,
  5.1566e-01,  1.8699e-01,  1.0851e-01,  9.0635e-02, -6.3388e-03,  1.8955e-01, -5.4753e-02, -3.0784e-03, -1.1288e-01, -5.4435e-02,
  1.3928e-02, -1.2155e-01, -4.8137e-02, -9.9726e-02,  1.3060e-01,  1.1935e-01, -1.9956e-03, -5.9368e-02, -7.6206e-02, -1.6091e-02,
 -7.2688e-02, -2.0103e-02,
  3.5022e-01, -3.1678e-02,  5.5305e-02,  1.2665e-01, -1.2807e-01,  1.6561e-01,  7.5635e-02, -5.2262e-03, -7.1193e-02, -1.2891e-01,
  1.8042e-01, -1.2747e-01,  2.4006e-01,  9.3703e-02, -7.8527e-02, -5.2496e-03,  2.3647e-03, -1.0257e-01,  1.3258e-01, -7.1104e-03,
  1.0125e-01, -1.1256e-01,  6.6460e-02, -4.8984e-03,  8.0950e-02, -1.0191e-01, -2.5256e-04, -1.0357e-01, -7.6349e-02,  8.6758e-02]
  
Post q (first token):
>> print(q_tokens[0, 0, 0, 0])
[ 8.9055e-02,   1.3856e-01,
 -1.0229e-01,   6.8262e-02,  8.5791e-02,  -4.7469e-02, -5.1419e-02, -7.4939e-02,  1.7939e-01,  1.5336e-01, -5.2171e-02,  1.0980e-02,
 -1.8042e-01,  -2.0570e-01,  1.9636e-01,   1.2098e-01, -5.7998e-02,  1.6651e-01, -5.1657e-02, -2.2154e-02, -9.6429e-02, -5.4814e-02,
 -9.1993e-02,   1.2487e-01,  8.0482e-02,  -3.4917e-02,  1.4918e-01,  5.5496e-02, -1.9748e-03, -7.8772e-02, -8.5214e-02, -9.1353e-03,
 -7.2688e-02,  -2.0103e-02,
  3.5022e-01,  -3.1678e-02,  5.5305e-02,  1.2665e-01, -1.2807e-01,  1.6561e-01,  7.5635e-02, -5.2262e-03, -7.1193e-02, -1.2891e-01,
  5.1566e-01,   9.4354e-02, -1.7563e-01, -4.8556e-02, -5.3319e-02, -9.0733e-02,  1.8302e-02, -1.0020e-01,  1.4498e-01, -3.0288e-03,
  4.4535e-02,   1.0887e-01,  1.6025e-02,  9.3542e-02, -3.6787e-02, -1.4680e-01,  3.8237e-04, -8.9704e-02, -6.6145e-02,  8.7763e-02]
"""