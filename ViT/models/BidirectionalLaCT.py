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
    # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
    q_dtype = q.dtype
    k_dtype = k.dtype
    sin, cos = rope
    rope_dtype = sin.dtype
    q = q.to(dtype=rope_dtype)
    k = k.to(dtype=rope_dtype)
    N = q.shape[-2]
    prefix = N - sin.shape[-2]
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
    w1_norm = vector_norm(w1, dim=2, keepdim=True)

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
    dhidden = torch.bmm(w1.transpose(1, 2), v)

    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

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

# @torch.compile # Commented out for debugging
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
    _test_layer()