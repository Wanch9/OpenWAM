"""Guarded CUDA/Triton RMSNorm + FP64 RoPE, with an ordinary Torch fallback."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


if triton is not None:

    @triton.jit
    def _rms_scale_fwd(X, R, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
        scale = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS)
        tl.store(R + row, scale)

    @triton.jit
    def _norm_rope_fwd(
        X,
        W,
        F,
        R,
        Y,
        PAIRS: tl.constexpr,
        WIDTH: tl.constexpr,
        SEQ: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NORMALIZE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = pair < PAIRS
        row = pair // (WIDTH // 2)
        col = (pair % (WIDTH // 2)) * 2
        a = tl.load(X + pair * 2, valid, 0).to(tl.float32)
        b = tl.load(X + pair * 2 + 1, valid, 0).to(tl.float32)
        if NORMALIZE:
            scale = tl.load(R + row, valid, 0)
            a = (a * scale).to(X.dtype.element_ty).to(tl.float32)
            b = (b * scale).to(X.dtype.element_ty).to(tl.float32)
            wa = tl.load(W + col, valid, 0).to(tl.float32)
            wb = tl.load(W + col + 1, valid, 0).to(tl.float32)
            a = (a * wa).to(X.dtype.element_ty).to(tl.float32)
            b = (b * wb).to(X.dtype.element_ty).to(tl.float32)
        a, b = a.to(tl.float64), b.to(tl.float64)
        fpos = ((row % SEQ) * (HEAD_DIM // 2) + (col % HEAD_DIM) // 2) * 2
        real = tl.load(F + fpos, valid, 0).to(tl.float64)
        imag = tl.load(F + fpos + 1, valid, 0).to(tl.float64)
        # Torch converts FP64 through FP32 when storing BF16/FP16.
        even = (a * real - b * imag).to(tl.float32)
        odd = (a * imag + b * real).to(tl.float32)
        tl.store(Y + pair * 2, even, valid)
        tl.store(Y + pair * 2 + 1, odd, valid)


def torch_norm_rope(x, weight, freqs, eps, heads, normalize=True):
    if normalize:
        xf = x.float()
        x = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight
    batch, seq, width = x.shape
    xc = torch.view_as_complex(x.double().reshape(batch, seq, heads, -1, 2))
    fc = freqs.to(x.device).reshape(1, seq, 1, width // heads // 2)
    return torch.view_as_real(xc * fc).flatten(2).to(x.dtype)


@torch.library.custom_op("openwam_kernels::norm_rope", mutates_args=(), device_types="cuda")
def _norm_rope_op(
    x: torch.Tensor, weight: torch.Tensor, freqs_real: torch.Tensor, eps: float, heads: int, normalize: bool
) -> torch.Tensor:
    batch, seq, width = x.shape
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    scale = torch.empty((batch * seq,), device=x.device, dtype=torch.float32) if normalize else x
    if normalize:
        _rms_scale_fwd[(batch * seq,)](
            x,
            scale,
            width,
            eps,
            triton.next_power_of_2(width),
            num_warps=4 if width <= 1024 else 8,
            enable_fp_fusion=False,
        )
    pairs = batch * seq * width // 2
    _norm_rope_fwd[(triton.cdiv(pairs, 256),)](
        x,
        weight,
        freqs_real,
        scale,
        out,
        pairs,
        width,
        seq,
        width // heads,
        normalize,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@_norm_rope_op.register_fake
def _norm_rope_fake(x, weight, freqs_real, eps, heads, normalize):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def can_fuse(x, weight, freqs, heads):
    return (
        triton is not None
        and x.device.type == "cuda"
        and x.ndim == 3
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.is_contiguous()
        and 0 < x.shape[-1] <= 4096
        and heads > 0
        and x.shape[-1] % (2 * heads) == 0
        and weight.ndim == 1
        and weight.numel() == x.shape[-1]
        and weight.dtype == x.dtype
        and weight.device == x.device
        and weight.is_contiguous()
        and freqs.dtype in (torch.complex64, torch.complex128)
        and freqs.device == x.device
        and freqs.is_contiguous()
        and freqs.numel() == x.shape[1] * (x.shape[-1] // heads // 2)
        and x.shape[0] > 0
        and x.shape[1] > 0
        and not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad or freqs.requires_grad))
    )


def fused_norm_rope(x, weight, freqs, eps, heads, normalize=True):
    if not can_fuse(x, weight, freqs, heads):
        return torch_norm_rope(x, weight, freqs, eps, heads, normalize)
    return _norm_rope_op(x, weight, torch.view_as_real(freqs), float(eps), int(heads), bool(normalize))
