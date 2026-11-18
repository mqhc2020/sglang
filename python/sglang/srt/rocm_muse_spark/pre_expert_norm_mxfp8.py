"""Muse Spark pre-expert RMSNorm + per-32 MXFP8 quantization of the expert input in one launch (gfx1250).

For a W4A8 expert backend, which quantizes the pre-expert-normed rows with aiter's
``downcast_to_mxfp(x, float8_e4m3fn, axis=-1)``. The norm runs one row per program (one block, eight warps, the
whole row in registers), is rounded to bf16 where a separate norm kernel would store it, and is quantized with
aiter's own ``_compute_mx_quant_and_scale`` (amax / 448, e8m0 scale rounded up). Both outputs therefore match a
one-row-per-program Triton norm followed by ``downcast_to_mxfp`` bit for bit; the bf16 normed row is not written.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.moe.quant_moe import _compute_mx_quant_and_scale

__all__ = ["pre_expert_norm_mxfp8", "supports"]


@triton.jit
def _pre_expert_norm_mxfp8_kernel(
    X,
    W,
    XQ,
    XS,
    eps,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    x = tl.load(X + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(tl.where(mask, x * x, 0.0))
    rstd = tl.rsqrt(sum_sq / H + eps)
    y = x * rstd
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w
    y = y.to(tl.bfloat16)

    q, s = _compute_mx_quant_and_scale(
        tl.reshape(y, [1, BLOCK_H]),
        tl.reshape(mask, [1, BLOCK_H]),
        XQ.dtype.element_ty,
        0,
        False,
    )
    tl.store(XQ + row * H + cols[None, :], q, mask=mask[None, :])
    scols = tl.arange(0, BLOCK_H // 32)
    tl.store(XS + row * (H // 32) + scols[None, :], s, mask=(scols < H // 32)[None, :])


def supports(x: torch.Tensor, weight: torch.Tensor | None) -> bool:
    return (
        x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.is_contiguous()
        and x.shape[0] > 0
        and x.shape[1] % 32 == 0
        and (weight is None or weight.dtype == torch.bfloat16)
    )


def pre_expert_norm_mxfp8(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(e4m3 [M, H], e8m0 scales as uint8 [M, H/32]) of the RMSNorm of ``x`` (bf16 [M, H]) scaled by ``weight``
    (the effective bf16 [H] weight, or None for no weight)."""
    assert supports(x, weight)
    m, h = x.shape
    xq = torch.empty((m, h), dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty((m, h // 32), dtype=torch.uint8, device=x.device)
    _pre_expert_norm_mxfp8_kernel[(m,)](
        x, weight if weight is not None else x, xq, xs, eps, h, triton.next_power_of_2(h), weight is not None,
        num_warps=8,
    )  # fmt: skip
    return xq, xs
