"""Fused Muse Spark pre-expert norm + MXFP8 quant (gfx1250) against a one-row-per-program Triton RMSNorm followed by
aiter's downcast_to_mxfp: e4m3 rows and e8m0 scales bit for bit, and the dequantized rows close to an fp32 norm."""

import pytest
import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx1250_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gfx1250_supported(),
    reason="needs a gfx1250 GPU",
)

EPS = 1e-6


@triton.jit
def _ref_norm_kernel(
    X, Y, W, eps, H: tl.constexpr, BLOCK_H: tl.constexpr, HAS_WEIGHT: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H
    x = tl.load(X + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(tl.where(mask, x * x, 0.0))
    rstd = tl.rsqrt(sum_sq / H + eps)
    y = x * rstd
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w
    tl.store(Y + row * H + cols, y, mask=mask)


def _reference(x, w):
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

    m, h = x.shape
    y = torch.empty_like(x)
    _ref_norm_kernel[(m,)](
        x,
        y,
        w if w is not None else x,
        EPS,
        h,
        triton.next_power_of_2(h),
        w is not None,
        num_warps=8,
    )
    return downcast_to_mxfp(y, torch.float8_e4m3fn, axis=-1)


def _inputs(m, h, has_weight):
    g = torch.Generator(device="cuda").manual_seed(m * 7 + h)
    scale = torch.logspace(-3, 2, m, device="cuda")[:, None]
    x = (torch.randn(m, h, device="cuda", generator=g) * scale).bfloat16()
    x[0, :32] = 0  # an all-zero scale group
    w = (
        (1 + 0.5 * torch.randn(h, device="cuda", generator=g)).bfloat16()
        if has_weight
        else None
    )
    return x, w


@pytest.mark.parametrize("h", [4096, 96])
@pytest.mark.parametrize("m", [1, 2, 8, 33, 64, 512, 8192])
@pytest.mark.parametrize("has_weight", [True, False])
def test_bitwise_vs_two_kernels(m, h, has_weight):
    from sglang.srt.rocm_muse_spark.pre_expert_norm_mxfp8 import pre_expert_norm_mxfp8

    x, w = _inputs(m, h, has_weight)
    rq, rs = _reference(x, w)
    fq, fs = pre_expert_norm_mxfp8(x, w, EPS)
    assert torch.equal(fq.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(fs, rs.view(torch.uint8))


def test_dequant_close_to_fp32_norm():
    from sglang.srt.rocm_muse_spark.pre_expert_norm_mxfp8 import pre_expert_norm_mxfp8

    x, w = _inputs(64, 4096, True)
    q, s = pre_expert_norm_mxfp8(x, w, EPS)
    scale = torch.exp2(s.float() - 127).repeat_interleave(32, dim=1)
    deq = q.float() * scale
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
    # e4m3 rounding: half an ulp of 3 mantissa bits, or half the subnormal step 2^-9 times the group scale;
    # plus the bf16 rounding of the normed row
    bound = ref.abs() * (2.0**-4 + 2.0**-8) + scale * 2.0**-10
    assert ((deq - ref).abs() <= bound).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
