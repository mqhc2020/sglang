"""CUDA-graph timing of the fused Muse Spark pre-expert norm + MXFP8 quant (gfx1250) against a Triton RMSNorm followed
by aiter's downcast_to_mxfp, at H = 4096 over decode and prefill row counts.

    python test/manual/kernels/bench_pre_expert_norm_mxfp8.py
"""

import torch
from test_pre_expert_norm_mxfp8 import EPS, _inputs, _reference

from sglang.srt.rocm_muse_spark.pre_expert_norm_mxfp8 import pre_expert_norm_mxfp8

H = 4096
ROWS = [1, 2, 4, 8, 16, 32, 48, 64, 128, 512, 2048, 8192]


def graph_us(fn, iters=50, replays=10):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        for _ in range(iters):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    times = []
    for _ in range(replays):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iters)
    return sorted(times)[len(times) // 2]


def main():
    print(f"{'rows':>5} {'two kernels us':>15} {'fused us':>9} {'saved us':>9}")
    for m in ROWS:
        x, w = _inputs(m, H, True)
        t_ref = graph_us(lambda: _reference(x, w))
        t_fused = graph_us(lambda: pre_expert_norm_mxfp8(x, w, EPS))
        print(f"{m:5d} {t_ref:15.2f} {t_fused:9.2f} {t_ref - t_fused:9.2f}", flush=True)


if __name__ == "__main__":
    main()
