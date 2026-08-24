#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GDN operator-level profiling (phase-0 baseline).

Measures prefill (chunk pipeline vs CANN fused op) and decode (recurrent step)
latencies with realistic Qwen3.5 shapes, so the phase-2 replacement order can
be decided by measured hot spots. Run on the A5 box from the vllm-ascend root:

    python3 scripts/gdn_op_profiling.py
"""

from __future__ import annotations

import time

import torch

from scripts.gdn_op_accuracy_check import l2norm, make_case, run_cann, run_triton

PREFILL = dict(nk=8, nv=8, dk=128, dv=128, seq_len=4096, seed=42)
DECODE = dict(nk=8, nv=8, dk=128, dv=128, seq_len=1, seed=42)


def timeit(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def bench_prefill():
    q, k, v, beta, g, s0 = make_case(**PREFILL)
    dev = torch.device("npu")
    q, k, v = q.to(dev), k.to(dev), v.to(dev)
    beta, g, s0 = beta.to(dev), g.to(dev), s0.to(dev)
    scale = PREFILL["dk"] ** -0.5

    t_triton = timeit(lambda: run_triton(q, k, v, beta, g, s0, scale))
    cann = run_cann(q, k, v, beta, g, s0, scale)
    t_cann = timeit(lambda: run_cann(q, k, v, beta, g, s0, scale)) if cann is not None else float("nan")
    print(f"prefill  T={PREFILL['seq_len']} Nk={PREFILL['nk']} Nv={PREFILL['nv']}: "
          f"triton_pipeline={t_triton:.3f}ms  cann_fused={t_cann:.3f}ms")


def bench_decode():
    """Single decode step: self-developed AscendC recurrent op (mainline)."""
    q, k, v, beta, g, s0 = make_case(**DECODE)
    dev = torch.device("npu")
    q, k, v = q.to(dev), k.to(dev), v.to(dev)
    beta, g, s0 = beta.to(dev), g.to(dev), s0.to(dev)
    scale = DECODE["dk"] ** -0.5
    nv, dv, dk = s0.shape[1], s0.shape[2], q.shape[2]
    state = torch.zeros(1, nv, dv, dk, dtype=torch.float32, device=dev)
    actual_seq_lengths = torch.tensor([1], dtype=torch.int32, device=dev)
    ssm_state_indices = torch.zeros(1, dtype=torch.int64, device=dev)

    def step():
        torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=l2norm(q).to(torch.bfloat16),
            key=l2norm(k).to(torch.bfloat16),
            value=v.to(torch.bfloat16),
            g=g.to(torch.float32),
            beta=beta.to(torch.bfloat16),
            state=state,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices,
        )

    t_rec = timeit(step, warmup=3, iters=100)
    print(f"decode   T=1 Nk={DECODE['nk']} Nv={DECODE['nv']}: "
          f"recurrent_ascendc={t_rec * 1e3:.1f}us/step")


if __name__ == "__main__":
    bench_prefill()
    bench_decode()
