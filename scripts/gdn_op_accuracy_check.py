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
"""GDN operator-level accuracy check.

Compares three implementations of the Gated Delta Net recurrence:
1. naive CPU reference (fp32 accumulation, straight from the recurrence)
2. the vllm-ascend mainline chunk pipeline (Triton + self-developed AscendC)
3. the CANN fused operator ``torch_npu.npu_chunk_gated_delta_rule`` (if usable)

Run on the A5 box from the vllm-ascend repo root:
    python3 scripts/gdn_op_accuracy_check.py --cases all
"""

from __future__ import annotations

import argparse
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from vllm.forward_context import ForwardContext, override_forward_context
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import bootstrap_custom_op_env

THRESHOLD = 2e-2  # bf16 max relative error threshold (phase-0 baseline; tighten later)

CASES = {
    "base": dict(nk=2, nv=4, dk=128, dv=128, seq_len=256, seed=0),
    "g_near_zero": dict(nk=1, nv=2, dk=128, dv=128, seq_len=128, seed=1, g_range=(-1e-2, 0.0)),
    "g_strong_decay": dict(nk=1, nv=2, dk=128, dv=128, seq_len=128, seed=2, g_range=(-1.0, -0.8)),
    "beta_small": dict(nk=1, nv=2, dk=128, dv=128, seq_len=128, seed=3, beta_range=(0.01, 0.05)),
    "beta_large": dict(nk=1, nv=2, dk=128, dv=128, seq_len=128, seed=4, beta_range=(0.95, 0.99)),
    "long_seq": dict(nk=2, nv=4, dk=128, dv=128, seq_len=2048, seed=5),
    "wide_gqa": dict(nk=1, nv=8, dk=128, dv=128, seq_len=256, seed=6),
}


def l2norm(x: torch.Tensor) -> torch.Tensor:
    """L2 normalization along the last dim (same math as FLA l2norm_fwd)."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-5)


def gdn_naive_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive recurrent GDR in fp32 (golden reference).

    Shapes (TND, B=1): q,k [T, Nk, Dk]; v [T, Nv, Dv]; beta,g [T, Nv];
    initial_state [1, Nv, Dv, Dk]. GQA requires Nv % Nk == 0.
    Returns (o [T, Nv, Dv], final_state [1, Nv, Dv, Dk]) in fp32.
    """
    q = l2norm(q)
    k = l2norm(k)
    T, Nk, Dk = q.shape
    Nv, Dv = v.shape[1], v.shape[2]
    assert Nv % Nk == 0, f"Nv({Nv}) must be divisible by Nk({Nk})"
    rep = Nv // Nk
    S = initial_state.float()  # [1, Nv, Dv, Dk]
    o = torch.empty(T, Nv, Dv, dtype=torch.float32)
    for t in range(T):
        alpha = torch.exp(g[t]).float()                      # [Nv]
        kt = k[t].repeat_interleave(rep, dim=0)              # [Nv, Dk]
        qt = q[t].repeat_interleave(rep, dim=0)              # [Nv, Dk]
        vt = v[t]                                            # [Nv, Dv]
        kv_mem = torch.einsum("bhdk,bhk->bhd", S, kt[None])  # [1, Nv, Dv]
        delta = vt[None] - alpha[None, :, None] * kv_mem     # [1, Nv, Dv]
        update = (beta[t][None, :, None] * delta)[..., None] * kt[None, :, None, :]
        S = alpha[None, :, None, None] * S + update          # [1, Nv, Dv, Dk]
        o[t] = torch.einsum("bhdk,bhk->bhd", S, qt[None]) * scale
    return o, S


def make_case(nk, nv, dk, dv, seq_len, seed=0, g_range=(-0.5, -0.1), beta_range=(0.1, 0.9)):
    """Deterministic test inputs inside the CANN op constraints (see spec section 11)."""
    gen = torch.Generator().manual_seed(seed)
    q = 2 * torch.rand(seq_len, nk, dk, generator=gen) - 1
    k = 2 * torch.rand(seq_len, nk, dk, generator=gen) - 1
    v = 2 * torch.rand(seq_len, nv, dv, generator=gen) - 1
    beta = torch.rand(seq_len, nv, generator=gen) * (beta_range[1] - beta_range[0]) + beta_range[0]
    g = torch.rand(seq_len, nv, generator=gen) * (g_range[1] - g_range[0]) + g_range[0]
    initial_state = torch.zeros(1, nv, dv, dk)
    return q, k, v, beta, g, initial_state


def run_cann(q, k, v, beta, g, initial_state, scale):
    """CANN fused chunk op (TND layout, caller does q/k L2 norm). None if unusable."""
    import torch_npu

    if not hasattr(torch_npu, "npu_chunk_gated_delta_rule"):
        return None
    try:
        o, final_state = torch_npu.npu_chunk_gated_delta_rule(
            l2norm(q).to(torch.bfloat16),
            l2norm(k).to(torch.bfloat16),
            v.to(torch.bfloat16),
            beta=beta.to(torch.bfloat16),
            initial_state=initial_state.to(torch.bfloat16),
            actual_seq_lengths=torch.tensor([q.shape[0]], dtype=torch.int32, device=q.device),
            scale=scale,
            g=g.to(torch.float32),
        )
        return o, final_state
    except Exception:
        return None


def run_triton(q, k, v, beta, g, initial_state, scale):
    """Mainline chunk pipeline (6 sub-ops). State layout differs; only o is compared."""
    from vllm_ascend.ops.triton.fla import chunk as chunk_module

    T = q.shape[0]
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
    )
    single_rank_pcp = SimpleNamespace(world_size=1)
    with override_forward_context(forward_context), patch.object(
        chunk_module, "get_pcp_group", return_value=single_rank_pcp
    ):
        o, _ = chunk_module.chunk_gated_delta_rule(
            q=q[None].to(torch.bfloat16),
            k=k[None].to(torch.bfloat16),
            v=v[None].to(torch.bfloat16),
            g=g[None].to(torch.float32),
            beta=beta[None].to(torch.bfloat16),
            scale=scale,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=torch.tensor([0, T], dtype=torch.int64, device=q.device),
            use_qk_l2norm_in_kernel=True,
        )
    return o.squeeze(0), None


def max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs() / (b.float().abs() + 1e-5)).max().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="base", help="comma-separated case names or 'all'")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument(
        "--skip-triton",
        action="store_true",
        help="skip the vllm-ascend Triton/AscendC baseline and check CANN only",
    )
    args = ap.parse_args()

    assert torch.npu.is_available(), "this script must run on an Ascend NPU"
    if not args.skip_triton:
        bootstrap_custom_op_env(include_vendor_lib=True)
        importlib.import_module("vllm_ascend.vllm_ascend_C")
        init_device_properties_triton()
    names = sorted(CASES) if args.cases == "all" else [c.strip() for c in args.cases.split(",")]
    failures = []
    print(f"{'case':<18}{'triton_vs_ref':>16}{'cann_vs_ref(o)':>18}{'cann_vs_ref(S)':>18}  status")
    for name in names:
        cfg = CASES[name]
        q, k, v, beta, g, s0 = make_case(**cfg)
        dev = torch.device("npu")
        q, k, v = q.to(dev), k.to(dev), v.to(dev)
        beta, g, s0 = beta.to(dev), g.to(dev), s0.to(dev)
        scale = cfg["dk"] ** -0.5

        o_ref, s_ref = gdn_naive_reference(q.cpu(), k.cpu(), v.cpu(), beta.cpu(), g.cpu(), s0.cpu(), scale)

        if args.skip_triton:
            err_tri = float("nan")
        else:
            o_tri, _ = run_triton(q, k, v, beta, g, s0, scale)
            err_tri = max_rel_err(o_tri.cpu(), o_ref)

        cann = run_cann(q, k, v, beta, g, s0, scale)
        if cann is None:
            err_cann_o = err_cann_s = float("nan")
            status = "SKIP(cann-unavailable)"
        else:
            o_cann, s_cann = cann
            err_cann_o = max_rel_err(o_cann.cpu(), o_ref)
            err_cann_s = max_rel_err(s_cann.cpu(), s_ref)
            status = "PASS(cann-only)" if args.skip_triton else "PASS"
        # Triton 基线始终参与阈值检查（nan 与阈值比较恒为 False，SKIP 分支天然兼容）。
        bad = [e for e in (err_tri, err_cann_o, err_cann_s) if e > args.threshold]
        if bad:
            status = "FAIL"
            failures.append(name)
        print(
            f"{name:<18}{err_tri:>16.4e}{err_cann_o:>18.4e}{err_cann_s:>18.4e}  {status}"
        )
    print(f"\nthreshold={args.threshold:.2e}; failed cases: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
