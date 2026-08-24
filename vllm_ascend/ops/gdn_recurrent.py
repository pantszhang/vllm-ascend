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
"""CANN probe-fallback wrapper for the decode recurrent GDN core.

Mainline runs the self-developed AscendC op
``torch.ops._C_ascend.npu_recurrent_gated_delta_rule`` (csrc, NOT the built-in
CANN operator). This wrapper keeps that path as the fallback and
opportunistically dispatches to the CANN ``recurrent_gated_delta_rule`` through
``torch_npu.npu_recurrent_gated_delta_rule`` when usable. Controlled by env
``VLLM_ASCEND_GDN_CANN_RECURRENT`` = auto/on/off.
"""

from __future__ import annotations

import torch

from vllm_ascend.ops._gdn_probe import probe_cann_interface


def _recurrent_smoke_test(fn) -> None:
    """Minimal single-step decode call: Dk == Dv == 128, Nk == Nv == 1.

    State is bf16 here because the CANN op only supports a bf16 state, while
    the mainline custom op keeps fp32 state - dtype compatibility across the
    prefill->decode boundary is validated separately at model level.
    """
    device = torch.npu.current_device()
    dk = dv = 128
    query = torch.zeros((1, 1, dk), dtype=torch.bfloat16, device=device)
    key = torch.zeros((1, 1, dk), dtype=torch.bfloat16, device=device)
    value = torch.zeros((1, 1, dv), dtype=torch.bfloat16, device=device)
    g = torch.full((1, 1), -0.1, dtype=torch.float32, device=device)
    beta = torch.full((1, 1), 0.5, dtype=torch.bfloat16, device=device)
    state = torch.zeros((1, 1, dv, dk), dtype=torch.bfloat16, device=device)
    actual_seq_lengths = torch.tensor([1], dtype=torch.int32, device=device)
    ssm_state_indices = torch.zeros(1, dtype=torch.int64, device=device)
    fn(
        query,
        key,
        value,
        g=g,
        beta=beta,
        state=state,
        scale=dk**-0.5,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
    )
    torch.npu.synchronize()


def recurrent_gated_delta_rule_run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for ``torch.ops._C_ascend.npu_recurrent_gated_delta_rule``.

    Same signature so call sites only change the dispatch target.
    """
    probe = probe_cann_interface(
        env_var="VLLM_ASCEND_GDN_CANN_RECURRENT",
        candidate_names=("npu_recurrent_gated_delta_rule",),
        smoke_test=_recurrent_smoke_test,
    )
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        state=state,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
    )
    if num_accepted_tokens is not None:
        kwargs["num_accepted_tokens"] = num_accepted_tokens
    if probe.available:
        try:
            return probe.fn(**kwargs)
        except Exception:
            pass  # fall through to the mainline custom op
    return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(**kwargs)
