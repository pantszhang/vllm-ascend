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
"""CANN probe-fallback wrapper for the GDN causal conv1d (prefill fwd + decode update).

Mainline runs the self-developed AscendC op
``torch.ops._C_ascend.npu_causal_conv1d_custom``. This wrapper keeps that path as
the byte-identical fallback and opportunistically dispatches to a CANN fused
conv1d interface when one exists and passes a smoke test. Controlled by env
``VLLM_ASCEND_GDN_CANN_CONV1D`` = auto/on/off.
"""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

from vllm_ascend.ops._gdn_probe import probe_cann_interface

logger = init_logger(__name__)


def _conv1d_smoke_test(fn) -> None:
    """Minimal conv call: H=8, D=128, kernel width 4, bf16, one short sequence.

    The CANN conv1d torch_npu signature varies across versions; any signature
    mismatch or runtime error marks the interface unavailable (fall back).
    """
    device = torch.npu.current_device()
    seq, h, d, kw = 4, 8, 128, 4
    x = torch.zeros((seq, h, d), dtype=torch.bfloat16, device=device)
    w = torch.zeros((h, 1, kw), dtype=torch.bfloat16, device=device)
    fn(x, w)
    torch.npu.synchronize()


def causal_conv1d_run(
    output: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weights_T: torch.Tensor,
    *,
    conv_state: torch.Tensor,
    bias_opt: torch.Tensor | None,
    query_start_loc_opt: torch.Tensor | None,
    cache_indices_opt: torch.Tensor | None,
    initial_state_mode_opt,
    num_accepted_tokens_opt,
    activation_mode: int,
    pad_slot_id: int,
    run_mode: int,
) -> torch.Tensor:
    """Drop-in replacement for ``torch.ops._C_ascend.npu_causal_conv1d_custom``.

    Same signature so call sites only change the dispatch target. When a usable
    CANN conv1d interface exists it is tried first; any error falls back to the
    self-developed AscendC op.
    """
    probe = probe_cann_interface(
        env_var="VLLM_ASCEND_GDN_CANN_CONV1D",
        candidate_names=("npu_fused_causal_conv1d", "npu_causal_conv1d"),
        smoke_test=_conv1d_smoke_test,
    )
    if probe.available:
        try:
            # CANN call: same data, dispatch to the probed torch_npu interface.
            # The CANN op may return a new tensor; the contract with callers is
            # in-place into `output`, so copy the result back.
            res = probe.fn(mixed_qkv, conv_weights_T, bias_opt)
            output.copy_(res)
            return output
        except Exception:
            logger.warning_once(
                "CANN conv1d dispatch failed at runtime; falling back to the "
                "self-developed AscendC op (VLLM_ASCEND_GDN_CANN_CONV1D=%s)",
                os.environ.get("VLLM_ASCEND_GDN_CANN_CONV1D", "auto"),
            )
    return torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        mixed_qkv,
        conv_weights_T,
        conv_state=conv_state,
        bias_opt=bias_opt,
        query_start_loc_opt=query_start_loc_opt,
        cache_indices_opt=cache_indices_opt,
        initial_state_mode_opt=initial_state_mode_opt,
        num_accepted_tokens_opt=num_accepted_tokens_opt,
        activation_mode=activation_mode,
        pad_slot_id=pad_slot_id,
        run_mode=run_mode,
    )
