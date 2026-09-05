# SPDX-License-Identifier: Apache-2.0
"""Real-NPU comparison for the Phase6-only FLA GDN prefill backend."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.device.device_config import get_fla_gdn_soc, is_fla_gdn_supported
from vllm_ascend.ops.gdn_fla import (
    FlaGDNPhase6Backend,
    GDNPhase6PrefillMetadata,
    GDNPhase6RuntimeSignature,
)
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states, prepare_chunk_indices

pytestmark = pytest.mark.skipif(
    not is_fla_gdn_supported(), reason="requires A2/A3/A5 FLA GDN support"
)
torch_npu.npu.set_compile_mode(jit_compile=False)

_KEY_DIM = 128
_VALUE_DIM = 128
_KEY_HEADS = 1
_VALUE_HEADS = 2
_CHUNK_SIZE = 64


def _metadata(cu_seqlens_host: tuple[int, ...]) -> GDNPhase6PrefillMetadata:
    cu_seqlens = torch.tensor(cu_seqlens_host, dtype=torch.int64)
    chunk_indices = prepare_chunk_indices(cu_seqlens, _CHUNK_SIZE)
    return GDNPhase6PrefillMetadata(
        cu_seqlens_host=cu_seqlens_host,
        chunk_indices_host=tuple(int(value) for value in chunk_indices.flatten().tolist()),
    )


def _prefill_inputs(cu_seqlens_host: tuple[int, ...]):
    tokens = cu_seqlens_host[-1]
    sequences = len(cu_seqlens_host) - 1
    torch.manual_seed(7)
    q = torch.randn((1, tokens, _KEY_HEADS, _KEY_DIM), dtype=torch.bfloat16, device="npu")
    return {
        "q": q,
        "k": torch.randn_like(q),
        "v": torch.randn(
            (1, tokens, _VALUE_HEADS, _VALUE_DIM),
            dtype=torch.bfloat16,
            device="npu",
        ),
        "g": F.logsigmoid(
            torch.randn((1, tokens, _VALUE_HEADS), dtype=torch.float32, device="npu")
        ),
        "beta": torch.sigmoid(
            torch.randn((1, tokens, _VALUE_HEADS), dtype=torch.bfloat16, device="npu")
        ),
        "initial_state": torch.randn(
            (sequences, _VALUE_HEADS, _VALUE_DIM, _KEY_DIM),
            dtype=torch.float32,
            device="npu",
        ),
        "has_initial_state": torch.tensor(
            [index % 2 == 1 for index in range(sequences)],
            dtype=torch.bool,
            device="npu",
        ),
        "scale": _KEY_DIM**-0.5,
        "metadata": _metadata(cu_seqlens_host),
    }


def _native_prefill(
    *,
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    has_initial_state,
    scale,
    metadata,
):
    del scale
    native_state = initial_state.transpose(-1, -2).contiguous()
    clear_ssm_states(native_state, has_initial_state)
    cu_seqlens = torch.tensor(
        metadata.cu_seqlens_host,
        dtype=torch.int64,
        device=q.device,
    )
    output, final_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=native_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        prebuilt_meta=None,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    return output, final_state.transpose(-1, -2).contiguous()


def _phase6_backend() -> FlaGDNPhase6Backend:
    soc = get_fla_gdn_soc()
    assert soc is not None
    backend = FlaGDNPhase6Backend.create(
        mode="fla_npu",
        signature=GDNPhase6RuntimeSignature(
            soc=soc,
            dtype="bfloat16",
            state_dtype="float32",
            num_key_heads=_KEY_HEADS,
            num_value_heads=_VALUE_HEADS,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
            chunk_size=_CHUNK_SIZE,
        ),
        layer_name="smoke.linear_attn",
        pcp_world_size=1,
    )
    assert backend is not None
    return backend


def _assert_phase6_matches_native(cu_seqlens_host: tuple[int, ...]) -> None:
    inputs = _prefill_inputs(cu_seqlens_host)
    expected_output, expected_state = _native_prefill(**inputs)
    backend = _phase6_backend()
    assert backend.prepare(inputs["q"].device)
    actual_output, actual_state = backend.prefill(**inputs)
    torch.npu.synchronize()

    assert torch.isfinite(actual_output.float()).all()
    assert torch.isfinite(actual_state.float()).all()
    cosine = F.cosine_similarity(
        actual_output.float().flatten(),
        expected_output.float().flatten(),
        dim=0,
    )
    assert cosine.item() >= 0.999
    torch.testing.assert_close(actual_output, expected_output, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(actual_state, expected_state, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("tokens", [1, 63, 64, 65])
def test_gdn_phase6_prefill_matches_native(tokens):
    _assert_phase6_matches_native((0, tokens))


def test_gdn_phase6_varlen_prefill_matches_native():
    _assert_phase6_matches_native((0, 1, 65))
