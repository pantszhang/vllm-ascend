# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the A5 FLA GDN prefill backend."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_fla import (
    FlaGDNPrefillBackend,
    GDNBackendMode,
    GDNPrefillMetadata,
    GDNRuntimeSignature,
    parse_gdn_backend_mode,
)


SIGNATURE = GDNRuntimeSignature(
    soc="ascend950",
    dtype="bfloat16",
    state_dtype="float32",
    num_key_heads=1,
    num_value_heads=2,
    key_dim=128,
    value_dim=128,
    chunk_size=64,
)


@pytest.fixture(autouse=True)
def clear_gdn_fla_process_caches():
    FlaGDNPrefillBackend.clear_process_caches_for_test()
    yield
    FlaGDNPrefillBackend.clear_process_caches_for_test()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", GDNBackendMode.AUTO),
        ("AUTO", GDNBackendMode.AUTO),
        ("fla_npu", GDNBackendMode.FLA_NPU),
        ("native", GDNBackendMode.NATIVE),
    ],
)
def test_parse_gdn_backend_mode(value, expected):
    assert parse_gdn_backend_mode(value) is expected


@pytest.mark.parametrize("value", ["", "fla", "ascend", "native,fla_npu"])
def test_parse_gdn_backend_mode_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="GDN backend mode"):
        parse_gdn_backend_mode(value)


def _fake_fla_outputs(tokens=64, sequences=1):
    return (
        torch.zeros((1, tokens, 2, 128), dtype=torch.bfloat16),
        torch.zeros((sequences, 2, 128, 128), dtype=torch.float32),
        torch.zeros((1, tokens, 2), dtype=torch.float32),
        torch.zeros((1, 2, tokens, 64), dtype=torch.bfloat16),
    )


def _backend(mode):
    backend = FlaGDNPrefillBackend.create(
        mode=mode,
        signature=SIGNATURE,
        layer_name="model.layers.0.linear_attn",
        pcp_world_size=1,
    )
    assert backend is not None
    return backend


def _prefill_inputs(tokens=65):
    return {
        "q": torch.zeros((1, tokens, 1, 128), dtype=torch.bfloat16),
        "k": torch.zeros((1, tokens, 1, 128), dtype=torch.bfloat16),
        "v": torch.zeros((1, tokens, 2, 128), dtype=torch.bfloat16),
        "g": torch.zeros((1, tokens, 2), dtype=torch.float32),
        "beta": torch.full((1, tokens, 2), 0.5, dtype=torch.bfloat16),
        "initial_state": torch.zeros((2, 2, 128, 128), dtype=torch.float32),
        "has_initial_state": torch.tensor([False, True]),
        "scale": 128**-0.5,
        "metadata": GDNPrefillMetadata(
            cu_seqlens_host=(0, 1, tokens),
            chunk_indices_host=(0, 0, 1, 0),
        ),
    }


def test_fla_backend_accepts_a5():
    backend = FlaGDNPrefillBackend.create(
        mode="auto",
        signature=SIGNATURE,
        layer_name="model.layers.0.linear_attn",
        pcp_world_size=1,
    )
    assert backend is not None


@pytest.mark.parametrize("soc", ["ascend910b", "ascend910_93"])
def test_fla_backend_auto_uses_native_for_a2_a3(soc):
    signature = dataclasses.replace(SIGNATURE, soc=soc)
    assert (
        FlaGDNPrefillBackend.create(
            mode="auto",
            signature=signature,
            layer_name="model.layers.0.linear_attn",
            pcp_world_size=1,
        )
        is None
    )


@pytest.mark.parametrize("soc", ["ascend910b", "ascend910_93"])
def test_fla_backend_strict_rejects_a2_a3(soc):
    signature = dataclasses.replace(SIGNATURE, soc=soc)
    with pytest.raises(RuntimeError, match="unsupported SoC"):
        FlaGDNPrefillBackend.create(
            mode="fla_npu",
            signature=signature,
            layer_name="model.layers.0.linear_attn",
            pcp_world_size=1,
        )


@pytest.mark.parametrize(
    "signature",
    [
        dataclasses.replace(SIGNATURE, value_dim=256),
        dataclasses.replace(SIGNATURE, num_value_heads=5),
        dataclasses.replace(SIGNATURE, chunk_size=128),
    ],
    ids=["value-dim", "gva-ratio", "chunk-size"],
)
def test_fla_backend_auto_uses_native_outside_pr472_contract(signature):
    assert (
        FlaGDNPrefillBackend.create(
            mode="auto",
            signature=signature,
            layer_name="model.layers.0.linear_attn",
            pcp_world_size=1,
        )
        is None
    )


def test_fla_backend_native_does_not_resolve_fla(monkeypatch):
    resolver = MagicMock()
    monkeypatch.setattr("vllm_ascend.ops.gdn_fla.resolve_chunk_gated_delta_rule_fwd", resolver)
    backend = FlaGDNPrefillBackend.create(
        mode="native",
        signature=SIGNATURE,
        layer_name="model.layers.0.linear_attn",
        pcp_world_size=1,
    )
    assert backend is None
    resolver.assert_not_called()


def test_fla_backend_auto_uses_native_for_pcp():
    with patch("vllm_ascend.ops.gdn_fla.logger.info") as info:
        backend = FlaGDNPrefillBackend.create(
            mode="auto",
            signature=SIGNATURE,
            layer_name="model.layers.0.linear_attn",
            pcp_world_size=2,
        )
    assert backend is None
    assert "pcp_world_size=2" in str(info.call_args)


def test_fla_backend_strict_rejects_pcp():
    with pytest.raises(RuntimeError, match="PCP world size 1"):
        FlaGDNPrefillBackend.create(
            mode="fla_npu",
            signature=SIGNATURE,
            layer_name="model.layers.0.linear_attn",
            pcp_world_size=2,
        )


def test_auto_resolve_failure_returns_native(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.resolve_chunk_gated_delta_rule_fwd",
        MagicMock(side_effect=ImportError("missing fla_npu")),
    )
    assert _backend("auto").prepare(torch.device("cpu")) is False


def test_strict_resolve_failure_raises(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.resolve_chunk_gated_delta_rule_fwd",
        MagicMock(side_effect=ImportError("missing fla_npu")),
    )
    with pytest.raises(RuntimeError, match="resolve"):
        _backend("fla_npu").prepare(torch.device("cpu"))


def test_scratch_probe_runs_once_per_device_and_signature(monkeypatch):
    raw = MagicMock(return_value=_fake_fla_outputs())
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.resolve_chunk_gated_delta_rule_fwd",
        lambda: (raw, "fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd"),
    )
    first = _backend("auto")
    second = _backend("auto")
    assert first.prepare(torch.device("cpu")) is True
    assert second.prepare(torch.device("cpu")) is True
    assert raw.call_count == 1


def test_prefill_uses_a5_prepare_pipeline_contract(monkeypatch):
    captured = {}

    def fake_fla(q, k, v, g, beta, **kwargs):
        captured.update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=kwargs["initial_state"],
            cu_seqlens=kwargs["cu_seqlens"],
            chunk_indices=kwargs["chunk_indices"],
            options=kwargs,
        )
        return (
            torch.zeros((q.shape[0], q.shape[1], v.shape[2], v.shape[3]), dtype=q.dtype),
            torch.zeros_like(kwargs["initial_state"]),
            torch.zeros_like(g),
            torch.zeros((q.shape[0], v.shape[2], q.shape[1], 64), dtype=q.dtype),
        )

    raw = MagicMock(side_effect=fake_fla)
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.resolve_chunk_gated_delta_rule_fwd",
        lambda: (raw, "fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd"),
    )
    backend = _backend("fla_npu")
    assert backend.prepare(torch.device("cpu")) is True
    captured.clear()
    output, final_state = backend.prefill(**_prefill_inputs())

    assert captured["q"].shape == (1, 65, 1, 128)
    assert captured["k"].shape == (1, 65, 1, 128)
    assert captured["v"].shape == (1, 65, 2, 128)
    assert captured["beta"].dtype == torch.bfloat16
    assert captured["initial_state"].shape == (2, 2, 128, 128)
    assert captured["cu_seqlens"] == (0, 1, 65)
    assert captured["chunk_indices"] == (0, 0, 1, 0)
    assert captured["options"]["use_exp2"] is True
    assert captured["options"]["use_qk_l2norm_in_kernel"] is True
    assert captured["options"]["use_gate_in_kernel"] is False
    assert captured["options"]["use_beta_sigmoid_in_kernel"] is False
    assert captured["options"]["allow_neg_eigval"] is False
    assert captured["options"]["output_a"] is False
    assert captured["options"]["state_v_first"] is True
    assert captured["options"]["layout"] == "BSND"
    assert output.shape == (1, 65, 2, 128)
    assert final_state.shape == (2, 2, 128, 128)


def test_live_fla_failure_is_logged_and_propagated(monkeypatch):
    raw = MagicMock(side_effect=[_fake_fla_outputs(), RuntimeError("device execution failed")])
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.resolve_chunk_gated_delta_rule_fwd",
        lambda: (raw, "fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd"),
    )
    backend = _backend("auto")
    assert backend.prepare(torch.device("cpu")) is True

    with patch("vllm_ascend.ops.gdn_fla.logger.exception") as error_log:
        with pytest.raises(RuntimeError, match="device execution failed"):
            backend.prefill(**_prefill_inputs())
    error_log.assert_called_once()
    assert raw.call_count == 2


def test_mixed_decode_prefill_uses_fla_only_for_prefill():
    conv_state = torch.zeros((3, 1, 2))
    ssm_state = torch.zeros((2, 1, 2, 2))

    def rearrange_mixed_qkv(value):
        if value is None:
            return None, None, None
        projected = value.reshape(1, value.shape[0], 1, 2)
        return projected, projected, projected

    layer = SimpleNamespace(
        prefix="layers.0.linear_attn",
        kv_cache=(conv_state, ssm_state),
        conv1d=SimpleNamespace(weight=torch.zeros((2, 1, 2)), bias=None),
        activation=None,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        rearrange_mixed_qkv=rearrange_mixed_qkv,
    )
    chunk_meta = SimpleNamespace(
        cu_seqlens_host=(0, 2),
        chunk_indices_chunk64_host=(0, 0),
    )
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=3,
        non_spec_query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([0, 1], dtype=torch.int32),
        prefill_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        prefill_state_indices=torch.tensor([1], dtype=torch.int64),
        prefill_has_initial_state=torch.tensor([True]),
    )
    metadata.non_spec_prefill_metadata = SimpleNamespace(
        causal_conv1d=SimpleNamespace(
            query_start_loc=metadata.non_spec_query_start_loc,
            cache_indices=torch.tensor([0, 1], dtype=torch.int32),
            initial_state_mode=torch.tensor([True, True]),
        ),
        chunk=chunk_meta,
    )
    metadata.non_spec_decode_metadata = SimpleNamespace(
        actual_seq_lengths=torch.tensor([0, 1], dtype=torch.int32),
    )
    fla_backend = SimpleNamespace(
        prepare=MagicMock(return_value=True),
        prefill=MagicMock(
            side_effect=lambda **kwargs: (
                torch.full_like(kwargs["v"], 20),
                kwargs["initial_state"] + 2,
            )
        ),
    )

    def native_causal_conv(output, value, *_args, **_kwargs):
        output.copy_(value)

    def native_recurrent(**kwargs):
        return torch.full_like(kwargs["value"], 10)

    forward_context = ForwardContext(
        no_compile_layers={layer.prefix: layer},
        attn_metadata={layer.prefix: metadata},
        slot_mapping={},
    )
    core_attn_out = torch.empty((3, 1, 2))
    gating = (torch.zeros((1, 3, 1)), torch.zeros((1, 3, 1)))

    with (
        override_forward_context(forward_context),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch.object(
            AscendGatedDeltaNetAttention,
            "_get_fla_gdn_prefill_backend",
            return_value=fla_backend,
        ) as get_backend,
        patch(
            "vllm_ascend.ops.gdn.torch.ops._C_ascend.npu_causal_conv1d_custom",
            side_effect=native_causal_conv,
        ) as native_conv,
        patch(
            "vllm_ascend.ops.gdn.torch.ops._C_ascend.npu_recurrent_gated_delta_rule",
            side_effect=native_recurrent,
        ) as native_decode,
        patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=lambda tensor: tensor),
        patch("vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating", return_value=gating),
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector"),
        patch("vllm_ascend.ops.gdn.wait_for_kv_layer_from_connector"),
        patch("vllm_ascend.ops.gdn.record_attention_compute_start"),
    ):
        AscendGatedDeltaNetAttention._forward_core(
            layer,
            torch.zeros((3, 2)),
            torch.zeros((3, 1)),
            torch.zeros((3, 1)),
            core_attn_out,
        )

    get_backend.assert_called_once()
    fla_backend.prepare.assert_called_once()
    fla_backend.prefill.assert_called_once()
    native_conv.assert_called_once()
    native_decode.assert_called_once()
    torch.testing.assert_close(core_attn_out[0], torch.full_like(core_attn_out[0], 10))
    torch.testing.assert_close(core_attn_out[1:], torch.full_like(core_attn_out[1:], 20))


@pytest.mark.parametrize("speculative", [False, True], ids=["ordinary-decode", "mtp-decode"])
def test_pure_decode_never_constructs_fla_prefill_backend(speculative):
    layer = SimpleNamespace(
        prefix="layers.0.linear_attn",
        kv_cache=(torch.zeros((1, 1, 2)), torch.zeros((1, 1, 2, 2))),
        conv1d=SimpleNamespace(weight=torch.zeros((2, 1, 2)), bias=None),
        activation=None,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        rearrange_mixed_qkv=lambda value: (
            (None, None, None)
            if value is None
            else (value.reshape(1, value.shape[0], 1, 2),) * 3
        ),
    )
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0 if speculative else 1,
        num_decode_tokens=0 if speculative else 1,
        num_spec_decodes=1 if speculative else 0,
        num_spec_decode_tokens=1 if speculative else 0,
        num_actual_tokens=1,
        non_spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32),
    )
    decode_conv = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        cache_indices=torch.tensor([0], dtype=torch.int32),
    )
    if speculative:
        decode_conv.num_accepted_tokens = torch.tensor([1], dtype=torch.int32)
        metadata.spec_sequence_masks = torch.tensor([True])
        metadata.spec_token_indx = torch.tensor([0], dtype=torch.int64)
        metadata.non_spec_token_indx = torch.empty(0, dtype=torch.int64)
        metadata.spec_state_indices_tensor = torch.tensor([[0]], dtype=torch.int32)
        metadata.spec_decode_metadata = SimpleNamespace(
            actual_seq_lengths=torch.tensor([1], dtype=torch.int32),
            spec_causal_conv1d=decode_conv,
        )
    else:
        metadata.non_spec_decode_metadata = SimpleNamespace(
            actual_seq_lengths=torch.tensor([0, 1], dtype=torch.int32),
            causal_conv1d=decode_conv,
        )

    def native_causal_conv(output, value, *_args, **_kwargs):
        output.copy_(value)

    def native_recurrent(**kwargs):
        return torch.full_like(kwargs["value"], 7)

    forward_context = ForwardContext(
        no_compile_layers={layer.prefix: layer},
        attn_metadata={layer.prefix: metadata},
        slot_mapping={},
    )
    get_backend = MagicMock()
    core_attn_out = torch.empty((1, 1, 2))
    gating = (torch.zeros((1, 1, 1)), torch.zeros((1, 1, 1)))

    with (
        override_forward_context(forward_context),
        patch.object(
            AscendGatedDeltaNetAttention,
            "_get_fla_gdn_prefill_backend",
            get_backend,
        ),
        patch(
            "vllm_ascend.ops.gdn.torch.ops._C_ascend.npu_causal_conv1d_custom",
            side_effect=native_causal_conv,
        ) as native_conv,
        patch(
            "vllm_ascend.ops.gdn.torch.ops._C_ascend.npu_recurrent_gated_delta_rule",
            side_effect=native_recurrent,
        ) as native_decode,
        patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=lambda tensor: tensor),
        patch("vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating", return_value=gating),
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector"),
        patch("vllm_ascend.ops.gdn.wait_for_kv_layer_from_connector"),
        patch("vllm_ascend.ops.gdn.record_attention_compute_start"),
    ):
        AscendGatedDeltaNetAttention._forward_core(
            layer,
            torch.zeros((1, 2)),
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            core_attn_out,
        )

    get_backend.assert_not_called()
    native_conv.assert_called_once()
    native_decode.assert_called_once()
    torch.testing.assert_close(core_attn_out, torch.full_like(core_attn_out, 7))
