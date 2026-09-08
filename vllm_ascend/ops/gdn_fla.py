# SPDX-License-Identifier: Apache-2.0
"""A5 FLA backend for Qwen GDN prefill on Ascend."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, ClassVar, Self

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

GDN_FWD_SYMBOL = "fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd"
_GDN_FWD_MODULE = "fla_npu.ops.ascendc"
_GDN_FWD_ATTRIBUTE = "chunk_gated_delta_rule_fwd"
_SUPPORTED_SOCS = frozenset({"ascend950"})
_SUPPORTED_STATE_DTYPES = frozenset({"bfloat16", "float32"})
_GDN_ACTIVATION_DTYPE = "bfloat16"
_GDN_KEY_DIM = 128
_GDN_VALUE_DIM = 128
_GDN_CHUNK_SIZE = 64
_MAX_GVA_RATIO = 4
_SCRATCH_TOKENS = 64


class GDNBackendMode(StrEnum):
    AUTO = "auto"
    FLA_NPU = "fla_npu"
    NATIVE = "native"


def parse_gdn_backend_mode(value: str) -> GDNBackendMode:
    try:
        return GDNBackendMode(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in GDNBackendMode)
        raise ValueError(f"Invalid GDN backend mode {value!r}; expected one of: {valid}.") from exc


@dataclass(frozen=True)
class GDNRuntimeSignature:
    soc: str
    dtype: str
    state_dtype: str
    num_key_heads: int
    num_value_heads: int
    key_dim: int
    value_dim: int
    chunk_size: int = _GDN_CHUNK_SIZE


@dataclass(frozen=True)
class GDNPrefillMetadata:
    cu_seqlens_host: tuple[int, ...]
    chunk_indices_host: tuple[int, ...]


@dataclass(frozen=True)
class _PreparationResult:
    ready: bool
    operator: Callable[..., Any] | None = None
    symbol: str | None = None
    stage: str | None = None
    exception_type: str | None = None
    reason: str | None = None


def resolve_chunk_gated_delta_rule_fwd() -> tuple[Callable[..., Any], str]:
    module = importlib.import_module(_GDN_FWD_MODULE)
    return getattr(module, _GDN_FWD_ATTRIBUTE), GDN_FWD_SYMBOL


def _first_line(exc: BaseException) -> str:
    return str(exc).splitlines()[0] if str(exc) else repr(exc)


def _tensor_metadata(name: str, value: Any) -> str:
    if not isinstance(value, torch.Tensor):
        return f"{name}={type(value).__name__}"
    return (
        f"{name}(shape={tuple(value.shape)},dtype={value.dtype},device={value.device},"
        f"stride={value.stride()},contiguous={value.is_contiguous()})"
    )


def _tensor_call_metadata(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    metadata = [_tensor_metadata(f"arg{index}", value) for index, value in enumerate(args)]
    for name, value in kwargs.items():
        if isinstance(value, (list, tuple)):
            metadata.append(f"{name}(type={type(value).__name__},len={len(value)})")
        else:
            metadata.append(_tensor_metadata(name, value))
    return "; ".join(metadata)


class FlaGDNPrefillBackend:
    """Validated access to the FLA A5 GDN prefill pipeline.

    ``auto`` may choose the unchanged vLLM-Ascend prefill path before a live
    request reaches FLA. Once a real FLA call starts, errors are logged
    and propagated instead of retrying the stateful request with another
    backend.
    """

    _preparation_results: ClassVar[
        dict[tuple[GDNRuntimeSignature, str, int | None], _PreparationResult]
    ] = {}
    _logged_decisions: ClassVar[set[tuple[Any, ...]]] = set()

    def __init__(
        self,
        *,
        mode: GDNBackendMode,
        signature: GDNRuntimeSignature,
        layer_name: str,
        pcp_world_size: int,
    ) -> None:
        self.mode = mode
        self.signature = signature
        self.layer_name = layer_name
        self.pcp_world_size = pcp_world_size
        self._operator: Callable[..., Any] | None = None
        self.symbol: str | None = None

    @classmethod
    def create(
        cls,
        *,
        mode: str | GDNBackendMode,
        signature: GDNRuntimeSignature,
        layer_name: str,
        pcp_world_size: int,
    ) -> Self | None:
        parsed_mode = mode if isinstance(mode, GDNBackendMode) else parse_gdn_backend_mode(mode)
        if parsed_mode is GDNBackendMode.NATIVE:
            return None

        reason = cls._eligibility_failure(signature, pcp_world_size)
        if reason is not None:
            if parsed_mode is GDNBackendMode.FLA_NPU:
                raise RuntimeError(
                    "A5 GDN strict fla_npu selection failed during eligibility: "
                    f"{reason}; soc={signature.soc} pcp_world_size={pcp_world_size} "
                    f"dtype={signature.dtype} state_dtype={signature.state_dtype}"
                )
            cls._log_once(
                ("eligibility", signature, pcp_world_size),
                logger.info,
                "A5 GDN selection: backend=native requested=auto stage=eligibility "
                "reason=%s soc=%s pcp_world_size=%s dtype=%s state_dtype=%s layer=%s",
                reason,
                signature.soc,
                pcp_world_size,
                signature.dtype,
                signature.state_dtype,
                layer_name,
            )
            return None

        return cls(
            mode=parsed_mode,
            signature=signature,
            layer_name=layer_name,
            pcp_world_size=pcp_world_size,
        )

    @staticmethod
    def _eligibility_failure(
        signature: GDNRuntimeSignature,
        pcp_world_size: int,
    ) -> str | None:
        if signature.soc not in _SUPPORTED_SOCS:
            return f"unsupported SoC {signature.soc!r}"
        if pcp_world_size != 1:
            return f"requires PCP world size 1, got pcp_world_size={pcp_world_size}"
        if signature.dtype != _GDN_ACTIVATION_DTYPE:
            return f"requires activation dtype {_GDN_ACTIVATION_DTYPE}, got {signature.dtype}"
        if signature.state_dtype not in _SUPPORTED_STATE_DTYPES:
            return f"unsupported state dtype {signature.state_dtype}"
        if signature.num_key_heads <= 0 or signature.num_value_heads <= 0:
            return "key/value head counts must be positive"
        if signature.num_value_heads % signature.num_key_heads != 0:
            return "value head count must be divisible by key head count"
        if signature.key_dim != _GDN_KEY_DIM:
            return f"requires key dimension {_GDN_KEY_DIM}, got {signature.key_dim}"
        if signature.value_dim != _GDN_VALUE_DIM:
            return f"requires value dimension {_GDN_VALUE_DIM}, got {signature.value_dim}"
        gva_ratio = signature.num_value_heads // signature.num_key_heads
        if gva_ratio > _MAX_GVA_RATIO:
            return f"requires GVA ratio <= {_MAX_GVA_RATIO}, got {gva_ratio}"
        if signature.chunk_size != _GDN_CHUNK_SIZE:
            return f"requires chunk size {_GDN_CHUNK_SIZE}, got {signature.chunk_size}"
        return None

    @classmethod
    def _log_once(
        cls,
        key: tuple[Any, ...],
        log: Callable[..., Any],
        message: str,
        *args: Any,
    ) -> None:
        if key in cls._logged_decisions:
            return
        cls._logged_decisions.add(key)
        log(message, *args)

    @classmethod
    def clear_process_caches_for_test(cls) -> None:
        cls._preparation_results.clear()
        cls._logged_decisions.clear()

    def prepare(self, device: torch.device) -> bool:
        key = (self.signature, device.type, device.index)
        cached = self._preparation_results.get(key)
        if cached is not None:
            return self._use_preparation_result(cached)

        try:
            operator, symbol = resolve_chunk_gated_delta_rule_fwd()
        except Exception as exc:
            result = _PreparationResult(
                ready=False,
                stage="resolve",
                exception_type=type(exc).__name__,
                reason=_first_line(exc),
            )
        else:
            try:
                self._scratch_probe(operator, device)
            except Exception as exc:
                result = _PreparationResult(
                    ready=False,
                    symbol=symbol,
                    stage="scratch_probe",
                    exception_type=type(exc).__name__,
                    reason=_first_line(exc),
                )
            else:
                result = _PreparationResult(ready=True, operator=operator, symbol=symbol)

        self._preparation_results[key] = result
        return self._use_preparation_result(result)

    def _use_preparation_result(self, result: _PreparationResult) -> bool:
        if result.ready:
            self._operator = result.operator
            self.symbol = result.symbol
            self._log_once(
                ("selected", self.signature),
                logger.info,
                "A5 GDN selection: backend=fla_npu implementation=a5_prepare_pipeline "
                "symbol=%s soc=%s layer=%s",
                result.symbol,
                self.signature.soc,
                self.layer_name,
            )
            return True

        details = (
            f"stage={result.stage} exception={result.exception_type} reason={result.reason} "
            f"soc={self.signature.soc} pcp_world_size={self.pcp_world_size} "
            f"dtype={self.signature.dtype} state_dtype={self.signature.state_dtype} "
            f"heads={self.signature.num_key_heads}/{self.signature.num_value_heads} "
            f"dims={self.signature.key_dim}/{self.signature.value_dim} "
            f"chunk_size={self.signature.chunk_size} layer={self.layer_name} "
            f"symbol={result.symbol or GDN_FWD_SYMBOL}"
        )
        if self.mode is GDNBackendMode.FLA_NPU:
            raise RuntimeError(f"A5 GDN strict fla_npu selection failed: {details}")
        self._log_once(
            ("prepare-fallback", self.signature, result.stage, result.reason),
            logger.warning,
            "A5 GDN selection: backend=native requested=auto %s",
            details,
        )
        return False

    def _scratch_probe(self, operator: Callable[..., Any], device: torch.device) -> None:
        signature = self.signature
        dtype = getattr(torch, signature.dtype)
        state_dtype = getattr(torch, signature.state_dtype)
        q = torch.full(
            (1, _SCRATCH_TOKENS, signature.num_key_heads, signature.key_dim),
            0.125,
            dtype=dtype,
            device=device,
        )
        k = torch.full_like(q, 0.25)
        v = torch.full(
            (1, _SCRATCH_TOKENS, signature.num_value_heads, signature.value_dim),
            0.0625,
            dtype=dtype,
            device=device,
        )
        g = torch.full(
            (1, _SCRATCH_TOKENS, signature.num_value_heads),
            -0.01,
            dtype=torch.float32,
            device=device,
        )
        beta = torch.full(
            (1, _SCRATCH_TOKENS, signature.num_value_heads),
            0.5,
            dtype=dtype,
            device=device,
        )
        state = torch.zeros(
            (1, signature.num_value_heads, signature.value_dim, signature.key_dim),
            dtype=state_dtype,
            device=device,
        )
        result = operator(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            output_final_state=True,
            chunk_size=signature.chunk_size,
            cu_seqlens=[0, _SCRATCH_TOKENS],
            chunk_indices=[0, 0],
            scale=signature.key_dim**-0.5,
            use_exp2=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            output_a=False,
            state_v_first=True,
            layout="BSND",
        )
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise RuntimeError("FLA GDN scratch probe must return four outputs")
        output, final_state, _, _ = result
        expected_output = (
            1,
            _SCRATCH_TOKENS,
            signature.num_value_heads,
            signature.value_dim,
        )
        expected_state = (1, signature.num_value_heads, signature.value_dim, signature.key_dim)
        if tuple(output.shape) != expected_output:
            raise RuntimeError(
                f"FLA GDN scratch output shape {tuple(output.shape)} != {expected_output}"
            )
        if final_state is None or tuple(final_state.shape) != expected_state:
            actual = None if final_state is None else tuple(final_state.shape)
            raise RuntimeError(f"FLA GDN scratch final-state shape {actual} != {expected_state}")
        if device.type == "npu":
            torch.npu.synchronize()
        if not torch.isfinite(output).all().item() or not torch.isfinite(final_state).all().item():
            raise RuntimeError("FLA GDN scratch probe returned non-finite output")

    def prefill(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        has_initial_state: torch.Tensor,
        scale: float,
        metadata: GDNPrefillMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._operator is None:
            raise RuntimeError("FLA GDN backend must be prepared before prefill")

        # vLLM stores GDN activations as BSND and recurrent state as
        # [N, Hv, V, K]. PR #472 accepts both layouts directly and performs
        # Q/K L2 normalization inside ChunkGatedDeltaRuleFwdPrepare.
        state = initial_state.clone()
        state[~has_initial_state, ...] = 0
        cu_seqlens = metadata.cu_seqlens_host
        chunk_indices = metadata.chunk_indices_host

        try:
            output, final_state, _, _ = self._operator(
                q,
                k,
                v,
                g,
                beta,
                initial_state=state,
                output_final_state=True,
                chunk_size=self.signature.chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                scale=scale,
                use_exp2=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=False,
                allow_neg_eigval=False,
                output_a=False,
                state_v_first=False,
                layout="BSND",
            )
        except Exception:
            logger.exception(
                "A5 GDN execution failed: backend=fla_npu "
                "implementation=a5_prepare_pipeline symbol=%s soc=%s layer=%s inputs=%s",
                self.symbol,
                self.signature.soc,
                self.layer_name,
                _tensor_call_metadata(
                    (q, k, v, g, beta),
                    {
                        "initial_state": state,
                        "cu_seqlens": cu_seqlens,
                        "chunk_indices": chunk_indices,
                    },
                ),
            )
            raise

        return output, final_state
