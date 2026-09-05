# Qwen GDN Phase6-Only Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task in the
> current session, with review checkpoints after each batch. Do not dispatch
> subagents for this implementation.

**Goal:** Reduce the Qwen3.5/Qwen3.6 FLA integration to the single prefill
operator `fla_npu.ops.ascendc.gdn_core_fwd_phase6`, while preserving native
causal convolution, ordinary decode, MTP decode, native prefill fallback, and
A2/A3/A5 support.

**Architecture:** `vllm_ascend/ops/gdn.py` continues to own Qwen GDN routing,
state/cache updates, native convolution, native decode, and native prefill.
`vllm_ascend/ops/gdn_fla.py` becomes a focused optional Phase6 backend that is
created only from the prefill branch, performs eligibility and scratch-probe
checks, normalizes the public FLA call, and propagates real execution failures.

**Tech Stack:** Python, PyTorch/torch-npu, vLLM custom-op routing,
flash-linear-attention-npu AscendC operator bindings, pytest, Markdown.

**Spec:**
`docs/superpowers/specs/2026-09-05-qwen-gdn-phase6-only-refactor-design.md`

## Global constraints

- The implementation branch is `a2-a3-a5-gdn-phase6-only-refactor`.
- The ownership baseline is commit
  `7b717fefb3f6a47c374585e7ba2da176a045707a`.
- Modify or remove production hunks only after running `git diff
  7b717fe...HEAD --` with the exact production path named by the current task
  and confirming that this feature branch introduced or changed them.
- Stop for user review before changing a baseline-owned production hunk.
- The only approved baseline-file exception is the minimum test-only change in
  `tests/ut/quantization/test_utils.py`.
- Do not run `python`, `python3`, pytest, Ruff, or build commands on the local
  Windows workspace. Python and NPU verification commands in this plan run
  later inside the A2/A3/A5 container.
- Preserve `VLLM_ASCEND_GDN_BACKEND=auto` as the default. Supported values are
  exactly `auto`, `fla_npu`, and `native`.
- Remove `VLLM_ASCEND_GDN_OP_BACKENDS`; there is no per-operator override after
  the refactor.
- A2 maps to `ascend910b`, A3 maps to `ascend910_93`, and A5 maps to
  `ascend950`. Do not gate the FLA path with `is_950()`.
- PCP world size greater than one must select native prefill and emit an
  explicit selection log.
- `auto` may fall back only during eligibility, symbol resolution, or the
  synthetic scratch probe. A failure from the real Phase6 request must be
  logged with a traceback and propagated without native retry.
- `gdn_core_fwd_phase6` remains an eager-prefill integration. It is compatible
  with `FULL_DECODE_ONLY` because decode graph capture stays on native paths;
  graph modes that capture prefill are not claimed.
- MTP prompt prefill may use Phase6. MTP convolution and recurrent decode stay
  native.
- Do not modify the `flash-linear-attention-npu` or upstream `vllm` repository.
- Each commit must stage only the files named by its task. Preserve unrelated
  dirty files until their explicit documentation task.

## File structure and responsibilities

| File | Planned responsibility |
| --- | --- |
| `vllm_ascend/ops/gdn_fla.py` | Phase6-only backend policy, preflight, layout conversion, execution, and logs |
| `vllm_ascend/ops/gdn.py` | Existing GDN routing plus one optional Phase6 prefill branch |
| `vllm_ascend/envs.py` | Register the single global backend variable |
| `vllm_ascend/platform.py` | Preload FLA OPP before vLLM-Ascend custom OPP registration when Phase6 can be selected |
| `vllm_ascend/device/device_config.py` | Retain semantic A2/A3/A5-to-FLA-SoC mapping |
| `vllm_ascend/quantization/utils.py` | Retain and document optional-config Dynamic MX graph safety |
| `tests/ut/ops/test_gdn_fla.py` | Phase6 backend and Qwen routing contract tests |
| `tests/ut/device/test_device_config.py` | A2/A3/A5 capability mapping regression tests |
| `tests/ut/test_platform.py` | Phase6 OPP preload policy tests |
| `tests/ut/quantization/test_utils.py` | Dynamic MX no-context regression test |
| `tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py` | Real-NPU Phase6/native prefill comparison only |
| `tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py` | Remove the branch-added 0.8B FLA smoke; preserve upstream tests |
| `tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py` | Keep the existing environment-overridable Qwen3.6 smoke and rename only its branch-added Phase6 test |
| `docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md` | Public short design entry pointing to current authoritative docs |
| `docs/superpowers/guides/2026-09-05-qwen-gdn-phase6-only-validation-guide-zh.md` | Chinese install, logging, operator, 35B serve, and result-recording guide |
| `CLAUDE.md` | Restore the one-line baseline content |
| `vllm_ascend/ops/gdn_a5.py` | Delete obsolete branch-created wildcard compatibility module |

---

### Task 1: Replace the generic adapter contract with Phase6-only unit tests

**Files:**

- Rewrite: `tests/ut/ops/test_gdn_fla.py`

**Ownership check:** This test file is absent at `7b717fe` and is wholly owned
by the feature branch. It may be replaced without touching upstream tests.

**Interfaces:**

- Consumes: the approved backend modes and Phase6 behavior from the spec.
- Produces: executable contracts for `GDNBackendMode`,
  `GDNPhase6RuntimeSignature`, `GDNPhase6PrefillMetadata`,
  `FlaGDNPhase6Backend`, `parse_gdn_backend_mode()`, and
  `resolve_gdn_core_fwd_phase6()`.

- [ ] **Step 1: Record the branch ownership evidence**

Run locally:

```powershell
git diff --name-status 7b717fe...HEAD -- tests/ut/ops/test_gdn_fla.py
git log --format="%h %an %s" 7b717fe..HEAD -- tests/ut/ops/test_gdn_fla.py
```

Expected: the file is reported as added by the feature branch. If it is not,
stop before rewriting it.

- [ ] **Step 2: Replace obsolete dispatcher tests with backend-policy tests**

Use this test-module import surface:

```python
import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_fla import (
    FlaGDNPhase6Backend,
    GDNBackendMode,
    GDNPhase6PrefillMetadata,
    GDNPhase6RuntimeSignature,
    parse_gdn_backend_mode,
)
```

Clear process-wide backend caches around every unit test:

```python
@pytest.fixture(autouse=True)
def clear_phase6_process_caches():
    FlaGDNPhase6Backend.clear_process_caches_for_test()
    yield
    FlaGDNPhase6Backend.clear_process_caches_for_test()
```

Add exact mode coverage:

```python
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
```

Add eligibility cases using one shared BF16 signature:

```python
SIGNATURE = GDNPhase6RuntimeSignature(
    soc="ascend950",
    dtype="bfloat16",
    state_dtype="float32",
    num_key_heads=1,
    num_value_heads=2,
    key_dim=128,
    value_dim=128,
    chunk_size=64,
)


def _fake_phase6_outputs(tokens=64, sequences=1):
    return (
        torch.zeros((1, 2, tokens, 128), dtype=torch.bfloat16),
        torch.zeros((sequences, 2, 128, 128), dtype=torch.float32),
        torch.zeros((1, tokens, 2), dtype=torch.float32),
        torch.zeros((1, 2, tokens, 64), dtype=torch.bfloat16),
    )


def _backend(mode):
    backend = FlaGDNPhase6Backend.create(
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
        "metadata": GDNPhase6PrefillMetadata(
            cu_seqlens_host=(0, 1, tokens),
            chunk_indices_host=(0, 0, 1, 0),
        ),
    }


@pytest.mark.parametrize("soc", ["ascend910b", "ascend910_93", "ascend950"])
def test_phase6_backend_accepts_a2_a3_a5(soc):
    backend = FlaGDNPhase6Backend.create(
        mode="auto",
        signature=dataclasses.replace(SIGNATURE, soc=soc),
        layer_name="model.layers.0.linear_attn",
        pcp_world_size=1,
    )
    assert backend is not None


def test_phase6_backend_auto_uses_native_for_pcp(monkeypatch):
    with patch("vllm_ascend.ops.gdn_fla.logger.info") as info:
        backend = FlaGDNPhase6Backend.create(
            mode="auto",
            signature=SIGNATURE,
            layer_name="model.layers.0.linear_attn",
            pcp_world_size=2,
        )
    assert backend is None
    assert "pcp_world_size=2" in str(info.call_args)


def test_phase6_backend_strict_rejects_pcp():
    with pytest.raises(RuntimeError, match="PCP world size 1"):
        FlaGDNPhase6Backend.create(
            mode="fla_npu",
            signature=SIGNATURE,
            layer_name="model.layers.0.linear_attn",
            pcp_world_size=2,
        )
```

Add resolve/probe behavior with a fake public operator:

```python
def test_auto_resolve_failure_returns_native(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.resolve_gdn_core_fwd_phase6",
        MagicMock(side_effect=ImportError("missing fla_npu")),
    )
    backend = FlaGDNPhase6Backend.create(
        mode="auto",
        signature=SIGNATURE,
        layer_name="model.layers.0.linear_attn",
        pcp_world_size=1,
    )
    assert backend is not None
    assert backend.prepare(torch.device("cpu")) is False


def test_strict_resolve_failure_raises(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.resolve_gdn_core_fwd_phase6",
        MagicMock(side_effect=ImportError("missing fla_npu")),
    )
    backend = FlaGDNPhase6Backend.create(
        mode="fla_npu",
        signature=SIGNATURE,
        layer_name="model.layers.0.linear_attn",
        pcp_world_size=1,
    )
    assert backend is not None
    with pytest.raises(RuntimeError, match="resolve"):
        backend.prepare(torch.device("cpu"))


def test_scratch_probe_runs_once_per_device_and_signature(monkeypatch):
    raw = MagicMock(return_value=_fake_phase6_outputs())
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.resolve_gdn_core_fwd_phase6",
        lambda: (raw, "fla_npu.ops.ascendc.gdn_core_fwd_phase6"),
    )
    FlaGDNPhase6Backend.clear_process_caches_for_test()
    first = _backend("auto")
    second = _backend("auto")
    assert first.prepare(torch.device("cpu")) is True
    assert second.prepare(torch.device("cpu")) is True
    assert raw.call_count == 1
```

Use small CPU fake tensors for layout tests. The fake operator must capture
arguments and return `(o, final_state, g_cumsum, a)` with shapes matching the
public Phase6 contract:

```python
def test_prefill_normalizes_phase6_layout_and_metadata(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.l2norm_fwd",
        lambda tensor: tensor,
    )

    def fake_phase6(q, k, v, g, beta, **kwargs):
        captured.update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=kwargs["initial_state"],
            cu_seqlens=kwargs["cu_seqlens"],
            chunk_indices=kwargs["chunk_indices"],
        )
        return (
            torch.zeros((q.shape[0], v.shape[1], q.shape[2], v.shape[3]), dtype=q.dtype),
            torch.zeros_like(kwargs["initial_state"]),
            torch.zeros_like(g),
            torch.zeros((q.shape[0], v.shape[1], q.shape[2], 64), dtype=torch.float32),
        )

    raw = MagicMock(side_effect=fake_phase6)
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.resolve_gdn_core_fwd_phase6",
        lambda: (raw, "fla_npu.ops.ascendc.gdn_core_fwd_phase6"),
    )
    backend = _backend("fla_npu")
    assert backend.prepare(torch.device("cpu")) is True
    captured.clear()
    output, final_state = backend.prefill(**_prefill_inputs())

assert captured["q"].shape == (1, 1, 65, 128)
assert captured["k"].shape == (1, 1, 65, 128)
assert captured["v"].shape == (1, 2, 65, 128)
assert captured["beta"].dtype == torch.float32
assert captured["initial_state"].shape == (2, 2, 128, 128)
assert captured["cu_seqlens"] == [0, 1, 65]
assert captured["chunk_indices"] == [0, 0, 1, 0]
assert output.shape == (1, 65, 2, 128)
assert final_state.shape == (2, 2, 128, 128)
```

Add a live-call failure test that first marks the synthetic probe successful,
then makes the second call raise `RuntimeError("device execution failed")`:

```python
def test_live_phase6_failure_is_logged_and_propagated(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.l2norm_fwd",
        lambda tensor: tensor,
    )
    raw = MagicMock(
        side_effect=[
            _fake_phase6_outputs(),
            RuntimeError("device execution failed"),
        ]
    )
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.resolve_gdn_core_fwd_phase6",
        lambda: (raw, "fla_npu.ops.ascendc.gdn_core_fwd_phase6"),
    )
    backend = _backend("auto")
    assert backend.prepare(torch.device("cpu")) is True

with patch("vllm_ascend.ops.gdn_fla.logger.exception") as error_log:
    with pytest.raises(RuntimeError, match="device execution failed"):
        backend.prefill(**_prefill_inputs())
error_log.assert_called_once()
assert raw.call_count == 2
```

Do not add a native callback to this test: the absence of such a callback is
the contract that real execution cannot retry natively.

- [ ] **Step 3: Add routing tests that reject FLA usage outside prefill**

Retain the existing lightweight mixed-batch fixture in
`test_fla_mixed_decode_prefill_routes_and_merges_outputs`, rename it to
`test_mixed_decode_prefill_uses_phase6_only_for_prefill`, and make these exact
changes:

```python
phase6_backend = SimpleNamespace(
    prepare=MagicMock(return_value=True),
    prefill=MagicMock(
        side_effect=lambda **kwargs: (
            torch.full_like(kwargs["v"], 20),
            kwargs["initial_state"] + 2,
        )
    ),
)

with (
    patch.object(
        AscendGatedDeltaNetAttention,
        "_get_fla_gdn_prefill_backend",
        return_value=phase6_backend,
    ) as get_backend,
    patch(
        "vllm_ascend.ops.gdn.torch.ops._C_ascend.npu_causal_conv1d_custom",
        side_effect=native_causal_conv,
    ) as native_conv,
    patch(
        "vllm_ascend.ops.gdn.torch.ops._C_ascend.npu_recurrent_gated_delta_rule",
        side_effect=native_recurrent,
    ) as native_decode,
):
    AscendGatedDeltaNetAttention._forward_core(
        layer,
        torch.zeros((3, 2)),
        torch.zeros((3, 1)),
        torch.zeros((3, 1)),
        core_attn_out,
    )

get_backend.assert_called_once()
phase6_backend.prepare.assert_called_once()
phase6_backend.prefill.assert_called_once()
native_conv.assert_called_once()
native_decode.assert_called_once()
```

Add pure ordinary-decode and pure MTP-decode variants of the same fixture with
`num_prefills=0`. Patch `_get_fla_gdn_prefill_backend` with a `MagicMock` and
assert `get_backend.assert_not_called()` after `_forward_core`. Retain the
existing native-output mocks needed to execute `_forward_core`, and assert the
native causal-convolution and recurrent operators were called. Remove every
assertion for `GDNOperator`, per-operator overrides, six-stage FLA composition,
FLA causal convolution, and FLA recurrent decode.

- [ ] **Step 4: Record the deferred test command**

Do not run it locally. Run later in an installed A2/A3/A5 container:

```bash
cd /home/z00886386/vllm-ascend
pytest -q tests/ut/ops/test_gdn_fla.py
```

Expected before Task 2: collection failure because the new Phase6-only public
types do not yet exist. Expected after Tasks 2 and 3: all tests pass.

- [ ] **Step 5: Commit the contract tests**

```bash
git add tests/ut/ops/test_gdn_fla.py
git commit -m "test(gdn): define Phase6-only backend contract"
```

---

### Task 2: Reduce `gdn_fla.py` to one focused Phase6 backend

**Files:**

- Rewrite: `vllm_ascend/ops/gdn_fla.py`
- Delete: `vllm_ascend/ops/gdn_a5.py`
- Test: `tests/ut/ops/test_gdn_fla.py`

**Ownership check:** Both production files were added after `7b717fe`. Confirm
that result before replacing or deleting them.

**Interfaces:**

- Consumes: the contracts from Task 1 and
  `fla_npu.ops.ascendc.gdn_core_fwd_phase6`.
- Produces:

The module exposes exactly four public types:

- `GDNBackendMode` with `AUTO`, `FLA_NPU`, and `NATIVE` values;
- immutable `GDNPhase6RuntimeSignature`, containing SoC, activation/state
  dtype names, key/value head counts, key/value dimensions, and chunk size;
- immutable `GDNPhase6PrefillMetadata`, containing host sequence boundaries
  and host chunk indices;
- `FlaGDNPhase6Backend`, with `create()`, `prepare()`, `prefill()`, and the
  test-only cache reset method.

The module also exposes `parse_gdn_backend_mode(value: str)` and
`resolve_gdn_core_fwd_phase6()`. Their exact bodies are specified in Steps 2
and 3, and the exact backend method signatures are specified by the Task 1
test calls and Steps 3-5.

- [ ] **Step 1: Verify ownership and delete the compatibility module**

Run locally:

```powershell
git diff --name-status 7b717fe...HEAD -- vllm_ascend/ops/gdn_fla.py vllm_ascend/ops/gdn_a5.py
rg -n "gdn_a5|A5GDNAdapter|A5GDNOperatorDispatcher" vllm_ascend tests docs
```

Expected: both files are branch additions, and `gdn_a5.py` has no runtime or
test consumer that must be migrated. Delete `vllm_ascend/ops/gdn_a5.py`.

- [ ] **Step 2: Replace generic configuration and operator enumeration**

Use this production-module import surface:

```python
from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, ClassVar, Self

import torch
from vllm.logger import init_logger
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

logger = init_logger(__name__)
```

Keep only these modes and parser:

```python
class GDNBackendMode(StrEnum):
    AUTO = "auto"
    FLA_NPU = "fla_npu"
    NATIVE = "native"


def parse_gdn_backend_mode(value: str) -> GDNBackendMode:
    try:
        return GDNBackendMode(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in GDNBackendMode)
        raise ValueError(
            f"Invalid GDN backend mode {value!r}; expected one of: {valid}."
        ) from exc
```

Delete `GDNOperator`, `GDNBackendConfig`, per-operator overrides,
`GDNOperatorSelection`, aliases with `A5` names, and all small-operator symbol
tables/resolvers.

- [ ] **Step 3: Implement lazy resolution and eligibility**

Resolution must import only the public Phase6 module:

```python
def resolve_gdn_core_fwd_phase6() -> tuple[Callable[..., Any], str]:
    module_name = "fla_npu.ops.ascendc"
    attribute = "gdn_core_fwd_phase6"
    module = importlib.import_module(module_name)
    return getattr(module, attribute), f"{module_name}.{attribute}"
```

`FlaGDNPhase6Backend.create()` must return `None` without importing FLA for
explicit `native`. It must validate `signature.soc` against
`{"ascend910b", "ascend910_93", "ascend950"}` and enforce the exact Qwen
integration subset of the FLA Phase6 contract:

```python
signature.dtype == "bfloat16"
signature.state_dtype in {"bfloat16", "float32"}
signature.num_key_heads > 0
signature.num_value_heads > 0
signature.num_value_heads % signature.num_key_heads == 0
signature.key_dim == 128
signature.value_dim in {128, 256}
signature.chunk_size == 64
pcp_world_size == 1
```

These checks match the public FLA binding and host validation: Phase6 accepts
K=128, V=128/256, chunk size 64/128, BF16/FP16 activations, and state dtype
float32 or activation dtype. This integration deliberately stays on the Qwen
BF16/chunk-64 subset already used by `gdn.py`.

For an ineligible `auto` request, log once and return `None`. For an ineligible
strict `fla_npu` request, raise this form:

```python
raise RuntimeError(
    "GDN Phase6 strict fla_npu selection failed during eligibility: "
    f"{reason}; soc={signature.soc} pcp_world_size={pcp_world_size} "
    f"dtype={signature.dtype} state_dtype={signature.state_dtype}"
)
```

- [ ] **Step 4: Implement cached scratch preflight**

Cache an internal immutable preparation result by
`(signature, device.type, device.index)` at class scope. A successful result
contains both the resolved callable and symbol; a failed result contains the
decision stage, exception type, and first-line reason. When another layer hits
the cache, copy the cached callable/symbol onto that backend instance before
returning `True`. This avoids a second probe without leaving the second backend
unable to execute. `clear_process_caches_for_test()` clears preparation results
and one-time-log keys.

The scratch probe uses one 64-token sequence and the runtime signature:

```python
q = torch.full((1, nk, 64, dk), 0.125, dtype=dtype, device=device)
k = torch.full((1, nk, 64, dk), 0.25, dtype=dtype, device=device)
v = torch.full((1, nv, 64, dv), 0.0625, dtype=dtype, device=device)
g = torch.full((1, 64, nv), -0.01, dtype=torch.float32, device=device)
beta = torch.full((1, 64, nv), 0.5, dtype=torch.float32, device=device)
state = torch.zeros((1, nv, dk, dv), dtype=state_dtype, device=device)
```

Invoke the resolved function with:

```python
result = operator(
    q,
    k,
    v,
    g,
    beta,
    initial_state=state,
    output_final_state=True,
    chunk_size=64,
    cu_seqlens=[0, 64],
    chunk_indices=[0, 0],
    scale=dk**-0.5,
)
```

Validate a four-item return, output shape `(1, nv, 64, dv)`, final-state shape
`(1, nv, dk, dv)`, finite output/state, and synchronize only when
`device.type == "npu"`.

On resolve/probe failure, `auto` logs a WARNING containing
`stage=resolve|scratch_probe`, exception type, first-line reason, SoC, PCP,
dtype, state dtype, dimensions, layer, and symbol when known, then returns
`False`. Strict mode raises with the same metadata. Cache the native decision
so repeated layers do not repeat the failing probe.

- [ ] **Step 5: Implement the real Phase6 call and layout conversion**

Before calling the operator:

```python
output_dtype = q.dtype
q = l2norm_fwd(q)
k = l2norm_fwd(k)
state = initial_state.clone()
state[~has_initial_state, ...] = 0
state = state.transpose(-1, -2).contiguous()
```

Call Phase6 exactly once for the real request:

```python
try:
    output, final_state, _, _ = self._operator(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        g,
        beta.float(),
        initial_state=state,
        output_final_state=True,
        chunk_size=self.signature.chunk_size,
        cu_seqlens=list(metadata.cu_seqlens_host),
        chunk_indices=list(metadata.chunk_indices_host),
        scale=scale,
    )
except Exception:
    logger.exception(
        "GDN Phase6 execution failed: backend=fla_npu symbol=%s "
        "soc=%s layer=%s inputs=%s",
        self.symbol,
        self.signature.soc,
        self.layer_name,
        _tensor_call_metadata(
            (q, k, v, g, beta),
            {
                "initial_state": state,
                "cu_seqlens": list(metadata.cu_seqlens_host),
                "chunk_indices": list(metadata.chunk_indices_host),
            },
        ),
    )
    raise
```

Return:

```python
return (
    output.to(output_dtype).transpose(1, 2).contiguous(),
    final_state.transpose(-1, -2).contiguous(),
)
```

The exception handler must not catch-and-run native prefill. Log tensor shapes,
dtypes, devices, strides, and metadata lengths; never log tensor values.

- [ ] **Step 6: Remove every obsolete implementation**

Delete the six-stage FLA composition, FLA causal convolution, recurrent decode,
solve-tri debug code, generic dispatcher, stateful probe cloning, warmup of
convolution/decode, and compatibility aliases. Verify locally:

```powershell
rg -n "GDNOperator|FlaGDNAdapter|FlaGDNOperatorDispatcher|run_gdn_decode_pipeline|causal_conv1d|recurrent_gated_delta_rule|chunk_local_cumsum|chunk_scaled_dot_kkt|solve_tri|recompute_w_u_fwd|chunk_gated_delta_rule_fwd_h|chunk_fwd_o|A5GDN" vllm_ascend/ops/gdn_fla.py vllm_ascend/ops/gdn_a5.py
```

Expected: no matches; `gdn_a5.py` no longer exists.

- [ ] **Step 7: Run local static checks**

```powershell
git diff --check -- vllm_ascend/ops/gdn_fla.py vllm_ascend/ops/gdn_a5.py
rg -n "<<<<<<<|=======|>>>>>>>" vllm_ascend/ops/gdn_fla.py
git diff 7b717fe...HEAD -- vllm_ascend/ops/gdn_fla.py vllm_ascend/ops/gdn_a5.py
```

Do not run the deferred pytest command locally.

- [ ] **Step 8: Commit the focused backend**

```bash
git add vllm_ascend/ops/gdn_fla.py vllm_ascend/ops/gdn_a5.py
git commit -m "refactor(gdn): reduce FLA backend to Phase6 prefill"
```

---

### Task 3: Rewire Qwen routing so only prefill can reach FLA

**Files:**

- Modify: `vllm_ascend/ops/gdn.py`
- Test: `tests/ut/ops/test_gdn_fla.py`

**Ownership check:** Use the baseline diff for every edited hunk. Remove only
the feature-branch adapter imports, caches, warmup, and conditional wrappers.
When a branch-added wrapper surrounds baseline native code, restore the
baseline native statements rather than rewriting the algorithm.

**Interfaces:**

- Consumes: `FlaGDNPhase6Backend`, `GDNPhase6RuntimeSignature`, and
  `GDNPhase6PrefillMetadata` from Task 2.
- Produces: one `_get_fla_gdn_prefill_backend()` helper used only inside the
  `num_prefills > 0` branch.

- [ ] **Step 1: Audit each target hunk against baseline**

Run locally:

```powershell
git diff 7b717fe...HEAD -- vllm_ascend/ops/gdn.py
git blame 7b717fe -- vllm_ascend/ops/gdn.py
```

Explicitly classify these feature-branch areas before editing:

- `gdn_fla` imports;
- `_fla_gdn_dispatchers` and `_get_fla_gdn_adapter()`;
- forward-wide FLA warmup;
- `fla_adapter` creation before convolution;
- causal-convolution adapter branches;
- ordinary-decode adapter branch;
- Phase6 prefill adapter branch.

- [ ] **Step 2: Replace imports and backend cache**

Import only:

```python
from vllm_ascend.ops.gdn_fla import (
    FlaGDNPhase6Backend,
    GDNPhase6PrefillMetadata,
    GDNPhase6RuntimeSignature,
)
```

Replace the generic dispatcher cache with an instance cache whose key includes
backend mode, runtime signature, and PCP world size:

```python
def _get_fla_gdn_prefill_backend(
    self,
    activation: torch.Tensor,
    state: torch.Tensor,
) -> FlaGDNPhase6Backend | None:
    soc = get_fla_gdn_soc()
    signature = GDNPhase6RuntimeSignature(
        soc=soc or "unsupported",
        dtype=str(activation.dtype).removeprefix("torch."),
        state_dtype=str(state.dtype).removeprefix("torch."),
        num_key_heads=self.num_k_heads // self.tp_size,
        num_value_heads=self.num_v_heads // self.tp_size,
        key_dim=self.head_k_dim,
        value_dim=self.head_v_dim,
        chunk_size=64,
    )
    pcp_world_size = get_pcp_group().world_size
    cache_key = (ascend_envs.VLLM_ASCEND_GDN_BACKEND, signature, pcp_world_size)
    cached = getattr(self, "_fla_gdn_phase6_backend_cache", None)
    if cached is None or cached[0] != cache_key:
        cached = (
            cache_key,
            FlaGDNPhase6Backend.create(
                mode=ascend_envs.VLLM_ASCEND_GDN_BACKEND,
                signature=signature,
                layer_name=self.prefix,
                pcp_world_size=pcp_world_size,
            ),
        )
        self._fla_gdn_phase6_backend_cache = cached
    return cached[1]
```

Do not call this helper before the prefill branch.

- [ ] **Step 3: Restore unconditional native convolution**

Remove both `if fla_adapter is not None` branches. For prefill and decode,
retain the existing native call shape:

```python
output_non_spec = torch.empty_like(mixed_qkv_non_spec)
torch.ops._C_ascend.npu_causal_conv1d_custom(
    output_non_spec,
    mixed_qkv_non_spec,
    conv_weights_T,
    conv_state=self_kv_cache[0],
    bias_opt=self.conv1d.bias,
    query_start_loc_opt=query_start_loc,
    cache_indices_opt=cache_indices,
    initial_state_mode_opt=initial_state_mode,
    num_accepted_tokens_opt=None,
    activation_mode=activation_num,
    pad_slot_id=PAD_SLOT_ID,
    run_mode=run_mode,
)
mixed_qkv_non_spec = output_non_spec
```

Use the existing variables in each branch; do not combine prefill and decode
metadata handling.

- [ ] **Step 4: Restore unconditional native ordinary and MTP decode**

Preserve the existing speculative decode call. In mixed ordinary decode,
remove the adapter condition and always execute:

```python
query_decode = l2norm_fwd(query_decode)
key_decode = l2norm_fwd(key_decode)
core_attn_out_decode = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
    query=query_decode.squeeze(0),
    key=key_decode.squeeze(0),
    value=value_decode.squeeze(0),
    g=g_non_spec[:, :num_decode_tokens].squeeze(0),
    beta=beta_non_spec[:, :num_decode_tokens].squeeze(0),
    state=ssm_state,
    scale=key_decode.shape[-1] ** -0.5,
    actual_seq_lengths=actual_seq_lengths,
    ssm_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_decodes],
).unsqueeze(0)
```

Also remove the adapter branch in the pure non-spec decode block following
`elif attn_metadata.num_decodes > 0`. Preserve its existing normalization and
native call exactly:

```python
query_non_spec = l2norm_fwd(query_non_spec)
key_non_spec = l2norm_fwd(key_non_spec)
core_attn_out_non_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
    query=query_non_spec.squeeze(0),
    key=key_non_spec.squeeze(0),
    value=value_non_spec.squeeze(0),
    g=g_non_spec.squeeze(0) if g_non_spec is not None else g_non_spec,
    beta=beta_non_spec.squeeze(0) if beta_non_spec is not None else beta_non_spec,
    state=ssm_state,
    scale=key_non_spec.shape[-1] ** -0.5,
    actual_seq_lengths=actual_seq_lengths,
    ssm_state_indices=non_spec_state_indices_tensor,
).unsqueeze(0)
```

- [ ] **Step 5: Move all FLA creation and preparation into prefill**

Inside `if attn_metadata.num_prefills > 0`, after slicing mixed decode tokens,
create and preflight the backend:

```python
phase6_backend = self._get_fla_gdn_prefill_backend(query_non_spec, ssm_state)
use_phase6 = phase6_backend is not None and phase6_backend.prepare(query_non_spec.device)
```

When `use_phase6` is true, call only Phase6:

```python
chunk_meta = attn_metadata.non_spec_prefill_metadata.chunk
initial_state = ssm_state[prefill_state_indices]
core_attn_out_non_spec, last_recurrent_state = phase6_backend.prefill(
    q=query_non_spec,
    k=key_non_spec,
    v=value_non_spec,
    g=g_non_spec,
    beta=beta_non_spec,
    initial_state=initial_state,
    has_initial_state=prefill_has_initial_state,
    scale=key_non_spec.shape[-1] ** -0.5,
    metadata=GDNPhase6PrefillMetadata(
        cu_seqlens_host=tuple(chunk_meta.cu_seqlens_host),
        chunk_indices_host=tuple(chunk_meta.chunk_indices_chunk64_host),
    ),
)
ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)
```

When `use_phase6` is false, fall through to the unchanged existing fused-native
or Triton-native prefill code. Do not duplicate that fallback in `gdn_fla.py`.

- [ ] **Step 6: Delete forward-wide warmup and compilation guard**

Remove the block in `forward()` that creates a generic adapter and warms
convolution/decode before graph tracing. Its purpose disappears because FLA is
resolved only from eager prefill. Preserve the rest of `forward()` exactly.

- [ ] **Step 7: Prove decode has no FLA dependency**

Run locally:

```powershell
rg -n "phase6|fla_gdn|FlaGDN" vllm_ascend/ops/gdn.py
rg -n -B 20 -A 40 "num_prefills > 0" vllm_ascend/ops/gdn.py
rg -n "npu_causal_conv1d_custom|npu_recurrent_gated_delta_rule" vllm_ascend/ops/gdn.py
git diff --check -- vllm_ascend/ops/gdn.py tests/ut/ops/test_gdn_fla.py
```

Expected: every FLA reference is either an import, helper definition, or below
the prefill condition. Native convolution and recurrent calls remain present.

- [ ] **Step 8: Commit routing changes**

```bash
git add vllm_ascend/ops/gdn.py tests/ut/ops/test_gdn_fla.py
git commit -m "refactor(gdn): isolate FLA Phase6 to prefill"
```

---

### Task 4: Simplify environment and OPP preload policy

**Files:**

- Modify: `vllm_ascend/envs.py`
- Modify: `vllm_ascend/platform.py`
- Preserve/verify: `vllm_ascend/device/device_config.py`
- Modify: `tests/ut/test_platform.py`
- Preserve/verify: `tests/ut/device/test_device_config.py`

**Ownership check:** The GDN environment entries and preload helper are
feature-branch hunks. The semantic device mapping was also introduced by this
branch and is retained rather than rewritten.

**Interfaces:**

- Consumes: `VLLM_ASCEND_GDN_BACKEND` and `is_fla_gdn_supported()`.
- Produces: deterministic import ordering for the sole Phase6 operator.

- [ ] **Step 1: Remove the per-operator environment variable**

Keep this registration in `envs.py`:

```python
"VLLM_ASCEND_GDN_BACKEND": lambda: os.getenv(
    "VLLM_ASCEND_GDN_BACKEND", "auto"
).lower(),
```

Delete `VLLM_ASCEND_GDN_OP_BACKENDS` and its examples.

- [ ] **Step 2: Simplify the preload helper**

In `_import_fla_npu_before_custom_opp()`:

```python
backend = os.environ.get("VLLM_ASCEND_GDN_BACKEND", "auto").strip().lower()
if backend == "native":
    return

try:
    from vllm_ascend.device.device_config import is_fla_gdn_supported
    if not is_fla_gdn_supported():
        return
except Exception:
    return

try:
    importlib.import_module("fla_npu.ops.ascendc")
except Exception as exc:
    if backend == "fla_npu":
        raise RuntimeError(
            "fla_npu import failed during strict GDN Phase6 preload"
        ) from exc
    logger.warning(
        "fla_npu preload failed; GDN Phase6 auto mode will use native prefill: %s: %s",
        type(exc).__name__,
        str(exc).splitlines()[0],
    )
```

Retain the documented reason for importing FLA before
`bootstrap_custom_op_env`: both custom OPP sets must be visible before kernel
manager indexing.

- [ ] **Step 3: Replace platform tests for override behavior**

Delete `test_honors_fla_operator_override`. Retain and tighten these cases:

```python
@pytest.mark.parametrize(
    "device_type",
    [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5],
)
@pytest.mark.parametrize("backend", ["auto", "fla_npu"])
def test_preloads_phase6_on_supported_accelerators(
    self, monkeypatch, device_type, backend
):
    imported_modules = []
    monkeypatch.setenv("VLLM_ASCEND_GDN_BACKEND", backend)
    monkeypatch.setattr(
        "vllm_ascend.device.device_config.get_ascend_device_type",
        lambda: device_type,
    )
    monkeypatch.setattr(
        "importlib.import_module",
        lambda module_name: imported_modules.append(module_name),
    )
    _import_fla_npu_before_custom_opp()
    assert imported_modules == ["fla_npu.ops.ascendc"]


@pytest.mark.parametrize(
    ("device_type", "backend"),
    [
        (AscendDeviceType._310P, "auto"),
        (AscendDeviceType.A2, "native"),
    ],
)
def test_skips_phase6_preload_for_ineligible_configurations(
    self, monkeypatch, device_type, backend
):
    imported_modules = []
    monkeypatch.setenv("VLLM_ASCEND_GDN_BACKEND", backend)
    monkeypatch.setattr(
        "vllm_ascend.device.device_config.get_ascend_device_type",
        lambda: device_type,
    )
    monkeypatch.setattr(
        "importlib.import_module",
        lambda module_name: imported_modules.append(module_name),
    )
    _import_fla_npu_before_custom_opp()
    assert imported_modules == []
```

Retain the existing ordering assertion that records FLA import before custom
OPP bootstrap. Add two import-failure tests: in `auto`, an import side effect
of `FileNotFoundError("missing FLA OPP")` must return normally and log one
warning; in strict `fla_npu`, the same side effect must raise
`RuntimeError` chained from `FileNotFoundError`. These cases cover both a
missing Python package and failures raised while preparing the packaged custom
OPP.

- [ ] **Step 4: Verify A2/A3/A5 semantic mapping remains exact**

Confirm the production map is unchanged:

```python
_FLA_GDN_SOC_BY_DEVICE_TYPE = {
    AscendDeviceType.A2: "ascend910b",
    AscendDeviceType.A3: "ascend910_93",
    AscendDeviceType.A5: "ascend950",
}
```

Keep the existing parameterized tests and 310P negative test. Search for an
A5-only routing guard:

```powershell
rg -n "is_950|AscendDeviceType\.A5" vllm_ascend/ops/gdn.py vllm_ascend/ops/gdn_fla.py vllm_ascend/platform.py
```

Expected: no `is_950` gating in the GDN runtime/preload paths.

- [ ] **Step 5: Verify the removed variable is absent from current code/tests**

```powershell
rg -n "VLLM_ASCEND_GDN_OP_BACKENDS" vllm_ascend tests
git diff --check -- vllm_ascend/envs.py vllm_ascend/platform.py tests/ut/test_platform.py
```

Expected: no matches for the removed variable.

- [ ] **Step 6: Record deferred unit tests and commit**

Run later in the NPU container:

```bash
cd /home/z00886386/vllm-ascend
pytest -q tests/ut/device/test_device_config.py tests/ut/test_platform.py
```

Commit locally after static checks:

```bash
git add vllm_ascend/envs.py vllm_ascend/platform.py \
  tests/ut/test_platform.py
git commit -m "refactor(gdn): simplify Phase6 backend configuration"
```

---

### Task 5: Retain and document Dynamic MX compile safety

**Files:**

- Modify: `vllm_ascend/quantization/utils.py`
- Modify under approved exception: `tests/ut/quantization/test_utils.py`

**Ownership check:** The production change from
`get_current_vllm_config()` to `get_current_vllm_config_or_none()` is
feature-branch-owned. The test file is baseline-owned, and the user explicitly
approved the minimum mock update and one new regression test.

**Interfaces:**

- Consumes: optional vLLM config context during graph tracing.
- Produces: algorithm `0` when all config contexts are absent, without changing
  model-specific behavior when a config exists.

- [ ] **Step 1: Add an accurate production comment**

Immediately before the optional lookup, add:

```python
# This missing-context case was first observed while tracing Qwen GDN with
# FULL_DECODE_ONLY. The helper is shared by all A5 Dynamic MX quantization
# paths, so keep the fallback model-agnostic: without any config context,
# algorithm 0 is the safe default.
vllm_config = get_current_vllm_config_or_none()
if vllm_config is None:
    return 0
```

Do not imply that the helper affects only GDN.

- [ ] **Step 2: Update only the stale mock targets**

In `tests/ut/quantization/test_utils.py`, replace:

```python
@patch("vllm.config.get_current_vllm_config")
```

with:

```python
@patch("vllm.config.get_current_vllm_config_or_none")
```

for the two existing omitted-config tests.

- [ ] **Step 3: Add no-context regression coverage**

Add:

```python
@patch(
    "vllm_ascend.quantization.utils.get_current_hardware_profile",
    return_value=get_hardware_profile(AscendDeviceType.A5),
)
@patch("vllm.config.get_current_vllm_config_or_none", return_value=None)
def test_defaults_to_zero_when_all_config_context_is_missing(
    self,
    _mock_current_config,
    _mock_profile,
):
    self.assertEqual(get_dynamic_mx_quant_scale_alg(), 0)
```

- [ ] **Step 4: Audit the approved exception and record deferred test**

Run locally:

```powershell
git diff 7b717fe...HEAD -- vllm_ascend/quantization/utils.py
git diff -- tests/ut/quantization/test_utils.py
git diff --check -- vllm_ascend/quantization/utils.py tests/ut/quantization/test_utils.py
```

Run later in the installed container:

```bash
pytest -q tests/ut/quantization/test_utils.py -k dynamic_mx
```

- [ ] **Step 5: Commit the compatibility fix and test**

```bash
git add vllm_ascend/quantization/utils.py tests/ut/quantization/test_utils.py
git commit -m "test(quant): cover missing Dynamic MX config context"
```

---

### Task 6: Reduce real-NPU and model tests to Phase6 scope

**Files:**

- Rewrite: `tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py`
- Modify branch-owned hunks only:
  `tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py`
- Modify branch-owned hunks only:
  `tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py`

**Ownership check:** The nightly operator file is branch-created. The two PR
test files predate this branch, so compare their diffs and remove/change only
the FLA additions made after `7b717fe`.

**Interfaces:**

- Consumes: the Phase6 backend and existing native Triton prefill reference.
- Produces: hardware evidence for fused Phase6 output/state and Qwen3.6-35B
  eager model behavior.

- [ ] **Step 1: Rewrite the nightly operator test around Phase6 only**

Keep test cases for token lengths `1`, `63`, `64`, `65`, and a two-sequence
varlen case `[0, 1, 65]`. Build a strict backend with
`GDNPhase6RuntimeSignature` using `get_fla_gdn_soc()`.

The candidate path is:

```python
backend = FlaGDNPhase6Backend.create(
    mode="fla_npu",
    signature=signature,
    layer_name="smoke.linear_attn",
    pcp_world_size=1,
)
assert backend is not None
assert backend.prepare(q.device)
actual_output, actual_state = backend.prefill(**inputs)
```

The expected path stays test-only and calls the existing vLLM-Ascend native
prefill implementation using the same Q/K normalization, initial-state clear,
metadata, and `chunk_gated_delta_rule`. Implement it in the test file as:

```python
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
```

This helper exists only in the real-NPU test; it must not be imported into
runtime code.

Assert:

```python
assert torch.isfinite(actual_output.float()).all()
assert torch.isfinite(actual_state.float()).all()
assert F.cosine_similarity(
    actual_output.float().flatten(),
    expected_output.float().flatten(),
    dim=0,
).item() >= 0.999
torch.testing.assert_close(actual_output, expected_output, rtol=5e-3, atol=5e-3)
torch.testing.assert_close(actual_state, expected_state, rtol=5e-3, atol=5e-3)
```

Delete small-operator, causal-convolution, and recurrent-decode tests from this
branch-created file.

- [ ] **Step 2: Remove the branch-added 0.8B FLA comparison**

In `test_qwen3_5_0_8b.py`, remove only
`test_qwen3_5_gdn_fla_eager_smoke` and imports that were added solely for that
test. Preserve the upstream MTP/graph test unchanged. The primary integration
validation will use the requested Qwen3.6-35B model instead of treating 0.8B
as acceptance evidence.

- [ ] **Step 3: Retarget the branch-added Qwen3.6 test name and description**

Keep the baseline filename for upstream stability. Rename only the added test:

```python
def test_qwen3_6_gdn_phase6_eager_smoke():
    """Compare strict Phase6 prefill with native GDN on a local Qwen3.6 model."""
```

Continue using:

```python
MODEL = os.environ.get("QWEN36_MODEL_PATH", "Qwen/Qwen3.6-27B")
```

On the validation host set `QWEN36_MODEL_PATH` to the 35B model. Do not rename
the whole upstream file or baseline FIA tests.

- [ ] **Step 4: Run provenance/static checks and commit**

Run locally:

```powershell
git diff 7b717fe...HEAD -- tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py
git diff --check -- tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py
```

Commit:

```bash
git add tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py \
  tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py \
  tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py
git commit -m "test(gdn): focus hardware coverage on Phase6 prefill"
```

---

### Task 7: Replace obsolete documents and restore repository guidance

**Files:**

- Modify: `docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md`
- Verify/preserve: `docs/source/developer_guide/Design_Documents/index.md`
- Create:
  `docs/superpowers/guides/2026-09-05-qwen-gdn-phase6-only-validation-guide-zh.md`
- Preserve:
  `docs/superpowers/specs/2026-09-05-qwen-gdn-phase6-only-refactor-design.md`
- Preserve:
  `docs/superpowers/plans/2026-09-05-qwen-gdn-phase6-only-refactor.md`
- Delete superseded branch-owned documents listed below.
- Restore: `CLAUDE.md`

**Ownership check:** The old GDN documents are feature-branch additions. The
current public design file has pre-existing uncommitted user edits; inspect and
incorporate their intent before replacing links. `CLAUDE.md` existed at
baseline, so restore its exact baseline content instead of deleting it.

**Interfaces:**

- Consumes: final runtime behavior from Tasks 2-6.
- Produces: one authoritative spec, one implementation plan, one Chinese
  validation guide, and one short public developer page.

- [ ] **Step 1: Snapshot the dirty documentation state before editing**

Run locally:

```powershell
git status --short
git diff -- docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md
Get-Content -Raw docs/superpowers/guides/2026-09-01-qwen35-qwen36-a3-fla-gdn-code-integration-guide-zh.md
```

Preserve useful install, public API, logging, and 35B command information in
the replacement guide. Do not silently discard those user-authored details.

- [ ] **Step 2: Write the Chinese Phase6-only validation guide**

The guide must contain these sections in order:

1. Scope and prominent PCP limitation.
2. Clone/install the matched `vllm`, `vllm-ascend`, and
   `flash-linear-attention-npu` repositories.
3. Build the FLA wheel with the correct A2/A3/A5 `FLA_NPU_SOC` value.
4. Confirm OPP packaging and import.
5. Run only `gdn_core_fwd_phase6` direct/operator tests.
6. Run deferred unit tests.
7. Run Qwen3.6-35B strict/native eager comparison.
8. Run optional `FULL_DECODE_ONLY` plus MTP coexistence validation.
9. Capture backend-selection logs and profiler kernel name.
10. Record results separately for A2, A3, and A5.

Include a full logging file that configures both namespaces:

```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "standard": {
      "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
    }
  },
  "handlers": {
    "stderr": {
      "class": "logging.StreamHandler",
      "formatter": "standard",
      "stream": "ext://sys.stderr"
    }
  },
  "loggers": {
    "vllm": {
      "handlers": ["stderr"],
      "level": "INFO",
      "propagate": false
    },
    "vllm_ascend": {
      "handlers": ["stderr"],
      "level": "INFO",
      "propagate": false
    }
  }
}
```

Explain that `VLLM_LOGGING_CONFIG_PATH` replaces vLLM's default dictionary,
which is why both logger namespaces must be present.

Use this first model-validation command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=3,4
export QWEN36_MODEL_PATH=/home/weights/Qwen3.6-35B-A3B
export VLLM_ASCEND_GDN_BACKEND=fla_npu
export VLLM_LOGGING_CONFIG_PATH=/tmp/vllm-logging-config.json

cd /home/z00886386/vllm-ascend
pytest -s -q \
  -o log_cli=true \
  --log-cli-level=INFO \
  tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py \
  -k gdn_phase6_eager_smoke \
  2>&1 | tee /tmp/qwen36-35b-gdn-phase6.log
```

State explicitly that visible physical cards 3 and 4 become logical NPU 0 and
1 inside the process.

- [ ] **Step 3: Rewrite the short public developer page**

The page must say:

- Qwen3.5 and Qwen3.6 share this vLLM-Ascend GDN path;
- the sole FLA symbol is `gdn_core_fwd_phase6` for eligible prefill;
- causal convolution, Q/K `l2norm_fwd`, ordinary decode, MTP decode, and
  native fallback remain vLLM/vLLM-Ascend implementations;
- `l2norm_fwd` is retained because Phase6 expects normalized Q/K and replacing
  it adds no fused-kernel benefit while widening numerical risk;
- A2/A3/A5 use capability mapping, not `is_950()`;
- PCP world size greater than one forces native;
- `FULL_DECODE_ONLY` compatibility does not mean Phase6 itself enters a graph;
- `auto` is the default and `fla_npu` is strict;
- current validation status states: A5 `vllm serve` has been verified; A2 and
  A3 end-to-end validation are pending; ais-bench validation is pending.

Link only to the 2026-09-05 spec, plan, and validation guide.

- [ ] **Step 4: Delete superseded GDN documents**

Delete exactly:

```text
docs/superpowers/plans/2026-08-25-qwen35-qwen36-gdn-a5.md
docs/superpowers/specs/2026-08-25-qwen35-qwen36-gdn-a5-design.md
docs/superpowers/specs/2026-08-29-qwen35-qwen36-gdn-fla-design.md
docs/superpowers/reports/2026-08-29-gdn-fla-a2-a3-a5-change-report.md
docs/superpowers/guides/2026-08-29-qwen-gdn-a2-a3-validation-guide-zh.md
docs/superpowers/guides/2026-09-01-qwen35-qwen36-a3-fla-gdn-code-integration-guide-zh.md
```

- [ ] **Step 5: Restore `CLAUDE.md` to the baseline**

The entire file becomes the exact baseline line:

```markdown
IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.
```

- [ ] **Step 6: Verify links, stale claims, and formatting**

Run locally:

```powershell
rg -n "2026-08-25-qwen|2026-08-29-qwen|2026-09-01-qwen|VLLM_ASCEND_GDN_OP_BACKENDS|FlaGDNAdapter|FlaGDNOperatorDispatcher|six standalone|recurrent_gated_delta_rule.*FLA" docs CLAUDE.md
rg -n "PCP|gdn_core_fwd_phase6|FULL_DECODE_ONLY|A2|A3|A5" docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md docs/superpowers/guides/2026-09-05-qwen-gdn-phase6-only-validation-guide-zh.md
git diff --check -- docs CLAUDE.md
```

Expected: the stale-symbol scan has no current-document matches; the required
scope and PCP scan has matches in both current documents.

- [ ] **Step 7: Commit documentation lifecycle changes**

```bash
git add CLAUDE.md docs/source/developer_guide/Design_Documents \
  docs/superpowers/guides docs/superpowers/plans docs/superpowers/reports \
  docs/superpowers/specs
git commit -m "docs(gdn): replace legacy adapter guidance with Phase6 scope"
```

Before committing, inspect `git diff --cached --name-status` and confirm it
contains only the listed documentation lifecycle files.

---

### Task 8: Perform final static audit and prepare hardware checkpoints

**Files:**

- Verify all files changed by Tasks 1-7.
- Create no additional production file unless a verified static finding
  requires a branch-owned correction.

**Interfaces:**

- Consumes: the completed refactor.
- Produces: a reviewable branch and a precise A2/A3/A5 validation handoff.

- [ ] **Step 1: Verify the final FLA symbol surface**

Run locally:

```powershell
rg -n "fla_npu\.ops" vllm_ascend/ops/gdn.py vllm_ascend/ops/gdn_fla.py vllm_ascend/platform.py
rg -n "GDNOperator|FlaGDNAdapter|FlaGDNOperatorDispatcher|A5GDN|gdn_a5|VLLM_ASCEND_GDN_OP_BACKENDS" vllm_ascend tests docs
rg -n "causal_conv1d|recurrent_gated_delta_rule|chunk_local_cumsum|chunk_scaled_dot_kkt|solve_tri|recompute_w_u_fwd|chunk_gated_delta_rule_fwd_h|chunk_fwd_o" vllm_ascend/ops/gdn_fla.py
```

Expected:

- the only runtime FLA operator symbol is
  `fla_npu.ops.ascendc.gdn_core_fwd_phase6`;
- obsolete adapter/config names have no matches;
- the focused backend has no small-op, convolution, or decode symbols.

- [ ] **Step 2: Verify provenance for every changed production hunk**

Run locally:

```powershell
git diff --name-status 7b717fe...HEAD
git diff 7b717fe...HEAD -- vllm_ascend/ops/gdn.py vllm_ascend/ops/gdn_fla.py vllm_ascend/envs.py vllm_ascend/platform.py vllm_ascend/device/device_config.py vllm_ascend/quantization/utils.py
git log --format="%h %an %ae %s" 7b717fe..HEAD -- vllm_ascend
```

Check each edited production hunk against the branch diff. The only approved
baseline-owned modification must remain the test-only quantization change.

- [ ] **Step 3: Run final local static verification**

```powershell
git diff --check 7b717fe...HEAD
rg -n "<<<<<<<|=======|>>>>>>>" vllm_ascend tests docs CLAUDE.md
git status --short --branch
git log --oneline --decorate -10
```

No Python command is run locally. Report this limitation explicitly rather
than claiming unit or hardware tests passed.

- [ ] **Step 4: Run unit tests on an installed A2/A3/A5 container**

```bash
cd /home/z00886386/vllm-ascend
pytest -q \
  tests/ut/ops/test_gdn_fla.py \
  tests/ut/device/test_device_config.py \
  tests/ut/test_platform.py \
  tests/ut/quantization/test_utils.py
```

Record the pass count, warnings, installed vLLM/vLLM-Ascend/torch/torch-npu
versions, CANN version, `SOC_VERSION`, and FLA wheel version.

- [ ] **Step 5: Run the real-NPU Phase6 comparison on each target family**

```bash
export VLLM_ASCEND_GDN_BACKEND=fla_npu
cd /home/z00886386/vllm-ascend
pytest -s -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py \
  2>&1 | tee /tmp/gdn-phase6-operator.log
```

Expected: token lengths 1/63/64/65 and the varlen case pass for output and
final state. Repeat independently on A2, A3, and A5; do not infer one family's
result from another.

- [ ] **Step 6: Run Qwen3.6-35B eager comparison first**

```bash
export ASCEND_RT_VISIBLE_DEVICES=3,4
export QWEN36_MODEL_PATH=/home/weights/Qwen3.6-35B-A3B
export VLLM_ASCEND_GDN_BACKEND=fla_npu
export VLLM_LOGGING_CONFIG_PATH=/tmp/vllm-logging-config.json

cd /home/z00886386/vllm-ascend
pytest -s -q \
  -o log_cli=true \
  --log-cli-level=INFO \
  tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py \
  -k gdn_phase6_eager_smoke \
  2>&1 | tee /tmp/qwen36-35b-gdn-phase6.log
```

Inspect for:

```text
requested=fla_npu
selected=fla_npu
symbol=fla_npu.ops.ascendc.gdn_core_fwd_phase6
stage=scratch_probe
soc=ascend910b|ascend910_93|ascend950
```

There must be no FLA selection log for causal convolution or recurrent decode.

- [ ] **Step 7: Run optional graph/MTP coexistence validation**

Use the approved serve configuration with
`"cudagraph_mode":"FULL_DECODE_ONLY"` and Qwen3.5 MTP. Verify that prompt
prefill selects Phase6 while graph capture/replay emits no FLA resolve or call.
The profiler must show `ChunkGdnCoreFwd` only during prefill and native decode
kernels during generation.

Do not claim Phase6 graph support from this result; it proves only coexistence
with decode-only capture.

- [ ] **Step 8: Stop at the final review checkpoint**

Summarize:

- commit list;
- diffstat;
- removed symbols/files;
- local static results;
- deferred versus executed tests;
- A2/A3/A5 results separately;
- known PCP and graph limitations.

Do not push, rebase, open a PR, or amend remote branches until the user reviews
this checkpoint and explicitly requests the remote action.
