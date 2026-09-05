# Qwen GDN Phase6-Only Refactor Design

## Status and authority

This document is the authoritative design for the Qwen3.5/Qwen3.6 FLA GDN
integration in vLLM-Ascend. It supersedes the earlier A5-only, multi-operator,
and three-operator designs on this branch.

The implementation targets Qwen GDN inference on Ascend A2, A3, and A5. The
only operator supplied by `flash-linear-attention-npu` is the fused prefill
operator:

```text
fla_npu.ops.ascendc.gdn_core_fwd_phase6
```

FLA causal convolution, recurrent decode, and the six standalone prefill
stages are outside the integration scope.

## Goals

- Use `gdn_core_fwd_phase6` for eligible Qwen3.5/Qwen3.6 prefill batches.
- Preserve the existing vLLM-Ascend causal convolution, ordinary decode, MTP
  decode, and native prefill implementations.
- Support A2, A3, and A5 through semantic device capability detection.
- Keep `auto`, `fla_npu`, and `native` as the complete backend policy.
- Make backend selection, safe fallback, and execution failures observable.
- Remove the obsolete generic multi-operator dispatcher and compatibility API.
- Preserve the previously validated `FULL_DECODE_ONLY` coexistence behavior
  without claiming that Phase6 itself supports graph capture.
- Retain the Dynamic MX quantization compile-safety fix required by the
  previously validated graph configuration.

## Non-goals

- Replacing `causal_conv1d` with an FLA operator.
- Replacing ordinary or MTP recurrent decode with an FLA operator.
- Calling FLA implementations of `chunk_local_cumsum`,
  `chunk_scaled_dot_kkt`, `solve_tri`, `recompute_w_u_fwd`,
  `chunk_gated_delta_rule_fwd_h`, or `chunk_fwd_o`.
- Capturing `gdn_core_fwd_phase6` in an ACL Graph.
- Enabling the FLA Phase6 path when PCP world size is greater than one.
- Changing upstream native GDN algorithms, cache ownership, metadata
  construction, or MTP token handling.

## Provenance constraint

The refactor may directly modify or remove only code introduced or changed by
the branch commits authored by `pantszhang` or by the associated Claude Code
work. The authoritative ownership boundary is the diff from baseline commit
`7b717fefb3f6a47c374585e7ba2da176a045707a`, not the rebased line blame alone.

For every implementation task:

1. Inspect `git diff 7b717fe...HEAD -- <file>`.
2. Confirm the target hunk was added or changed by this branch.
3. Preserve baseline native code whenever deleting a branch-added wrapper.
4. Stop and request a decision before changing a baseline-owned hunk.

One explicit exception is approved: the existing upstream file
`tests/ut/quantization/test_utils.py` may receive the minimum changes required
to test the retained `get_current_vllm_config_or_none()` behavior.

`CLAUDE.md` already existed at the baseline. The branch-added expansion must be
removed by restoring the baseline one-line file; the upstream file itself must
not be deleted.

## Supported execution matrix

The PCP limitation is a release-blocking constraint and must remain prominent
in user-facing documentation and selection logs.

| Dimension | Supported behavior |
| --- | --- |
| A2 | Phase6-eligible, FLA SoC `ascend910b` |
| A3 | Phase6-eligible, FLA SoC `ascend910_93` |
| A5 | Phase6-eligible, FLA SoC `ascend950` |
| Other Ascend devices | Native GDN only |
| DP | Supported |
| TP | Supported subject to the Phase6 shape contract |
| EP | Does not alter GDN backend selection |
| PCP world size = 1 | Phase6-eligible |
| PCP world size > 1 | Phase6 prohibited; native prefill with an explicit log |
| Ordinary decode | Native causal convolution and native recurrent GDN |
| MTP/spec decode | Native causal convolution and native recurrent GDN |
| MTP prompt prefill | Phase6-eligible under the normal prefill rules |
| Eager prefill | Phase6-eligible |
| `FULL_DECODE_ONLY` | Compatible: Phase6 prefill stays outside the graph |
| Graph modes capturing prefill | Unsupported and not claimed |

## Backend policy

`VLLM_ASCEND_GDN_BACKEND` is the only GDN FLA configuration variable. Its
default remains `auto`.

### `auto`

On A2/A3/A5 with PCP world size one, attempt to resolve and scratch-probe the
Phase6 operator. Resolve, eligibility, or scratch-probe failure selects the
existing native prefill implementation. The fallback is logged once per
runtime signature.

### `fla_npu`

Require the Phase6 operator. An ineligible runtime, missing symbol, or failed
scratch probe raises an actionable startup/first-prefill error.

### `native`

Do not import or resolve FLA. All GDN phases use the existing vLLM-Ascend
implementation.

`VLLM_ASCEND_GDN_OP_BACKENDS` and all per-operator override parsing are
removed. A single replaceable operator does not justify a per-operator policy.

## Runtime architecture

`vllm_ascend/ops/gdn.py` remains the owner of model routing, metadata, cache
updates, causal convolution, decode, MTP, and native prefill. The FLA module is
a narrow optional prefill backend.

```text
AscendGatedDeltaNetAttention._forward_core
|
+-- causal convolution ---------------------- existing native operator
+-- ordinary/MTP recurrent decode ----------- existing native operator
`-- prefill
    |-- eligible Phase6 backend ------------- gdn_core_fwd_phase6
    `-- no eligible Phase6 backend ---------- existing native prefill
```

`vllm_ascend/ops/gdn_fla.py` is reduced to these responsibilities:

- parse the global backend mode;
- represent a Phase6 runtime signature and the required host metadata;
- validate Phase6 eligibility;
- lazily resolve `fla_npu.ops.ascendc.gdn_core_fwd_phase6`;
- scratch-probe once per process, device, and runtime signature;
- normalize Phase6 tensor layouts and arguments;
- execute the real prefill call;
- emit selection, fallback, and failure logs.

The target public types are:

```text
GDNBackendMode
GDNPhase6RuntimeSignature
GDNPhase6PrefillMetadata
FlaGDNPhase6Backend
```

The following obsolete abstractions are removed:

```text
GDNOperator
GDNOperatorSelection
FlaGDNOperatorDispatcher
FlaGDNAdapter
A5GDNAdapter
A5GDNOperatorDispatcher
run_gdn_decode_pipeline
six-stage FLA normalizers and composition
FLA causal convolution and recurrent resolvers
solve_tri-specific debug logging
stateful runtime probes
```

The compatibility module `vllm_ascend/ops/gdn_a5.py` is removed. It was added
only on this branch, has no repository consumers, exposes internals through a
wildcard import, and retains the obsolete A5-only API name.

## Eligibility and fallback

Eligibility is evaluated before a real request calls Phase6:

1. `native` mode exits without importing FLA.
2. The device maps to A2, A3, or A5 through `get_fla_gdn_soc()`.
3. PCP world size equals one.
4. Activation dtype is BF16 and the runtime head/state contract is supported.
5. The Phase6 public symbol resolves.
6. A scratch probe succeeds for the runtime signature.

In `auto` mode, failures in steps 2-6 select native prefill. In strict
`fla_npu` mode, they raise. A failure from the real Phase6 invocation always
logs a full exception and propagates; the same request is never retried with
native code because an asynchronous NPU error may have invalidated the stream
or context.

The native fallback is the existing `gdn.py` prefill path. The FLA module must
not duplicate the native six-stage pipeline.

## Phase6 data flow

The Phase6 backend receives normalized Qwen prefill tensors and the existing
SSM state slice. It performs only the transformations required by the public
FLA contract:

1. Apply the existing vLLM Q/K L2 normalization exactly once.
2. Preserve grouped-value-attention head counts; Phase6 handles its required
   head mapping internally.
3. Convert Q/K/V to the Phase6 head-first layout.
4. Clone and clear entries without an initial state, then convert the state
   tail layout expected by Phase6.
5. Convert host sequence/chunk metadata to the public FLA argument form.
6. Pass beta as FP32 as required by the Phase6 operator definition.
7. Convert output and final state back to the layout/dtype expected by
   `gdn.py`.
8. Let `gdn.py` perform the existing cache assignment after successful return.

No causal-convolution state or recurrent-decode state is owned or mutated by
the Phase6 backend.

## Graph and MTP boundary

Phase6 is an eager prefill operator in this integration. The design does not
claim that it can be captured in an ACL Graph.

`FULL_DECODE_ONLY` remains compatible because prompt prefill executes outside
the decode graph. Decode graph tracing and replay must not resolve, probe,
warm up, or call FLA. Moving backend creation into the prefill branch removes
the current need for a forward-wide FLA warmup guard.

MTP does not select an FLA decode backend. An MTP request may use Phase6 for
its prompt prefill; all speculative causal convolution and recurrent GDN calls
remain native.

## Logging

Selection is logged once per process/device/runtime signature. Logs include
the requested and selected backend, Phase6 symbol, SoC, PCP world size, dtype,
state dtype, head counts, head dimensions, chunk size, and decision stage.

Expected categories are:

- INFO: Phase6 selected;
- INFO: native selected by explicit `native`, unsupported device, or PCP;
- WARNING: `auto` fallback after resolve, validation, or scratch-probe failure;
- ERROR with traceback: strict selection failure or real execution failure.

Logs include tensor shape, dtype, device, stride, and metadata lengths but do
not include tensor values. A real execution failure is never followed by a
native retry.

vLLM's built-in logging configuration configures the `vllm` namespace but not
the independent `vllm_ascend` namespace. The validation guide must provide a
complete `VLLM_LOGGING_CONFIG_PATH` JSON that configures both namespaces; the
custom JSON replaces rather than extends the built-in logging dictionary.

## Dynamic MX quantization compatibility

The branch retains the compile-safe use of
`get_current_vllm_config_or_none()` in
`vllm_ascend/quantization/utils.py`. The comment must state that the failure
was first observed with Qwen GDN `FULL_DECODE_ONLY`, while the helper is shared
by all A5 Dynamic MX quantization paths.

A2/A3 return the default algorithm before this lookup because they do not
advertise `DYNAMIC_MX_QUANT_SCALE_ALG_ONE`. On A5, an explicit config or
forward-context value keeps model-specific behavior. Only the absence of every
config context falls back to algorithm zero instead of raising during tracing.

The approved upstream-test exception updates the old mock target in
`tests/ut/quantization/test_utils.py` and adds coverage for the no-context
fallback. No other upstream quantization logic is modified.

## Test strategy

Local development must not run Python because the local environment has no
usable Python/NPU stack. Local verification is limited to Git and text/static
checks such as `git diff --check`, conflict-marker scans, obsolete-symbol
scans, documentation-link scans, and provenance audits.

Tests are still authored for later execution in an A2/A3/A5 environment:

- backend mode parsing and default `auto` behavior;
- A2/A3/A5 SoC mapping and unsupported-device fallback;
- PCP world-size rejection;
- BF16 and Phase6 runtime eligibility;
- symbol resolution, scratch-probe caching, and strict/auto behavior;
- Phase6 argument and layout normalization;
- real execution failure propagation without native retry;
- causal convolution, ordinary decode, and MTP decode never resolving FLA;
- MTP prompt prefill remains Phase6-eligible;
- Dynamic MX optional-config behavior;
- real-NPU Phase6 output/state comparison with native prefill;
- Qwen3.6-35B eager serve validation;
- optional `FULL_DECODE_ONLY` and MTP coexistence validation.

Standalone FLA tests for the six small operators, FLA causal convolution, and
FLA recurrent decode are not acceptance requirements for this integration.

## Documentation lifecycle

After this design and its implementation plan exist, remove these superseded
branch-owned documents:

```text
docs/superpowers/plans/2026-08-25-qwen35-qwen36-gdn-a5.md
docs/superpowers/specs/2026-08-25-qwen35-qwen36-gdn-a5-design.md
docs/superpowers/specs/2026-08-29-qwen35-qwen36-gdn-fla-design.md
docs/superpowers/reports/2026-08-29-gdn-fla-a2-a3-a5-change-report.md
docs/superpowers/guides/2026-08-29-qwen-gdn-a2-a3-validation-guide-zh.md
docs/superpowers/guides/2026-09-01-qwen35-qwen36-a3-fla-gdn-code-integration-guide-zh.md
```

Create one replacement Chinese validation guide focused on Phase6 and update
`docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md` to reference
only the authoritative current design and guide.

## Acceptance criteria

- The only FLA operator symbol referenced by vLLM-Ascend GDN runtime code is
  `fla_npu.ops.ascendc.gdn_core_fwd_phase6`.
- `VLLM_ASCEND_GDN_OP_BACKENDS` no longer exists in runtime code or current
  documentation.
- Causal convolution and every decode path call the existing native operators
  directly.
- A2/A3/A5 are eligible through semantic device mapping; there is no A5-only
  routing guard.
- PCP world size greater than one always selects native prefill and logs why.
- `auto` safely falls back only before live execution; strict/live failures
  propagate with actionable logs.
- Native prefill remains the sole fallback and is not reimplemented in the FLA
  module.
- Superseded documents and compatibility aliases are removed.
- Every changed production hunk is branch-owned, except the explicitly
  approved quantization-test adjustment.
- A2/A3/A5 device validation commands and expected logs are documented, with
  results clearly marked as executed or pending.
