# Qwen GDN A5 FLA forward design

## Goal

Use the FLA `chunk_gated_delta_rule_fwd` public interface for ordinary Qwen
GDN Prefill on Ascend 950. Keep causal convolution, ordinary Decode,
MTP/speculative Decode, and all non-A5 devices on existing native paths.

This design supersedes the Phase6-only A2/A3/A5 design dated 2026-09-05.

## FLA dependency contract

The integration targets the interface introduced by
flash-linear-attention-npu PR #472 (merge commit
`c254124994ba3d797353b3313dba3c8007f6db1f`). vLLM-Ascend resolves only:

```text
fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd
```

It does not fall back to the removed `gdn_core_fwd_phase6` Python symbol.
Requiring the new symbol prevents an old FLA wheel from being mistaken for a
successful PR #472 validation.

On A5 the call sets:

```text
use_exp2=true
use_qk_l2norm_in_kernel=true
use_gate_in_kernel=false
use_beta_sigmoid_in_kernel=false
allow_neg_eigval=false
output_a=false
state_v_first=true
layout=BSND
```

FLA consequently composes `ChunkGatedDeltaRuleFwdPrepare`,
`ChunkGatedDeltaRuleFwdH`, and `ChunkFwdO` inside one public ACLNN forward
interface. The six historical standalone preparation operators are not
resolved or dispatched by vLLM-Ascend.

## Eligibility

The FLA backend is constructed only when all conditions hold:

- SoC is `ascend950`;
- activation dtype is BF16;
- state dtype is BF16 or FP32;
- key and value head counts are positive;
- `Hv` is divisible by `Hk` and `Hv/Hk <= 4`;
- `K=128`, `V=128`, and `chunk_size=64`;
- PCP world size is one.

A2 and A3 are outside the current scope. In `auto` mode they use native
Prefill. Strict `fla_npu` mode reports an eligibility error.

## Data flow

```text
Qwen ordinary Prefill
  -> native causal_conv1d and cache update
  -> native fused_gdn_gating produces natural-log g and sigmoid(beta)
  -> FlaGDNPrefillBackend.prepare (once per process/device/signature)
  -> FLA chunk_gated_delta_rule_fwd
       layout=BSND, state_v_first=true
       internal Q/K L2Norm
       internal Prepare -> FwdH -> FwdO
  -> write returned V-first final state to the vLLM SSM cache
```

Q/K/V and final output remain BSND. Initial and final states remain V-first.
No adapter-side QKV transpose, state transpose, output transpose, external
Q/K L2Norm, or beta FP32 cast is performed.

## Failure handling

`auto` may select native only before a live FLA request, when eligibility,
symbol resolution, or the scratch probe fails. Strict `fla_npu` turns those
conditions into attributed errors. Once live execution begins, exceptions are
logged with implementation, symbol, SoC, layer, tensor metadata, and host
metadata lengths, then propagated without retry.

## Graph and decode boundary

The FLA call is constructed only in the ordinary Prefill branch. Decode and
MTP Decode continue through the native recurrent operator. `FULL_DECODE_ONLY`
therefore does not capture this Prefill interface. This design makes no claim
that the FLA interface itself is graph-capturable.

## Verification

Unit tests specify A5-only selection, PR #472 shape/options, preflight caching,
fallback, live-error propagation, and Prefill-only routing. Real-NPU tests
compare fixed-length and varlen A5 results with native Prefill. Final acceptance
also requires Qwen3.6-35B serve comparison and profiler inspection.
