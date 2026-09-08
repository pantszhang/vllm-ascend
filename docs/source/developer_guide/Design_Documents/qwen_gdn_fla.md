# Qwen GDN integration with flash-linear-attention-npu

Qwen3.5 and Qwen3.6 share the same vLLM-Ascend GDN implementation. The
current integration replaces only eligible A5 Prefill computation through one
stable FLA entry point:

```text
fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd
```

The authoritative documents are:

- [A5 FLA forward design](../../../superpowers/specs/2026-09-08-qwen-gdn-a5-fla-forward-design.md)
- [Chinese validation guide](../../../superpowers/guides/2026-09-08-qwen-gdn-a5-fla-forward-validation-guide-zh.md)

## Scope

Only Ascend 950 is currently eligible. A2, A3, 310P, and unknown device
families stay on the native path. A2/A3 support is intentionally deferred.

The A5 path requires BF16 activations, BF16 or FP32 state, `K=V=128`,
`chunk_size=64`, `Hv % Hk == 0`, `Hv/Hk <= 4`, and PCP world size one.

vLLM-Ascend calls one public FLA interface. With `use_exp2=true` and
`use_qk_l2norm_in_kernel=true`, FLA internally dispatches
`ChunkGatedDeltaRuleFwdPrepare`, `ChunkGatedDeltaRuleFwdH`, and
`ChunkFwdO`. vLLM-Ascend does not call the six historical standalone
preparation operators.

The integration does not replace causal convolution, ordinary Decode, or
MTP/speculative Decode. Those operations continue to use vLLM-Ascend native
operators.

## Layout contract

The A5 call directly consumes vLLM layouts:

- Q/K/V: BSND, `[B,T,H,D]`;
- recurrent state: V-first, `[N,Hv,V,K]`;
- output: BSND, `[B,T,Hv,V]`;
- final state: V-first, `[N,Hv,V,K]`.

Q/K L2 normalization is performed inside the FLA Prepare operator. Beta is
already sigmoid-transformed by vLLM-Ascend, so the FLA beta-sigmoid option
remains disabled. `output_a=false` avoids writing an unused intermediate.

## Runtime policy

`VLLM_ASCEND_GDN_BACKEND` accepts:

- `auto` (default): use FLA only after eligibility, symbol resolution, and a
  scratch probe succeed; otherwise use native Prefill;
- `fla_npu`: require the A5 FLA path and raise an attributed error when it is
  unavailable or ineligible;
- `native`: do not import or call FLA for GDN.

The fallback boundary is before a live request. A live FLA execution failure
is logged and propagated; the stateful request is not replayed with native
Prefill.

> **PCP limitation:** PCP world size greater than one uses native Prefill in
> `auto` mode and is rejected in strict `fla_npu` mode.

FLA is imported before vLLM-Ascend custom-OPP bootstrap so both vendor OPP
directories are visible before kernel-manager indexing. Each package keeps its
own vendor directory; similarly named `libcust_opapi.so` files must not be
copied over one another. The FLA Python package, op-api library, kernels, and
OPP must come from the same compatible build because the new interface changes
the ACLNN ABI.

Successful selection logs include:

```text
backend=fla_npu implementation=a5_prepare_pipeline
symbol=fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd soc=ascend950
```
