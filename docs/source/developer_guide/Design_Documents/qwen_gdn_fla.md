# Qwen GDN integration with flash-linear-attention-npu

Qwen3.5 and Qwen3.6 share the same vLLM-Ascend GDN implementation. This
integration replaces only eligible Prefill core computation with the fused
FLA entry point:

```text
fla_npu.ops.ascendc.gdn_core_fwd_phase6
```

The authoritative documents are:

- [Phase6-only design](../../../superpowers/specs/2026-09-05-qwen-gdn-phase6-only-refactor-design.md)
- [Implementation plan](../../../superpowers/plans/2026-09-05-qwen-gdn-phase6-only-refactor.md)
- [Chinese validation guide](../../../superpowers/guides/2026-09-05-qwen-gdn-phase6-only-validation-guide-zh.md)

## Scope

The FLA operator is eligible on A2, A3, and A5 through a device-capability
mapping rather than an `is_950()` gate:

| Hardware | FLA build target | vLLM-Ascend device family |
| --- | --- | --- |
| A2 | `ascend910b` | `AscendDeviceType.A2` |
| A3 | `ascend910_93` | `AscendDeviceType.A3` |
| A5 | `ascend950` | `AscendDeviceType.A5` |

The installed FLA wheel and custom OPP must match the target SoC. 310P and
unknown device families stay on the native path.

The current integration deliberately does not replace:

- causal convolution or convolution-cache updates;
- Q/K `l2norm_fwd`;
- ordinary Decode;
- MTP/speculative Decode;
- the native Prefill fallback.

Those operations continue to use the existing vLLM or vLLM-Ascend
implementations. The six standalone stages represented inside Phase6 are not
individually selected or called by this integration.

`l2norm_fwd` remains unchanged because Phase6 expects normalized Q/K inputs.
Replacing the already suitable vLLM implementation would add no fused-kernel
benefit and would widen the numerical compatibility surface.

## Runtime policy

`VLLM_ASCEND_GDN_BACKEND` accepts:

- `auto` (default): use Phase6 only after eligibility, symbol resolution, and
  a scratch probe succeed; otherwise select the unchanged native Prefill path;
- `fla_npu`: require Phase6 and raise an attributed error if eligibility,
  resolution, or the scratch probe fails;
- `native`: do not import or call FLA for GDN.

The safe fallback boundary is before a live Phase6 request. A failure during a
real Phase6 call is logged with symbol, SoC, layer, tensor metadata, and host
metadata lengths, then propagated. A stateful request is never replayed through
native Prefill after partial Phase6 execution.

> **PCP limitation:** PCP world size greater than one always uses native
> Prefill in `auto` mode. Strict `fla_npu` mode rejects this configuration.

## Execution and graph boundary

The Phase6 call is constructed only inside the Prefill branch. It consumes
normalized Q/K, performs the fused GDN core, returns output plus final SSM
state, and writes the converted final state back to the existing cache.

MTP does not disable Phase6 for prompt Prefill. MTP Decode still calls the
existing native recurrent operator. `FULL_DECODE_ONLY` is therefore compatible
with the integration: eager Prefill may use Phase6 while Decode capture/replay
contains only the native Decode path. This coexistence does **not** mean that
`gdn_core_fwd_phase6` itself supports ACL Graph capture.

FLA is imported before vLLM-Ascend custom-OPP bootstrap so both vendor OPP
directories are visible before kernel-manager indexing. Each package keeps its
own vendor directory; similarly named `libcust_opapi.so` files must not be
copied over one another.

## Logging and validation status

Successful strict or automatic selection includes:

```text
backend=fla_npu
symbol=fla_npu.ops.ascendc.gdn_core_fwd_phase6
soc=ascend910b|ascend910_93|ascend950
```

An automatic native decision identifies `stage=eligibility`, `stage=resolve`,
or `stage=scratch_probe` and includes the reason. There must be no FLA selection
log for causal convolution or recurrent Decode.

Current validation status:

- A5 `vllm serve`: verified;
- A2 end-to-end: pending;
- A3 end-to-end: pending;
- ais-bench: pending.

See the Chinese validation guide for installation, direct Phase6 tests,
Qwen3.6-35B comparison, logging, profiler checks, and the per-SoC result table.
