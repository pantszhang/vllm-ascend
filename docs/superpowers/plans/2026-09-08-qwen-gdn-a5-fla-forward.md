# Qwen GDN A5 FLA Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace eligible Qwen3.5/Qwen3.6 A5 ordinary-Prefill GDN computation with the FLA PR #472 forward interface without changing convolution or Decode paths.

**Architecture:** `gdn.py` remains the single routing point. A focused adapter resolves and probes `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd`, passes vLLM's BSND tensors and V-first state directly, and falls back to native only before live FLA execution. Device capability mapping limits this first increment to Ascend 950.

**Tech Stack:** Python, PyTorch/torch-npu, vLLM-Ascend, FLA AscendC ctypes ACLNN interface, pytest

**Spec:** `docs/superpowers/specs/2026-09-08-qwen-gdn-a5-fla-forward-design.md`

## Global Constraints

- The minimum FLA integration point is merge commit `c254124994ba3d797353b3313dba3c8007f6db1f` from PR #472.
- Only `ascend950` is eligible; A2, A3, 310P, and unknown devices use native Prefill.
- Only ordinary Prefill may call FLA; causal convolution, ordinary Decode, and MTP Decode remain native.
- PCP world size must equal one.
- The accepted contract is BF16 activation, BF16/FP32 state, `K=V=128`, `chunk_size=64`, and `Hv/Hk <= 4`.
- Do not replay a live request through native after FLA execution has started.
- Do not run Python or pytest in the local non-NPU workspace; execute those checks on the A5 host.

---

### Task 1: Specify the A5 interface contract

**Files:**
- Modify: `tests/ut/ops/test_gdn_fla.py`
- Modify: `tests/ut/device/test_device_config.py`
- Modify: `tests/ut/test_platform.py`

**Interfaces:**
- Consumes: `VLLM_ASCEND_GDN_BACKEND` with `auto`, `fla_npu`, or `native`.
- Produces: test contracts for `FlaGDNPrefillBackend`, `GDNRuntimeSignature`, and `GDNPrefillMetadata`.

- [x] **Step 1: Write selection tests**

```python
assert FlaGDNPrefillBackend.create(
    mode="auto", signature=A5_SIGNATURE, layer_name="layer", pcp_world_size=1
) is not None
assert FlaGDNPrefillBackend.create(
    mode="auto", signature=A2_SIGNATURE, layer_name="layer", pcp_world_size=1
) is None
```

- [x] **Step 2: Write the PR #472 call-contract test**

```python
assert captured["q"].shape == (1, 65, 1, 128)
assert captured["options"]["layout"] == "BSND"
assert captured["options"]["state_v_first"] is True
assert captured["options"]["use_exp2"] is True
assert captured["options"]["use_qk_l2norm_in_kernel"] is True
assert captured["options"]["output_a"] is False
```

- [ ] **Step 3: Run the unit tests on A5**

```bash
cd /home/z00886386/vllm-ascend
pytest -q tests/ut/device/test_device_config.py tests/ut/ops/test_gdn_fla.py tests/ut/test_platform.py
```

Expected: all selected tests pass.

### Task 2: Implement A5-only routing and the new FLA adapter

**Files:**
- Modify: `vllm_ascend/device/device_config.py`
- Modify: `vllm_ascend/envs.py`
- Modify: `vllm_ascend/platform.py`
- Modify: `vllm_ascend/ops/gdn_fla.py`
- Modify: `vllm_ascend/ops/gdn.py`

**Interfaces:**
- Consumes: BSND Q/K/V, `[B,T,Hv]` FP32 gate, post-sigmoid beta, V-first state, and prebuilt host metadata.
- Produces: BSND output and V-first final state through `FlaGDNPrefillBackend.prefill(...)`.

- [x] **Step 1: Restrict the device capability map to A5**

```python
_FLA_GDN_SOC_BY_DEVICE_TYPE = {
    AscendDeviceType.A5: "ascend950",
}
```

- [x] **Step 2: Resolve only the new public FLA symbol**

```python
GDN_FWD_SYMBOL = "fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd"

def resolve_chunk_gated_delta_rule_fwd():
    module = importlib.import_module("fla_npu.ops.ascendc")
    return module.chunk_gated_delta_rule_fwd, GDN_FWD_SYMBOL
```

- [x] **Step 3: Pass native layouts directly to FLA**

```python
output, final_state, _, _ = operator(
    q, k, v, g, beta,
    initial_state=state,
    output_final_state=True,
    chunk_size=64,
    cu_seqlens=metadata.cu_seqlens_host,
    chunk_indices=metadata.chunk_indices_host,
    scale=scale,
    use_exp2=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    output_a=False,
    state_v_first=True,
    layout="BSND",
)
```

- [x] **Step 4: Keep native routing unchanged outside ordinary Prefill**

Verify by inspection that `_get_fla_gdn_prefill_backend` is called only inside `if attn_metadata.num_prefills > 0`, while recurrent Decode and speculative Decode still call `npu_recurrent_gated_delta_rule`.

- [ ] **Step 5: Run focused tests on A5**

```bash
cd /home/z00886386/vllm-ascend
pytest -q tests/ut/ops/test_gdn_fla.py
```

Expected: selection, probe caching, contract, failure propagation, and Prefill-only routing pass.

### Task 3: Add real-NPU regression coverage

**Files:**
- Modify: `tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py`
- Modify: `tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py`

**Interfaces:**
- Consumes: an A5-compatible FLA wheel and custom OPP from PR #472 or newer.
- Produces: native-vs-FLA output/state comparison and Qwen3.6-35B end-to-end comparison.

- [x] **Step 1: Cover fixed and variable-length operator cases**

```python
@pytest.mark.parametrize("tokens", [1, 63, 64, 65])
def test_gdn_fla_prefill_matches_native(tokens):
    _assert_fla_matches_native((0, tokens))

def test_gdn_fla_varlen_prefill_matches_native():
    _assert_fla_matches_native((0, 1, 65))
```

- [x] **Step 2: Keep the two-card comparison selectable by the new name**

```bash
pytest -s -q tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py -k gdn_fla_eager_smoke
```

- [ ] **Step 3: Run the operator comparison on A5**

```bash
cd /home/z00886386/vllm-ascend
VLLM_ASCEND_GDN_BACKEND=fla_npu pytest -s -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py
```

Expected: fixed-length and varlen output plus final-state comparisons pass.

- [ ] **Step 4: Run Qwen3.6-35B on two A5 cards**

```bash
cd /home/z00886386/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=3,4 QWEN36_MODEL_PATH=/home/weights/Qwen3.6-35B-A3B VLLM_ASCEND_GDN_BACKEND=fla_npu pytest -s -q -o log_cli=true --log-cli-level=INFO tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py -k gdn_fla_eager_smoke
```

Expected: FLA and native outputs match within the test tolerance, with no native fallback in strict mode.

### Task 4: Publish the authoritative documentation and inspect the profile

**Files:**
- Modify: `docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md`
- Create: `docs/superpowers/specs/2026-09-08-qwen-gdn-a5-fla-forward-design.md`
- Create: `docs/superpowers/guides/2026-09-08-qwen-gdn-a5-fla-forward-validation-guide-zh.md`
- Delete: the superseded 2026-09-05 Phase6-only design, plan, and guide.

**Interfaces:**
- Consumes: the completed implementation and A5 validation results.
- Produces: one public design page, one authoritative spec, and one Chinese runbook.

- [x] **Step 1: Document the exact dependency, layouts, flags, fallback, and PCP boundary**

```text
backend=fla_npu implementation=a5_prepare_pipeline
symbol=fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd soc=ascend950
```

- [x] **Step 2: Remove superseded Phase6-only documents**

Confirm no public document links to `2026-09-05-qwen-gdn-phase6-only-*`.

- [ ] **Step 3: Inspect the A5 profiler trace**

Expected: `ChunkGatedDeltaRuleFwdPrepare`, `ChunkGatedDeltaRuleFwdH`, and `ChunkFwdO` appear; adapter-side Q/K/V transpose, beta cast, and state transpose are absent.

- [ ] **Step 4: Commit after A5 verification**

```bash
git add docs tests vllm_ascend
git commit -m "refactor(gdn): use A5 FLA forward pipeline"
```
