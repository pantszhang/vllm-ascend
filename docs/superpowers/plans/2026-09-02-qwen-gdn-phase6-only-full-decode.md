# Qwen GDN Phase6-only FULL_DECODE_ONLY Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Qwen3.5/Qwen3.6 的 FLA 接入收敛为仅普通 Prefill 使用 `gdn_core_fwd_phase6`，同时允许开启 MTP 的 Decode 使用 vLLM-Ascend native 箚子进入 `FULL_DECODE_ONLY` ACL Graph。

**Architecture:** `AscendGatedDeltaNetAttention` 的卷积和所有 Decode 分支固定使用 native；只有确定进入普通 Prefill 分支后才创建 Phase6 专用 adapter。FLA dispatcher 只解析 `fla_npu.ops.ascendc.gdn_core_fwd_phase6`，`auto` 可以在执行前回落完整 native Prefill 链，严格 `fla_npu` 模式失败即报错。

**Tech Stack:** Python 3.11、PyTorch/torch-npu、vLLM、vLLM-Ascend、ACL Graph、npugraph_ex、pytest、FLA ACLNN/AscendC Phase6。

**Spec:** `docs/superpowers/specs/2026-09-02-qwen-gdn-phase6-only-full-decode-design.md`

## Global Constraints

- 唯一允许选择 `fla_npu` 的逻辑算子是 `gdn_core_fwd`，其符号固定为 `fla_npu.ops.ascendc.gdn_core_fwd_phase6`。
- `causal_conv1d`、`recurrent_gated_delta_rule`、`l2norm_fwd` 和六小算子不得继续作为 FLA 替换点。
- 普通 Decode、MTP Decode、ACL Graph warmup/capture/replay 不得导入、解析、probe 或调用 FLA causal/recurrent。
- 删除 `self.num_spec > 0` 对整个 FLA Prefill adapter 的禁用；MTP=3 时普通 Prefill 仍可选择 Phase6。
- `fla_npu + FULL_DECODE_ONLY` 合法；`fla_npu + FULL` 必须报错；`auto + FULL` 必须在 Prefill 执行前固定为 native。
- `auto` 只能在 Phase6 修改状态前回落；严格 `fla_npu` 不得静默回落。
- 保持 A2、A3、A5 的能力判断，不增加 `is_950()` 等单 SoC 门禁。
- 不修改 `flash-linear-attention-npu` 仓，不修改 PD connector，不宣称 `PIECEWISE` 兼容。
- 不覆盖工作区已有的 `docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md` 修改或未跟踪指南；修改前先检查 diff，提交时逐文件暂存。
- 当前 Windows 工作区没有可用 Python；本地只做静态检查。pytest 和模型测试在装有 torch-npu/FLA 的 A5 环境执行。

---

## File Structure

- Modify: `vllm_ascend/ops/gdn_fla.py` — Phase6-only backend 配置、符号解析、probe、adapter 和 Prefill pipeline。
- Modify: `vllm_ascend/ops/gdn.py` — 按执行阶段路由；native convolution/native Decode；普通 Prefill 延迟获取 Phase6 adapter。
- Modify: `vllm_ascend/platform.py` — FLA OPP 预加载条件和 FULL/FULL_DECODE_ONLY 配置校验。
- Modify: `tests/ut/ops/test_gdn_fla.py` — Phase6-only dispatcher、MTP Prefill 和 Decode 隔离测试。
- Modify: `tests/ut/test_platform.py` — OPP 预加载和图模式配置测试。
- Modify carefully: `docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md` — 用户文档中的最终 Phase6-only 语义；保留已有未提交内容。
- Create after hardware runs: `docs/superpowers/reports/2026-09-02-qwen-gdn-phase6-only-validation.md` — 只记录真实执行命令和结果，不预填成功结论。

---

### Task 1: 收敛 backend 配置为 Phase6-only

**Files:**
- Modify: `vllm_ascend/ops/gdn_fla.py:75-109`
- Modify: `vllm_ascend/ops/gdn_fla.py:1453-1498`
- Test: `tests/ut/ops/test_gdn_fla.py:61-114`

**Interfaces:**
- Consumes: `VLLM_ASCEND_GDN_BACKEND` 和 `VLLM_ASCEND_GDN_OP_BACKENDS` 字符串。
- Produces: `parse_gdn_backend_config(mode: str, operator_overrides: str) -> GDNBackendConfig`；只有 `GDNOperator.GDN_CORE_FWD` 可显式选择 `FLA_NPU`。

- [ ] **Step 1: 将配置测试改成 Phase6-only，并先运行得到失败**

在 `tests/ut/ops/test_gdn_fla.py` 保留全局模式测试，将按算子测试改成：

```python
def test_parse_gdn_backend_config_accepts_phase6_override():
    config = parse_gdn_backend_config("native", "gdn_core_fwd=fla_npu")
    assert config.mode is GDNBackendMode.NATIVE
    assert config.mode_for(GDNOperator.GDN_CORE_FWD) is GDNBackendMode.FLA_NPU


@pytest.mark.parametrize(
    "override",
    [
        "causal_conv1d=fla_npu",
        "recurrent_gated_delta_rule=fla_npu",
        "chunk_local_cumsum=fla_npu",
        "chunk_scaled_dot_kkt=fla_npu",
        "solve_tri=fla_npu",
        "recompute_w_u_fwd=fla_npu",
        "chunk_gated_delta_rule_fwd_h=fla_npu",
        "chunk_fwd_o=fla_npu",
    ],
)
def test_parse_gdn_backend_config_rejects_non_phase6_fla_override(override):
    with pytest.raises(ValueError, match="only gdn_core_fwd may use fla_npu"):
        parse_gdn_backend_config("native", override)
```

Run on A5:

```bash
cd /home/z00886386/vllm-ascend
pytest -q tests/ut/ops/test_gdn_fla.py \
  -k 'parse_gdn_backend_config_accepts_phase6_override or parse_gdn_backend_config_rejects_non_phase6_fla_override'
```

Expected: the rejection cases fail because current parser still accepts the old FLA operators.

- [ ] **Step 2: 实现最小配置约束**

保留 `GDNOperator` 中 native Prefill pipeline 使用的内部标识，但新增唯一可替换集合：

```python
_FLA_REPLACEABLE_OPERATORS = frozenset({GDNOperator.GDN_CORE_FWD})
```

在解析完 `operator` 和 `backend` 后增加：

```python
if (
    backend is GDNBackendMode.FLA_NPU
    and operator not in _FLA_REPLACEABLE_OPERATORS
):
    raise ValueError(
        f"Invalid GDN operator backend override {raw_entry!r}: "
        "only gdn_core_fwd may use fla_npu."
    )
```

删除 `_STAGE1_REPLACEMENTS`、`_STAGE1_NATIVE_ONLY` 以及 A5-only 兼容别名中不再有调用者的部分；如果 `rg` 证明别名仍被外部测试导入，则保留一版弃用别名并在后续任务统一改名。

- [ ] **Step 3: 运行配置测试和静态引用检查**

Run on A5:

```bash
pytest -q tests/ut/ops/test_gdn_fla.py -k 'parse_gdn_backend_config'
```

Expected: all selected tests pass.

Run locally or on A5:

```bash
rg -n '_STAGE1_REPLACEMENTS|_STAGE1_NATIVE_ONLY|A5GDNAdapter|A5GDNOperatorDispatcher' \
  vllm_ascend tests
```

Expected: no unexpected production callers remain.

- [ ] **Step 4: Commit**

```bash
git add vllm_ascend/ops/gdn_fla.py tests/ut/ops/test_gdn_fla.py
git commit -m "refactor(gdn): restrict FLA selection to phase6"
```

---

### Task 2: 将 FLA dispatcher/adapter 缩减为 Phase6 Prefill

**Files:**
- Modify: `vllm_ascend/ops/gdn_fla.py:1-1450`
- Test: `tests/ut/ops/test_gdn_fla.py:116-680`

**Interfaces:**
- Consumes: Task 1 的 `GDNBackendConfig` 和 `GDNOperator.GDN_CORE_FWD`。
- Produces: `FlaGDNPhase6Dispatcher.select(...) -> GDNOperatorSelection` 和 `FlaGDNPrefillAdapter.prefill(...) -> tuple[torch.Tensor, torch.Tensor]`；`run_gdn_prefill_pipeline(...)` 保持输出布局契约。

- [ ] **Step 1: 先写 Phase6 唯一符号和 strict validation 测试**

先把测试文件导入改为：

```python
import importlib
from unittest.mock import Mock, patch
```

新增/改写测试：

```python
def test_phase6_dispatcher_resolves_only_gdn_core(monkeypatch):
    imported = []

    def fake_import(name):
        imported.append(name)
        return SimpleNamespace(gdn_core_fwd_phase6=lambda *args, **kwargs: None)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    operator, symbol = resolve_fla_operator(GDNOperator.GDN_CORE_FWD)
    assert callable(operator)
    assert symbol == "fla_npu.ops.ascendc.gdn_core_fwd_phase6"
    assert imported == ["fla_npu.ops.ascendc"]


def test_strict_phase6_validation_reports_missing_symbol(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_fla.resolve_fla_operator",
        Mock(side_effect=AttributeError("gdn_core_fwd_phase6")),
    )
    with pytest.raises(RuntimeError, match="gdn_core_fwd"):
        FlaGDNPrefillAdapter(
            parse_gdn_backend_config("fla_npu", ""),
            SIGNATURE,
            layer_name="model.layers.0.linear_attn",
            is_supported_soc=True,
        )
```

Run on A5:

```bash
pytest -q tests/ut/ops/test_gdn_fla.py \
  -k 'phase6_dispatcher_resolves_only_gdn_core or strict_phase6_validation_reports_missing_symbol'
```

Expected: fail until the old generic adapter and symbol table are removed/renamed.

- [ ] **Step 2: 删除不再允许的 FLA 入口**

从 `gdn_fla.py` 删除：

- `log_solve_tri_debug`；
- FLA causal/recurrent/six-stage symbol mapping；
- `_resolve_fla_recurrent_operator`；
- `_validate_causal_conv_probe_state`；
- `FlaGDNAdapter.causal_conv1d()`；
- `FlaGDNAdapter.decode()`；
- `run_gdn_decode_pipeline()`；
- causal/recurrent/six-stage 的 normalizer、probe 和 warmup 逻辑。

将唯一符号表改成：

```python
_FLA_OPERATOR_PATHS = {
    GDNOperator.GDN_CORE_FWD: (
        "fla_npu.ops.ascendc",
        "gdn_core_fwd_phase6",
    ),
}
```

将类名改为 `FlaGDNPhase6Dispatcher` 和 `FlaGDNPrefillAdapter`，同步生产代码与测试导入。除非 `rg` 找到仓外兼容要求，否则不保留暗示全 GDN 接入的旧别名。

- [ ] **Step 3: 让 Prefill 小算子链固定为 native**

`_prefill_operators()` 继续构造 native pipeline 所需函数，但对六小算子只调用 `select_native_only()`，不得传入 FLA normalizer。只有 `GDN_CORE_FWD` 调用 Phase6 resolver：

```python
core_selection = self.dispatcher.select(
    GDNOperator.GDN_CORE_FWD,
    self.signature,
    native=native_gdn_core_unused,
    native_symbol="gdn-core-native-chain",
    fla_resolver=self._normalized_fla_resolver(
        GDNOperator.GDN_CORE_FWD,
        fla_gdn_core,
    ),
)
if core_selection.backend is GDNBackendMode.FLA_NPU:
    selected[GDNOperator.GDN_CORE_FWD] = self._logged_operator(
        GDNOperator.GDN_CORE_FWD,
        core_selection,
        native=native_gdn_core_unused,
        native_symbol="gdn-core-native-chain",
        phase="prefill",
        stateful=False,
    )
```

`run_gdn_prefill_pipeline()` 的 fused 分支布局转换保持不变，native 时 `operators.get(GDN_CORE_FWD)` 返回空并走原有完整链。

- [ ] **Step 4: 将 warmup 缩减为 Phase6-only**

`FlaGDNPrefillAdapter.warmup()` 只构造 chunk size 64 的 q/k/v/g/beta/state 和 metadata，然后调用一次 `self.prefill()`。删除 `conv_weight`、`conv_bias` 参数和两次 causal convolution probe。签名改为：

```python
def warmup(self, *, device: torch.device, dtype: torch.dtype, state_dtype: torch.dtype) -> None:
    ...
```

日志改为：

```text
GDN FLA Phase6 warmup completed
```

- [ ] **Step 5: 删除旧测试并运行 Phase6/prefill 测试**

删除只验证以下能力的测试：

- causal adapter 参数映射；
- causal prefill/decode 分别 probe；
- functional recurrent state copy；
- FLA mixed Decode/Prefill 合并；
- 六小算子 FLA 选择。

保留并调整：

- dispatcher auto/strict/native；
- Phase6 output contract；
- Prefill pipeline layout、initial/final state；
- A2/A3/A5 routing；
- adapter/dispatcher cache。

Run on A5:

```bash
pytest -q tests/ut/ops/test_gdn_fla.py
```

Expected: all tests pass and test names no longer宣称 causal/recurrent FLA coverage.

- [ ] **Step 6: Commit**

```bash
git add vllm_ascend/ops/gdn_fla.py tests/ut/ops/test_gdn_fla.py
git commit -m "refactor(gdn): keep only FLA phase6 prefill adapter"
```

---

### Task 3: 按阶段路由 GDN，并允许 MTP 服务使用 Prefill Phase6

**Files:**
- Modify: `vllm_ascend/ops/gdn.py:20-114`
- Modify: `vllm_ascend/ops/gdn.py:239-276`
- Modify: `vllm_ascend/ops/gdn.py:350-749`
- Test: `tests/ut/ops/test_gdn_fla.py:682-end`

**Interfaces:**
- Consumes: Task 2 的 `FlaGDNPrefillAdapter`、`FlaGDNPhase6Dispatcher` 和 `GDNPrefillMetadata`。
- Produces: `_get_fla_gdn_prefill_adapter(activation, state) -> FlaGDNPrefillAdapter | None`；仅普通 Prefill 分支调用。

- [ ] **Step 1: 写 MTP 不再禁用 Phase6 adapter 的失败测试**

扩展 `_fake_gdn_layer()` 支持 `num_spec`，新增：

```python
@pytest.mark.parametrize("num_spec", [0, 3])
def test_phase6_prefill_adapter_is_available_with_mtp(monkeypatch, num_spec):
    monkeypatch.setenv("VLLM_ASCEND_GDN_BACKEND", "fla_npu")
    monkeypatch.delenv("VLLM_ASCEND_GDN_OP_BACKENDS", raising=False)
    layer = _fake_gdn_layer()
    layer.num_spec = num_spec
    with (
        patch("vllm_ascend.ops.gdn.get_fla_gdn_soc", return_value="ascend950"),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch.object(FlaGDNPrefillAdapter, "_validate_strict_symbols"),
    ):
        adapter = AscendGatedDeltaNetAttention._get_fla_gdn_prefill_adapter(
            layer,
            torch.zeros((1, 128), dtype=torch.bfloat16),
            torch.float32,
        )
    assert isinstance(adapter, FlaGDNPrefillAdapter)
```

Run on A5:

```bash
pytest -q tests/ut/ops/test_gdn_fla.py -k phase6_prefill_adapter_is_available_with_mtp
```

Expected: `num_spec=3` case fails under the current global guard.

- [ ] **Step 2: 重命名并收窄 adapter getter**

将 `_get_fla_gdn_adapter` 改为 `_get_fla_gdn_prefill_adapter`：

```python
def _get_fla_gdn_prefill_adapter(
    self,
    activation: torch.Tensor,
    state: torch.Tensor | torch.dtype,
) -> FlaGDNPrefillAdapter | None:
    fla_soc = get_fla_gdn_soc()
    if fla_soc is None or get_pcp_group().world_size != 1:
        return None
    ...
```

删除 `getattr(self, "num_spec", 0) > 0`。缓存名同步改为 `_fla_gdn_prefill_adapter_cache`，runtime signature 的 `mtp` 设置为：

```python
mtp=getattr(self, "num_spec", 0) > 0
```

该字段只用于日志与选择缓存，不再禁用 Phase6。

- [ ] **Step 3: causal convolution 全部固定 native**

删除 `core_attn()` 开头的统一 `fla_adapter` 创建。删除 Prefill 和 Decode 中的：

```python
if fla_adapter is not None:
    fla_adapter.causal_conv1d(...)
```

两处均无条件保留现有：

```python
torch.ops._C_ascend.npu_causal_conv1d_custom(...)
```

MTP 的 spec causal 分支本来就是 native，保持不变。

- [ ] **Step 4: 普通 Decode 和 mixed Decode 全部固定 native**

删除两处 `fla_adapter.decode(...)` 分支，无条件执行现有：

```python
query_decode = l2norm_fwd(query_decode)
key_decode = l2norm_fwd(key_decode)
core_attn_out_decode = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(...).unsqueeze(0)
```

以及普通非 spec Decode 的等价 native 调用。MTP spec recurrent 分支保持原样。

- [ ] **Step 5: 只在普通 Prefill 分支获取并 warmup Phase6**

在 `if attn_metadata.num_prefills > 0:` 且完成 mixed decode token 切分后创建 adapter：

```python
fla_prefill_adapter = None
if spec_sequence_masks is None and not getattr(forward_context, "capturing", False):
    fla_prefill_adapter = self._get_fla_gdn_prefill_adapter(
        query_non_spec,
        ssm_state,
    )
```

若 adapter 非空，在真实 Phase6 调用前执行一次缓存 warmup：

```python
fla_prefill_adapter.warmup(
    device=query_non_spec.device,
    dtype=query_non_spec.dtype,
    state_dtype=ssm_state.dtype,
)
```

删除 `forward()` 顶部对所有请求执行的旧 FLA warmup，确保 Decode FX/capture 路径不触碰 FLA。

- [ ] **Step 6: 添加 Decode 隔离测试**

在现有 fake metadata forward 测试基础上增加三种断言：

```python
@pytest.mark.parametrize("capturing,spec", [(False, False), (True, False), (True, True)])
def test_decode_never_requests_fla_prefill_adapter(monkeypatch, capturing, spec):
    getter = Mock(side_effect=AssertionError("Decode touched FLA Phase6"))
    monkeypatch.setattr(
        AscendGatedDeltaNetAttention,
        "_get_fla_gdn_prefill_adapter",
        getter,
    )
    # 使用现有 fake layer/metadata 构造普通或 MTP Decode，执行 core_attn。
    # native causal/recurrent mock 返回契约正确的 tensor。
    _run_fake_decode(capturing=capturing, speculative=spec)
    getter.assert_not_called()
```

`_run_fake_decode` 从现有 `test_fla_mixed_decode_prefill_routes_and_merges_outputs`
的 fake layer、`GDNAttentionMetadata` 和 `_C_ascend` mock 提取，固定接口为：

```python
def _run_fake_decode(*, capturing: bool, speculative: bool) -> torch.Tensor:
    """Execute core_attn with one decode token and return core_attn_out."""
```

普通 Decode metadata 必须设置 `num_prefills=0`、`num_decodes=1`、
`spec_sequence_masks=None`；MTP metadata 使用现有 spec fixture，并设置一个 accepted
token。`ForwardContext.capturing` 直接使用参数值。不要仅测试 helper；三种参数都必须
经过 `core_attn()`，且保持现有 fixture 的 tensor shape。

- [ ] **Step 7: 运行 GDN 单测**

Run on A5:

```bash
pytest -q tests/ut/ops/test_gdn_fla.py
```

Expected: all tests pass；日志测试只包含 Phase6 FLA selection。

- [ ] **Step 8: Commit**

```bash
git add vllm_ascend/ops/gdn.py tests/ut/ops/test_gdn_fla.py
git commit -m "feat(gdn): route MTP decode natively under phase6 prefill"
```

---

### Task 4: 增加 FULL_DECODE_ONLY 配置保护

**Files:**
- Modify: `vllm_ascend/platform.py:74-110`
- Modify: `vllm_ascend/platform.py:976-1040`
- Modify: `vllm_ascend/platform.py` call site near `check_and_update_config()`
- Test: `tests/ut/test_platform.py:26-117`
- Test: `tests/ut/test_platform.py` compilation-mode tests

**Interfaces:**
- Consumes: 最终 `vllm_config.compilation_config.cudagraph_mode` 和 GDN backend 环境变量。
- Produces: `_validate_fla_gdn_graph_mode(vllm_config) -> None`；严格 FULL 报错，auto FULL 记录 native policy，FULL_DECODE_ONLY 不变。

- [ ] **Step 1: 写图模式矩阵失败测试**

新增：

```python
@pytest.mark.parametrize("mtp", [False, True])
def test_fla_phase6_allows_full_decode_only(monkeypatch, mtp):
    config = TestNPUPlatform.mock_vllm_config()
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    config.speculative_config = MagicMock() if mtp else None
    monkeypatch.setenv("VLLM_ASCEND_GDN_BACKEND", "fla_npu")
    monkeypatch.delenv("VLLM_ASCEND_GDN_OP_BACKENDS", raising=False)
    _validate_fla_gdn_graph_mode(config)


def test_strict_fla_phase6_rejects_full_graph(monkeypatch):
    config = TestNPUPlatform.mock_vllm_config()
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    monkeypatch.setenv("VLLM_ASCEND_GDN_BACKEND", "fla_npu")
    with pytest.raises(ValueError, match="FULL_DECODE_ONLY"):
        _validate_fla_gdn_graph_mode(config)


def test_auto_phase6_disables_fla_for_full_graph(monkeypatch, caplog):
    config = TestNPUPlatform.mock_vllm_config()
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    monkeypatch.setenv("VLLM_ASCEND_GDN_BACKEND", "auto")
    _validate_fla_gdn_graph_mode(config)
    assert config.additional_config["gdn_phase6_prefill_backend"] == "native"
    assert "FULL" in caplog.text
```

Run on A5:

```bash
pytest -q tests/ut/test_platform.py -k 'fla_phase6 or phase6_disables_fla'
```

Expected: fail because helper and native policy do not exist.

- [ ] **Step 2: 实现配置校验并保存 worker 可见策略**

新增 `_validate_fla_gdn_graph_mode(vllm_config)`。使用最终解析后的
`CUDAGraphMode`，将 `auto + FULL` 的结果写入可随 `vllm_config` 进入 worker
的 `additional_config`：

```python
def _validate_fla_gdn_graph_mode(vllm_config: VllmConfig) -> None:
    from vllm.config.compilation import CUDAGraphMode
    from vllm_ascend.ops.gdn_fla import (
        GDNBackendMode,
        GDNOperator,
        parse_gdn_backend_config,
    )

    config = parse_gdn_backend_config(
        os.environ.get("VLLM_ASCEND_GDN_BACKEND", "auto"),
        os.environ.get("VLLM_ASCEND_GDN_OP_BACKENDS", ""),
    )
    phase6_mode = config.mode_for(GDNOperator.GDN_CORE_FWD)
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    additional_config = vllm_config.additional_config
    if additional_config is None:
        additional_config = {}
        vllm_config.additional_config = additional_config
    additional_config["gdn_phase6_prefill_backend"] = "configured"

    if cudagraph_mode.mixed_mode() != CUDAGraphMode.FULL:
        return
    if phase6_mode is GDNBackendMode.FLA_NPU:
        raise ValueError(
            "FLA gdn_core_fwd_phase6 does not support Prefill FULL graph; "
            "use cudagraph_mode=FULL_DECODE_ONLY or GDN backend=native."
        )
    if phase6_mode is GDNBackendMode.AUTO:
        additional_config["gdn_phase6_prefill_backend"] = "native"
        logger.warning(
            "Disabling FLA gdn_core_fwd_phase6 because Prefill FULL graph was requested; "
            "using the native GDN prefill chain."
        )
```

在 `_update_compilation_modes()` 完成最终 mode 调整后调用该 helper。不要修改用户的环境变量。

- [ ] **Step 3: 让 Prefill getter 读取最终策略**

在 `gdn.py` 使用：

```python
from vllm.config import get_current_vllm_config

phase6_policy = (
    get_current_vllm_config().additional_config or {}
).get("gdn_phase6_prefill_backend", "configured")
if phase6_policy == "native":
    return None
```

严格模式的 FULL 冲突在 platform 阶段已经报错；getter 保留防御性检查，避免直接构造 layer 的测试绕过平台校验。

同时给 `GDNRuntimeSignature` 增加：

```python
cudagraph_mode: str = "NONE"
```

getter 使用最终配置填写：

```python
cudagraph_mode = str(
    get_current_vllm_config().compilation_config.cudagraph_mode
)
```

Phase6 selection 日志输出 `mtp=<bool>`、`acl_graph=False` 和
`cudagraph_mode=<value>`。`acl_graph` 对 Phase6 必须恒为 false；如果它变成 true，
测试应失败而不是继续调用 FLA。

- [ ] **Step 4: 收窄 FLA OPP 预加载判断**

`_import_fla_npu_before_custom_opp()` 只在以下情况加载 FLA：

```text
global backend in {auto, fla_npu}
或 gdn_core_fwd=fla_npu
```

`causal_conv1d=fla_npu` 等非法 override 不得作为预加载理由；它们由配置 parser 报错。
实现时复用 `parse_gdn_backend_config()`，只检查：

```python
config.mode_for(GDNOperator.GDN_CORE_FWD) in {
    GDNBackendMode.AUTO,
    GDNBackendMode.FLA_NPU,
}
```

- [ ] **Step 5: 运行平台测试**

Run on A5:

```bash
pytest -q tests/ut/test_platform.py -k 'FlaGDNPreload or fla_phase6 or phase6_disables_fla'
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_ascend/platform.py vllm_ascend/ops/gdn.py tests/ut/test_platform.py tests/ut/ops/test_gdn_fla.py
git commit -m "feat(gdn): allow phase6 with full-decode ACL graph"
```

---

### Task 5: 更新权威文档和完成软件级回归

**Files:**
- Modify carefully: `docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md`
- Modify: `docs/superpowers/specs/2026-09-02-qwen-gdn-phase6-only-full-decode-design.md` only if implementation changed an approved interface
- Test: `tests/ut/ops/test_gdn_fla.py`
- Test: `tests/ut/test_platform.py`

**Interfaces:**
- Consumes: Tasks 1-4 的最终配置、类名、日志和图模式行为。
- Produces: 用户可执行的 Phase6-only 配置说明；没有三算子 FLA 接入的过期声明。

- [ ] **Step 1: 在编辑权威文档前保护用户改动**

```bash
git diff -- docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md
git status --short
```

Expected: 看清当前 3 行用户修改和未跟踪指南；后续只使用小范围 patch，不覆盖整文件。

- [ ] **Step 2: 更新权威文档的最终语义**

文档必须明确写出：

```text
唯一 FLA 接口：gdn_core_fwd_phase6
Prefill causal_conv1d：native
普通/MTP Decode recurrent：native
MTP 不再全局禁用 Prefill Phase6
正式支持：FULL_DECODE_ONLY
本期不支持：FLA Phase6 + FULL/PIECEWISE
严格 fla_npu 失败即报错；auto 执行前回落
```

删除或改写任何“FLA causal/recurrent/六小算子已接入”的表格、日志示例和测试声明。

- [ ] **Step 3: 做静态残留扫描**

```bash
rg -n 'fla_adapter\.causal_conv1d|fla_adapter\.decode|_resolve_fla_recurrent_operator|fla_npu.*recurrent_gated_delta_rule|fla_npu.*causal_conv1d' \
  vllm_ascend tests docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md
```

Expected: production/tests/权威文档无已取消接入的残留；历史 spec/plan 不纳入此断言。

```bash
rg -n 'gdn_core_fwd_phase6|FULL_DECODE_ONLY|num_spec' \
  vllm_ascend/ops/gdn.py vllm_ascend/ops/gdn_fla.py \
  docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md
```

Expected: Phase6、MTP 和图模式边界都有明确实现或说明。

- [ ] **Step 4: 在 A5 运行软件级回归**

```bash
cd /home/z00886386/vllm-ascend
pytest -q tests/ut/ops/test_gdn_fla.py tests/ut/test_platform.py
```

Expected: all tests pass；记录总通过数和 warnings，不把 warning 当成失败隐藏。

- [ ] **Step 5: Commit**

只暂存本任务实际修改；不要用 `git add docs/`：

```bash
git add docs/source/developer_guide/Design_Documents/qwen_gdn_fla.md
git commit -m "docs(gdn): document phase6-only graph routing"
```

如果该文件仍包含无法与用户改动安全拆分的内容，先不提交它，在交付说明中列为保留的用户改动。

---

### Task 6: A5 上先验证 Qwen3.6-35B TP2

**Files:**
- Create after real run: `docs/superpowers/reports/2026-09-02-qwen-gdn-phase6-only-validation.md`

**Interfaces:**
- Consumes: Tasks 1-5 的分支和已安装的 FLA Phase6 wheel；模型 `/home/weights/Qwen3.6-35B-A3B`。
- Produces: native 与 FLA Phase6 的日志、输出和 profiler 证据；决定是否允许进入 397B 验证。

- [ ] **Step 1: 确认代码、wheel 和设备映射**

```bash
cd /home/z00886386/vllm-ascend
git branch --show-current
git log -5 --oneline
python3 -m pip show flash-linear-attention-npu
npu-smi info
```

Expected: 分支为 `gdn-phase6-only-full-decode`；FLA wheel 可见；物理卡 3、4 空闲。该命令只在 A5 环境运行，不在当前 Windows 工作区运行。

- [ ] **Step 2: 启动 native 基线服务**

```bash
export ASCEND_RT_VISIBLE_DEVICES=3,4
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_GDN_BACKEND=native
unset VLLM_ASCEND_GDN_OP_BACKENDS

vllm serve /home/weights/Qwen3.6-35B-A3B \
  --served-model-name qwen36 \
  --host 0.0.0.0 \
  --port 8890 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --seed 1024 \
  --max-num-seqs 8 \
  --max-model-len 32768 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --all2all-backend allgather_reducescatter \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
  --distributed-executor-backend mp \
  2>&1 | tee /tmp/qwen36-phase6-native.log
```

Expected: 服务启动；MTP 不挂；Decode 完成 ACL Graph capture/replay。

- [ ] **Step 3: 保存固定请求的 native 输出**

从另一终端执行：

```bash
curl -s http://127.0.0.1:8890/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36","prompt":"请用三点解释线性注意力。","max_tokens":128,"temperature":0,"seed":1024}' \
  | tee /tmp/qwen36-native-output.json
```

Expected: HTTP 成功并生成 128 token 以内的完整响应。停止服务后再启动实验组，避免两个服务争用卡。

- [ ] **Step 4: 启动 FLA Phase6 实验服务**

复用 Step 2 命令，只替换：

```bash
export VLLM_ASCEND_GDN_BACKEND=fla_npu
unset VLLM_ASCEND_GDN_OP_BACKENDS
```

并添加 profiler：

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/qwen36-phase6-profile","torch_profiler_with_stack":false}'
```

日志保存到：

```text
/tmp/qwen36-phase6-fla.log
```

Expected: 日志包含 `op=gdn_core_fwd backend=fla_npu`、`gdn_core_fwd_phase6`、`mtp_enabled=true` 和 `FULL_DECODE_ONLY`。

- [ ] **Step 5: 发送相同请求并核对证据**

```bash
curl -s http://127.0.0.1:8890/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36","prompt":"请用三点解释线性注意力。","max_tokens":128,"temperature":0,"seed":1024}' \
  | tee /tmp/qwen36-fla-output.json

grep -E 'gdn_core_fwd|gdn_core_fwd_phase6|ChunkGdnCoreFwd|ACLGraph|capture|replay|MTP|fallback|ERROR' \
  /tmp/qwen36-phase6-fla.log
```

Expected:

- Prefill 使用 Phase6；
- Decode capture/replay 成功；
- 没有 FLA causal/recurrent selection；
- 没有 strict fallback；
- 请求正常结束；
- temperature=0 输出与 native 基线一致；若 token 不完全一致，先停止并定位数值误差，不进入 397B。

- [ ] **Step 6: 写入真实验证报告并提交**

报告必须包含：commit、wheel 版本、CANN/torch-npu 版本、SoC、完整命令、pytest 数量、日志片段、输出比较和 profiler kernel 名称。未执行项目写“未执行”，失败项目写真实错误，禁止预填“通过”。

```bash
git add docs/superpowers/reports/2026-09-02-qwen-gdn-phase6-only-validation.md
git commit -m "docs(gdn): record qwen36 phase6 graph validation"
```

---

### Task 7: 使用 Qwen3.5-397B 生产配置完成最终验收

**Files:**
- Modify after real run: `docs/superpowers/reports/2026-09-02-qwen-gdn-phase6-only-validation.md`

**Interfaces:**
- Consumes: Task 6 已通过的 commit 和用户提供的 Qwen3.5-397B 完整启动配置。
- Produces: TP4+EP+W4A4+MTP3+FULL_DECODE_ONLY 的最终验收结论。

- [ ] **Step 1: 运行 native 基线**

使用用户提供的完整生产命令，只设置：

```bash
export VLLM_ASCEND_GDN_BACKEND=native
unset VLLM_ASCEND_GDN_OP_BACKENDS
```

profiler 目录使用独立路径，例如：

```text
/mnt/share/l00951447/profiling/qwen35_native_graph_in_128k_out_1k_prefixcache_0.9
```

保存固定 prompt、temperature=0、seed=1024 的输出和服务日志。

- [ ] **Step 2: 运行 FLA Phase6 实验组**

使用完全相同的模型、设备、并行、MTP、图、Prefix Cache、Async Scheduling 和请求，只替换：

```bash
export VLLM_ASCEND_GDN_BACKEND=fla_npu
unset VLLM_ASCEND_GDN_OP_BACKENDS
```

使用用户指定 profiler 目录：

```text
/mnt/share/l00951447/profiling/qwen35_fla_graph_in_128k_out_1k_prefixcache_0.9
```

- [ ] **Step 3: 核对最终验收矩阵**

逐项记录：

```text
服务启动
MTP 不挂起
Prefill Phase6 命中
Decode native causal/recurrent
ACL Graph capture/replay
输出一致性
NaN/Inf
多轮状态污染
Prefix Cache 命中回归
TTFT
TPOT
MTP acceptance rate
Phase6 kernel 耗时
```

任何一项失败都必须保留原始错误和复现命令，不能用“其他功能问题”掩盖。

- [ ] **Step 4: 更新报告和提交**

```bash
git add docs/superpowers/reports/2026-09-02-qwen-gdn-phase6-only-validation.md
git commit -m "docs(gdn): record qwen35 production graph validation"
```

- [ ] **Step 5: 最终分支检查**

```bash
git status --short --branch
git log --oneline --decorate -10
git diff --check HEAD~5..HEAD
```

Expected: 只剩实施前已经存在、明确属于用户的未提交文档；本功能代码、测试和报告均已提交。
