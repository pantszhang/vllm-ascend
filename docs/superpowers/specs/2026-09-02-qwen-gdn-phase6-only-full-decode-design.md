# Qwen GDN Phase6-only 接入与 FULL_DECODE_ONLY 设计

日期：2026-09-02  
状态：设计已确认，待实施

## 1. 背景

当前 vLLM-Ascend 为 Qwen3.5/Qwen3.6 GDN 路径接入了
`flash-linear-attention-npu`（下文简称 FLA）的三个公开接口：

- `causal_conv1d`；
- `gdn_core_fwd_phase6`；
- `recurrent_gated_delta_rule`。

其中 FLA Python 包装通过 ACLNN 两段式接口调用自定义 AscendC 算子。
`causal_conv1d` 和 `recurrent_gated_delta_rule` 当前不具备本项目要求的
图安全接口，导致 Decode 在 ACL Graph capture/replay 场景下必须绕过 FLA。

本设计收缩 FLA 接入边界：只保留普通 Prefill 阶段的融合算子
`fla_npu.ops.ascendc.gdn_core_fwd_phase6`。Decode、MTP、卷积和状态更新继续
使用 vLLM/vLLM-Ascend 原有实现。

与引入 FLA 前相比，最终净变化只有一项：原有 Prefill GDN core 计算链新增
一个可选的 FLA Phase6 融合实现。

## 2. 目标

1. 唯一 FLA 接入点为 `gdn_core_fwd_phase6`。
2. 普通 Prefill 可以使用 FLA Phase6，同时保持 Prefill 在 ACL Graph 外执行。
3. 普通 Decode 和 MTP Decode 始终使用 vLLM-Ascend native 算子，并支持
   `FULL_DECODE_ONLY` ACL Graph。
4. 服务开启 MTP 时不再全局禁用 Prefill Phase6。
5. 配置、日志和 profiler 能准确区分 Phase6 FLA 与 Decode native 路径。
6. 保持 A2、A3、A5 的现有 Phase6 能力判断，不恢复 SoC 硬编码限制。

## 3. 非目标

- 不让 FLA `causal_conv1d` 入图。
- 不让 FLA `recurrent_gated_delta_rule` 入图。
- 不让 `gdn_core_fwd_phase6` 进入 Prefill Full Graph。
- 不修改 FLA 仓的算子实现、ACLNN 接口或 wheel 构建。
- 不修改 PD connector，也不把 PD 分离作为本期验收范围。
- 不接入或保留 FLA 六小算子替换链。
- 不改变 GDN cache 的 shape、dtype、layout、索引或生命周期。

## 4. 接入边界

### 4.1 最终 FLA 接口

唯一对外依赖：

```python
fla_npu.ops.ascendc.gdn_core_fwd_phase6
```

对应 ACLNN 接口：

```text
aclnnGdnCoreFwdPhase6GetWorkspaceSize
aclnnGdnCoreFwdPhase6
```

对应 FLA AscendC 算子为 `ChunkGdnCoreFwd`。

### 4.2 始终使用 native 的功能

| 功能 | 最终实现 |
|---|---|
| Prefill causal convolution | `torch.ops._C_ascend.npu_causal_conv1d_custom` |
| Decode causal convolution | `torch.ops._C_ascend.npu_causal_conv1d_custom` |
| 普通 Decode recurrent | `torch.ops._C_ascend.npu_recurrent_gated_delta_rule` |
| MTP Decode recurrent | vLLM-Ascend 现有 native speculative 路径 |
| `l2norm_fwd` | vLLM 现有实现 |
| Phase6 不可用时的 Prefill | vLLM/vLLM-Ascend 原有 Prefill 计算链 |

这里的 `native` 表示 vLLM/vLLM-Ascend 原有实现，不表示全部为 CANN 内置
算子。

## 5. 执行架构

```text
AscendGatedDeltaNetAttention.forward
|
+-- causal_conv1d
|   `-- 始终调用 vLLM-Ascend native
|
+-- 执行阶段判断
|   |
|   +-- 普通 Prefill
|   |   +-- backend=fla_npu -> FLA gdn_core_fwd_phase6
|   |   +-- backend=auto    -> Phase6 或完整 native Prefill 链
|   |   `-- backend=native  -> 完整 native Prefill 链
|   |
|   +-- 普通 Decode
|   |   `-- vLLM-Ascend native recurrent
|   |
|   `-- MTP/speculative Decode
|       `-- vLLM-Ascend native speculative 路径
|
`-- l2norm/output
    `-- 保持 vLLM 原有实现
```

FLA adapter 不再在整个 GDN forward 开始时统一创建。只有代码已经确定进入
普通 Prefill 分支后，才解析、选择并调用 Phase6。

## 6. MTP 行为

当前 `_get_fla_gdn_adapter()` 使用 `self.num_spec > 0` 全局禁止 FLA。该约束
需要删除，因为它把“服务开启 MTP”和“当前执行阶段是 MTP Decode”错误地
等同起来。

修改后的规则为：

```text
服务开启 MTP + 当前为普通 Prefill
    -> 允许 FLA Phase6

当前为普通 Decode 或 MTP Decode
    -> 始终 native
```

本设计不要求 Phase6 处理 draft token，也不改变 MTP 接受、拒绝和 cache 更新
语义。MTP 只影响 Decode 调度，不影响普通 Prefill 是否可以选择 Phase6。

## 7. ACL Graph 与编译边界

### 7.1 正式支持模式

本期正式支持：

```text
cudagraph_mode=FULL_DECODE_ONLY
```

`PIECEWISE` 不属于本期正式支持或验收范围；它保持现有 native 图行为，本设计
不声明 FLA Phase6 与该模式兼容。

其执行边界为：

```text
Prefill: NONE，图外执行 FLA Phase6
Decode:  FULL，捕获和回放 native GDN Decode
```

Decode 的 warmup、FX 优化、ACL Graph capture 和 replay 都不得导入、解析、
探测或调用任何 FLA causal/recurrent 接口。

### 7.2 FULL 模式

FLA Phase6 本期不支持 Prefill Full Graph：

- 显式 `fla_npu + FULL`：配置阶段报错，提示使用
  `FULL_DECODE_ONLY`；
- `auto + FULL`：Phase6 在捕获前固定为 native，并打印一次警告；
- `native + FULL`：保持现有 native 行为。

不能在 Graph capture 或 replay 期间动态选择 backend。

### 7.3 Npugraph_ex

Ascend 的 `FULL_DECODE_ONLY` 默认还包含 Npugraph_ex FX 优化。阶段路由必须
保证 Decode 编译路径只看到 native Torch Custom Op，不能依赖
`forward_context.capturing` 在执行末端临时回落。

## 8. 配置语义

`VLLM_ASCEND_GDN_BACKEND` 只控制 Prefill core：

| 值 | 行为 |
|---|---|
| `native` | 始终执行原有 Prefill 链，不加载 Phase6 |
| `auto` | Phase6 可用则选择 FLA，否则在执行前固定回落 native |
| `fla_npu` | 强制 Phase6；缺失、不兼容或执行失败时直接报错 |

保留显式按算子配置：

```bash
export VLLM_ASCEND_GDN_BACKEND=native
export VLLM_ASCEND_GDN_OP_BACKENDS=gdn_core_fwd=fla_npu
```

`VLLM_ASCEND_GDN_OP_BACKENDS` 只接受逻辑算子 `gdn_core_fwd` 的 FLA
覆盖。以下 FLA 覆盖均应在配置阶段报错：

- `causal_conv1d=fla_npu`；
- `recurrent_gated_delta_rule=fla_npu`；
- 六小算子中的任意 `operator=fla_npu`。

全局 `VLLM_ASCEND_GDN_BACKEND=fla_npu` 的准确含义为“强制普通 Prefill
使用 FLA Phase6”，不再表示整个 GDN 流程使用 FLA。

## 9. 能力判断与错误处理

Phase6 选择前检查：

- FLA 包和符号可加载；
- SoC 在当前 Phase6 支持范围内；
- activation 为 BF16；
- state dtype、head 数、head dim 和 chunk size 受支持；
- 当前阶段为普通 Prefill；
- 当前执行不在 ACL Graph capture/replay 中；
- PCP 等现有不兼容条件未触发。

严格 `fla_npu` 模式下，任一条件不满足都必须失败。`auto` 模式只能在
Phase6 执行前回落，不能在状态可能已被部分修改后重试 native 链。

若能够在模型 warmup 阶段完成 Phase6 probe，应尽早确定 backend；无法提前
构造真实 shape 时，最迟在第一次普通 Prefill 调用前确定并缓存选择。

## 10. 日志与可观测性

Phase6 选择日志应包含：

```text
op=gdn_core_fwd
backend=fla_npu|native
symbol=fla_npu.ops.ascendc.gdn_core_fwd_phase6
soc=<soc>
dtype=<dtype>
state_dtype=<dtype>
mtp_enabled=<true|false>
cudagraph_mode=<mode>
```

Decode 只记录 native 和 ACL Graph eligibility，不再记录 FLA
`causal_conv1d` 或 FLA `recurrent_gated_delta_rule` 的选择、probe 或回落。

Profiler 中：

- Prefill 应出现 `ChunkGdnCoreFwd`/`GdnCoreFwdPhase6`；
- Decode 应出现 vLLM-Ascend native causal/recurrent；
- Decode 应出现 ACL Graph replay；
- Decode 不应出现 FLA causal/recurrent；
- Phase6 不应出现在 Decode Graph 中。

## 11. 代码范围

### `vllm_ascend/ops/gdn.py`

- causal convolution 固定走 native；
- 仅在普通 Prefill 分支获取 Phase6 adapter；
- 删除 `num_spec > 0` 的全局 FLA 禁用；
- 普通 Decode 和 MTP Decode 固定走 native recurrent；
- 不在 Decode 编译/捕获路径触碰 FLA dispatcher。

### `vllm_ascend/ops/gdn_fla.py`

- 收敛为 Phase6-only dispatcher/adapter；
- 删除 FLA causal/recurrent resolver、包装、probe 和 warmup；
- 删除六小算子的 FLA 替换能力；
- native fallback 作为完整 Prefill 链处理；
- 对外只允许 `gdn_core_fwd` 选择 FLA。

建议使用明确命名，例如 `FlaGDNPrefillAdapter` 和
`FlaGDNPhase6Dispatcher`，兼容别名仅在确有外部调用者时保留。

### `vllm_ascend/platform.py`

- 只在 Phase6 可能被选择时提前加载 FLA OPP；
- 增加 `FULL` 与严格 Phase6 的配置冲突检查；
- `FULL_DECODE_ONLY` 不因 Phase6 被禁用。

### 测试与文档

- 删除已取消的 FLA causal/recurrent 和六小算子替换测试；
- 新增 Phase6-only、MTP 和图模式路由测试；
- 更新设计文档、部署指南、配置和日志说明。

本期不修改 `flash-linear-attention-npu` 仓。

## 12. 测试设计

### 12.1 单元测试

1. `native`、`auto`、`fla_npu` 的 Phase6 选择。
2. 仅 `gdn_core_fwd=fla_npu` 为合法 FLA 覆盖。
3. `self.num_spec=3` 时普通 Prefill 仍选择 Phase6。
4. 普通 Decode、MTP Decode、capture 和 replay 不调用 FLA resolver。
5. causal convolution 始终 native。
6. Decode recurrent 始终 native。
7. `fla_npu + FULL_DECODE_ONLY` 合法。
8. `fla_npu + FULL` 报错。
9. `auto + FULL` 固定使用 native 并产生一次警告。
10. 严格模式缺符号、错误 dtype 或 probe 失败时不回落。

### 12.2 FLA 单算子验证

只验证 `gdn_core_fwd_phase6`，覆盖目标 SoC、BF16、chunk size 64、实际
head shape、varlen、initial state 和 final state。FLA causal/recurrent 不属于
本期接入验收。

### 12.3 第一阶段模型验证

模型：`/home/weights/Qwen3.6-35B-A3B`。  
设备：物理卡 3、4；容器内映射为逻辑卡 0、1。  
并行：TP2。  
功能：MTP=3、`FULL_DECODE_ONLY`。

分别运行：

```text
native 基线
FLA Phase6 实验组
```

使用相同 seed、prompt、sampling 参数和输入/输出长度，对比 token、日志、
状态稳定性和 profiler。

### 12.4 最终模型验证

使用 Qwen3.5-397B-A17B W4A4 的生产启动命令：

- TP4；
- Expert Parallel；
- MTP=3；
- `FULL_DECODE_ONLY`；
- Prefix Cache；
- Async Scheduling；
- 多流 shared expert；
- FlashComm1；
- torch profiler。

最终验收要求：

1. 服务正常启动并完成请求；
2. MTP 不挂起；
3. 普通 Prefill 确认调用 Phase6；
4. Decode 确认只调用 native GDN 算子；
5. ACL Graph 成功 capture/replay；
6. 输出与 native 基线一致或满足既定一致性标准；
7. 无 NaN/Inf、状态污染或 Prefix Cache 回归；
8. 记录 TTFT、TPOT、MTP acceptance rate 和 Phase6 耗时。

## 13. 验收结论定义

只有同时满足以下条件，才能宣称本功能完成：

```text
FLA 接入点数量 = 1
接入点 = gdn_core_fwd_phase6
MTP 开启时 Prefill Phase6 生效
Decode 使用 native
FULL_DECODE_ONLY capture/replay 成功
Qwen3.6-35B TP2 首测通过
Qwen3.5-397B TP4+EP 最终测试通过
```

Qwen3.6-35B 通过但 397B 尚未完成时，只能报告“阶段验证通过”，不能报告
生产配置验收完成。
