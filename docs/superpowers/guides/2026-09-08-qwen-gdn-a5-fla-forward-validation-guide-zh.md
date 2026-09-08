# Qwen GDN A5 FLA Forward 验证指南

## 1. 范围

本版本只验证 Ascend 950。A2/A3 暂时使用 native GDN Prefill。FLA 只替换
普通 Prefill GDN 核心计算；causal conv、Decode 和 MTP Decode 仍使用
vLLM-Ascend 算子。

目标 FLA 接口：

```text
fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd
```

FLA wheel、OPP、op-api 动态库和 kernel 必须由包含 PR #472 合入提交
`c254124994ba3d797353b3313dba3c8007f6db1f` 的同一源码版本编译。

## 2. 基础检查

```bash
cd /home/z00886386/vllm-ascend
git branch --show-current
git rev-parse HEAD

python3 - <<'PY'
from fla_npu.ops import ascendc
print(ascendc.chunk_gated_delta_rule_fwd)
assert callable(ascendc.chunk_gated_delta_rule_fwd)
PY
```

如果只有 `gdn_core_fwd_phase6` 而没有新符号，说明安装的仍是旧 FLA wheel。

## 3. vLLM-Ascend 单元测试

```bash
cd /home/z00886386/vllm-ascend
pytest -q \
  tests/ut/device/test_device_config.py \
  tests/ut/ops/test_gdn_fla.py \
  tests/ut/test_platform.py
```

## 4. A5 真机算子对比

```bash
export VLLM_ASCEND_GDN_BACKEND=fla_npu
export VLLM_LOGGING_LEVEL=INFO

cd /home/z00886386/vllm-ascend
pytest -s -q \
  -o log_cli=true \
  --log-cli-level=INFO \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py \
  2>&1 | tee /tmp/qwen-gdn-a5-fla-op.log
```

覆盖 token 长度 `1/63/64/65` 和变长序列 `(0,1,65)`，同时对比 output
和 final state。

## 5. Qwen3.6-35B E2E

```bash
export ASCEND_RT_VISIBLE_DEVICES=3,4
export QWEN36_MODEL_PATH=/home/weights/Qwen3.6-35B-A3B
export VLLM_ASCEND_GDN_BACKEND=fla_npu
export VLLM_LOGGING_LEVEL=INFO

cd /home/z00886386/vllm-ascend
pytest -s -q \
  -o log_cli=true \
  --log-cli-level=INFO \
  tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py \
  -k gdn_fla_eager_smoke \
  2>&1 | tee /tmp/qwen36-35b-gdn-a5-fla.log
```

文件名中的 `27b` 是历史测试名称；`QWEN36_MODEL_PATH` 会覆盖实际模型为
35B。

成功选择日志必须包含：

```text
backend=fla_npu implementation=a5_prepare_pipeline
symbol=fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd soc=ascend950
```

严格模式下不应出现 `backend=native`。Profiler 中 A5 新路径应能看到
`ChunkGatedDeltaRuleFwdPrepare`、`ChunkGatedDeltaRuleFwdH` 和 `ChunkFwdO`，
不应再以旧 `gdn_core_fwd_phase6` Python 入口调用。

## 6. 需重点观察的性能项

- vLLM adapter 不再单独发起 Q/K/V transpose。
- vLLM adapter 不再单独发起 Q/K L2Norm。
- vLLM adapter 不再对 beta 执行 `.float()`。
- initial/final state 不再交换 V/K 维度。
- 对比 Cast 与 GDN 内部第一个任务之间的无算子间隙。

## 7. 回落语义

`auto` 只能在真实请求进入 FLA 前，因 eligibility、resolve 或 scratch probe
失败而回落 native。真实 FLA 执行一旦开始，错误会打印完整定位信息并向上
抛出，不会用 native 重放已经修改状态的请求。
