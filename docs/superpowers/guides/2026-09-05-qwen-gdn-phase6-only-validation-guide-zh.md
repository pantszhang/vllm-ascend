# Qwen3.5/Qwen3.6 GDN Phase6-only 安装与验证指南

## 1. 范围与强约束

本次 vLLM-Ascend 接入的 FLA 算子只有：

```text
fla_npu.ops.ascendc.gdn_core_fwd_phase6
```

它只用于符合条件的 Prefill GDN 核心。以下部分固定使用 vLLM 或
vLLM-Ascend 现有实现：

- `causal_conv1d` 和卷积缓存更新；
- Q/K `l2norm_fwd`；
- 普通 Decode；
- MTP/speculative Decode；
- Phase6 启用前的 native Prefill 回落。

> **PCP 强约束：当前 Phase6 接入只支持 PCP world size 1。** `auto` 模式在
> PCP world size 大于 1 时记录原因并使用 native Prefill；严格 `fla_npu`
> 模式直接拒绝启动。不要把 PCP 多 rank 的 native 成功误记为 Phase6 成功。

Qwen3.5 和 Qwen3.6 共用这套 GDN 接入代码。A2、A3、A5 通过设备能力映射
进入同一接口，不使用 `is_950()` 限制。

## 2. 获取并安装三个仓库

以下示例统一使用 `/home/z00886386`。容器内应先加载与 PyTorch/torch-npu
匹配的 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
mkdir -p /home/z00886386
cd /home/z00886386

git clone https://github.com/vllm-project/vllm.git
git clone git@github.com:pantszhang/vllm-ascend.git
git clone --branch chw git@github.com:yjmyl/flash-linear-attention-npu.git
```

切到本次 vLLM-Ascend 验证分支，并让 vLLM 使用仓库记录的匹配提交：

```bash
cd /home/z00886386/vllm-ascend
git fetch origin
git checkout -B a2-a3-a5-gdn-phase6-only-refactor \
  origin/a2-a3-a5-gdn-phase6-only-refactor

VLLM_COMMIT=$(tr -d '[:space:]' < .github/vllm-main-verified.commit)
cd /home/z00886386/vllm
git fetch origin
git checkout "${VLLM_COMMIT}"
```

安装 vLLM。`VLLM_TARGET_DEVICE=empty` 避免在上游 vLLM 仓构建 CUDA 扩展：

```bash
cd /home/z00886386/vllm
VLLM_TARGET_DEVICE=empty \
python3 -m pip install --no-build-isolation --no-deps -e .
```

构建 vLLM-Ascend 前选择实际硬件的 `SOC_VERSION`：

| 硬件 | vLLM-Ascend `SOC_VERSION` | FLA `FLA_NPU_SOC` |
| --- | --- | --- |
| A2 | `ascend910b1` | `ascend910b` |
| A3 | `ascend910_9391` | `ascend910_93` |
| A5 | 以环境中实际 950 型号为准，例如 `ascend950dt_9582` | `ascend950` |

不要仅凭表格猜测 A5 子型号。先在目标容器查询：

```bash
npu-smi info
python3 - <<'PY'
import torch
import torch_npu

torch.npu.set_device(0)
print("device_name:", torch_npu.npu.get_device_name(0))
print("soc_version:", torch_npu.npu.get_soc_version())
PY
```

然后构建并安装 vLLM-Ascend，例如 A3：

```bash
export SOC_VERSION=ascend910_9391
cd /home/z00886386/vllm-ascend
git submodule update --init --recursive
python3 -m pip install --no-build-isolation --no-deps -e .
```

不要设置一个与实际 NPU 不匹配的 `SOC_VERSION`。安装后可检查自动生成的
`vllm_ascend/_build_info.py`，确认 `__device_type__` 是预期的 A2、A3 或 A5。

## 3. 构建并安装 FLA wheel

选择与目标硬件对应的 `FLA_NPU_SOC`。下面以 A3 为例：

```bash
export FLA_NPU_SOC=ascend910_93
cd /home/z00886386/flash-linear-attention-npu
git submodule update --init --recursive

python3 -m pip wheel \
  --no-build-isolation \
  --no-deps \
  . \
  -w dist

python3 -m pip install \
  --force-reinstall \
  --no-deps \
  dist/flash_linear_attention_npu-*.whl
```

A2 改为 `FLA_NPU_SOC=ascend910b`，A5 改为
`FLA_NPU_SOC=ascend950`。

正常源码构建不要设置 `FLA_NPU_SKIP_RUN_BUILD=TRUE`。该开关只适合已经有
完整、匹配当前 SoC 的构建产物而仅重新打包 wheel 的场景；误用会把缺失或
过期 OPP 打进 wheel。

## 4. 确认 FLA OPP 和公开接口

FLA 与 vLLM-Ascend 都可能提供名为 `libcust_opapi.so` 的库。不要把一个仓库
的文件复制到另一个仓库的 vendor 目录，也不要用同名文件互相覆盖。两个
自定义算子包应保留各自 OPP vendor 目录，由初始化逻辑共同加入
`ASCEND_CUSTOM_OPP_PATH`。

检查安装位置、OPP 目录和公开 Phase6 符号：

```bash
python3 - <<'PY'
from pathlib import Path

import fla_npu
from fla_npu.ops import ascendc

package_dir = Path(fla_npu.__file__).resolve().parent
vendor_dir = package_dir / "opp" / "vendors" / "fla_npu_transformer"
print("fla_npu:", package_dir)
print("vendor_dir:", vendor_dir)
print("vendor_exists:", vendor_dir.is_dir())
print("phase6:", ascendc.gdn_core_fwd_phase6)
assert vendor_dir.is_dir()
assert callable(ascendc.gdn_core_fwd_phase6)
PY
```

vLLM-Ascend 会在自身 custom OPP bootstrap 之前导入
`fla_npu.ops.ascendc`，确保 kernel manager 建索引前已经看到两个 OPP
目录。

## 5. 只运行 FLA Phase6 单算子验证

本次不以 causal conv、recurrent 或六个 standalone 小算子的通过情况作为
接入验收。先运行 FLA 仓中直接调用 `gdn_core_fwd_phase6` 的验证脚本：

```bash
export ASCEND_RT_VISIBLE_DEVICES=3
cd /home/z00886386/flash-linear-attention-npu/torch_custom/fla_npu/test

python3 validate_gdn_phase6_gva_dense_t.py \
  --device 0 \
  --key-heads 2 \
  --value-heads 8 \
  --tokens 130 \
  --chunk-size 64
```

这里宿主机物理卡 3 在进程中映射为逻辑 NPU 0，所以脚本参数是
`--device 0`。重点确认四个返回张量均为有限值，native GVA 与展开 head 的
参考结果一致。

随后运行 vLLM-Ascend 的真实 NPU Phase6/native Prefill 对比：

```bash
export VLLM_ASCEND_GDN_BACKEND=fla_npu
cd /home/z00886386/vllm-ascend

pytest -s -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py \
  2>&1 | tee /tmp/gdn-phase6-operator.log
```

该文件覆盖 token 长度 `1`、`63`、`64`、`65`，以及
`cu_seqlens=[0,1,65]` 的双序列 varlen 情况，同时比较 output 和 final
state。

## 6. 运行延后的单元测试

```bash
cd /home/z00886386/vllm-ascend

pytest -q \
  tests/ut/ops/test_gdn_fla.py \
  tests/ut/device/test_device_config.py \
  tests/ut/test_platform.py \
  tests/ut/quantization/test_utils.py
```

记录总通过数、warnings，以及 vLLM、vLLM-Ascend、torch、torch-npu、CANN
和 FLA wheel 版本。单元测试通过不能替代本节前后的真实 NPU 与模型验证。

## 7. 首先验证 Qwen3.6-35B eager

创建日志配置 `/tmp/vllm-logging-config.json`：

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

`VLLM_LOGGING_CONFIG_PATH` 会替换 vLLM 默认的 logging dictionary，而不是
只在其上追加一个 logger。因此配置中必须同时保留 `vllm` 和
`vllm_ascend` 两个 namespace，否则可能看不到框架日志或 Phase6 选择日志。

使用宿主机物理卡 3、4 运行两卡 Qwen3.6-35B 比较：

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

`ASCEND_RT_VISIBLE_DEVICES=3,4` 表示进程只看到宿主机物理卡 3、4；进程内
它们重新编号为逻辑 NPU 0、1。测试文件名保留上游的 `27b`，但实际模型由
`QWEN36_MODEL_PATH` 指向 35B，本次验收以日志中的实际模型路径为准。

严格模式成功时应看到类似：

```text
GDN Phase6 selection: backend=fla_npu
symbol=fla_npu.ops.ascendc.gdn_core_fwd_phase6
soc=ascend910b|ascend910_93|ascend950
```

不得出现针对 `causal_conv1d` 或 `recurrent_gated_delta_rule` 的 FLA 选择
日志。若真实 Phase6 调用失败，应看到 `GDN Phase6 execution failed`，且请求
直接失败，不会在执行后重新跑 native。

## 8. 可选：验证 FULL_DECODE_ONLY 与 MTP 共存

这一步验证“eager Prefill 使用 Phase6、Decode-only 图中使用 native”，不是
验证 Phase6 本身能够入图。使用支持 Qwen3.5 MTP 的模型，例如：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_ASCEND_GDN_BACKEND=fla_npu
export VLLM_LOGGING_CONFIG_PATH=/tmp/vllm-logging-config.json

vllm serve "${QWEN35_MTP_MODEL_PATH}" \
  --served-model-name qwen35 \
  --host 0.0.0.0 \
  --port 8890 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-num-seqs 32 \
  --max-model-len 133120 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --all2all-backend allgather_reducescatter \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
  --async-scheduling \
  --distributed-executor-backend mp
```

验收点：

1. Prompt Prefill 有一次 Phase6 选择和执行；
2. 图捕获/重放阶段没有新的 FLA import、resolve 或 Phase6 调用；
3. MTP Decode 使用 `npu_recurrent_gated_delta_rule` native 路径；
4. 服务能完成生成，且图捕获没有因 FLA Prefill 失败。

## 9. 日志和 profiling 检查

快速筛选后端决策：

```bash
grep -E \
  "GDN Phase6 selection|GDN Phase6 execution failed|fla_npu preload failed" \
  /tmp/qwen36-35b-gdn-phase6.log
```

`auto` 模式的安全回落会包含：

```text
backend=native requested=auto stage=eligibility|resolve|scratch_probe reason=...
```

严格 `fla_npu` 模式不允许上述回落。成功选择日志并不单独证明真实模型 shape
已经完成，测试最终 PASS 和生成结果也必须同时确认。

开启 torch profiler 后，在导出的 `trace_view.json` 中查找：

```bash
grep -R "ChunkGdnCoreFwd" /path/to/profiler/output
```

预期只在 Prefill 观察到 Phase6 对应 kernel；Decode/MTP Decode 应观察到现有
native recurrent kernel。若日志显示 Phase6，但 profiling 只有旧 Prefill
算子，应检查实际 worker 使用的 vLLM-Ascend checkout、wheel 安装路径、环境
变量继承及 profile 捕获窗口。

## 10. 按 SoC 分别记录结果

每种芯片必须独立验证，不能用 A5 结果推断 A2/A3：

| 项目 | A2 | A3 | A5 |
| --- | --- | --- | --- |
| FLA wheel target | `ascend910b` | `ascend910_93` | `ascend950` |
| FLA Phase6 直接测试 | 待记录 | 待记录 | 待回归 |
| vLLM-Ascend 单元测试 | 待记录 | 待记录 | 待回归 |
| Phase6/native NPU 对比 | 待记录 | 待记录 | 待回归 |
| Qwen3.6-35B eager | 待记录 | 待记录 | 已验证 A5 serve，比较测试待记录 |
| FULL_DECODE_ONLY + MTP | 可选/待记录 | 可选/待记录 | 可选/待记录 |
| ais-bench | 待验证 | 待验证 | 待验证 |

每次记录还应包含：commit、FLA 分支/commit、CANN 版本、torch/torch-npu
版本、`SOC_VERSION`、可见物理卡、逻辑 device 编号、PASS 数和完整日志路径。
