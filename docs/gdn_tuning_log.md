# GDN 接入与调优记录

分支：feat/gdn-qwen35-36
记录者与日期：2026-08-24
环境：A5（Ascend 950DT），CANN 版本（执行时填写），vllm-ascend main 基线 commit d2818b08d

## 阶段 0 基线（2026-08 摸底）

- 算子级精度脚本输出：（粘贴 `gdn_op_accuracy_check.py --cases all` 结果）
- 性能摸底输出：（粘贴 `gdn_op_profiling.py` 结果）
- 热点排序结论：（按 prefill/decode 各 kernel 实测耗时排序，注明阶段 2 替换顺序依据）

## 阶段 1 chunk 解锁

- VLLM_ASCEND_GDN_FUSED_CHUNK 三态验证结果：
- CANN 包 A5 chunk 实现是否就绪：（是/否 + 依据）

## 阶段 2 逐项替换

- 替换点 A（causal_conv1d）：torch_npu 接口确认 / 精度对比 / 性能对比 / 决策
- 替换点 B（recurrent）：torch_npu 接口确认 / 精度对比（含跨阶段组合）/ 性能对比 / 决策

## 阶段 3 调优

- 实验矩阵：（每次实验记录：参数变更、TTFT/TPOT、结论）
- chunk size 调优：
- 状态 dtype（bf16 vs fp32）权衡：
- MegaGDN（huawei-csl/megagdn-pto）评估结论：
