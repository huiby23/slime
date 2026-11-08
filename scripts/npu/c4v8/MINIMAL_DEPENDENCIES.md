# 最小依赖闭包与提交范围

以下路径均相对工作区或仓库，不包含部署主机地址或个人路径。这里区分“执行时实际依赖”与“原部署 guard 为来源审计而绑定的文件”。最终由根任务放入隔离 Git 工作区；不要修改或重新指纹化正在运行的源码。

## 1. 数值核心

必须保留以下源码及其固定依赖版本：

```text
experiments/c4v8_device_core/device_attention.py
experiments/c4v8_device_core/device_quant.py
experiments/c4v8_device_core/exact_fp32.py
experiments/c4v8_fast_attention/fast_attention.py
experiments/c4v8_fused_nvfp4/nvfp4_triton.py
experiments/c4v8_fused_qdq/forward_api.py
experiments/c4v8_fused_qdq/fused_mxfp8.py
experiments/c4v8_staged_decode/staged_decode.py
experiments/c4v8_staged_decode/staged_kernels.py
experiments/c4v8_rl_training_bridge/numerical_profile.py
```

`fast_attention` 直接导入 `device_attention` 的配置／参考，因此 portable core、quant 和 exact division 仍是依赖，即使实际 NPU 路径使用融合实现。

`experiments/c4v8_fused_qdq/whole_decode.py` 没有进入实际路由，但仍在 `numerical_profile.source_files()` 中。保持当前 helper 原样时必须保留它并标记 **hash-binding-only / unrouted**。只在隔离导出中删除引用会产生新 implementation SHA，不能把旧 NPU 报告改成验证新 SHA 的报告。

## 2. 实际训练／SGLang 接口

```text
experiments/c4v8_rl_training_bridge/bridge.py
experiments/c4v8_rl_training_bridge/layouts.py
experiments/c4v8_rl_training_bridge/megatron_spec.py
experiments/c4v8_rl_training_bridge/recipe_support.py
experiments/c4v8_rollout_bridge/python-c4v8
experiments/c4v8_rollout_bridge/c4v8_runtime.py
experiments/c4v8_rollout_bridge/c4v8_bootstrap.py
experiments/c4v8_rollout_bridge/sglang_bridge.py
experiments/c4v8_rollout_bridge/logical_kv.py
experiments/c4v8_rollout_bridge/hardware_compat.py
```

实际 prepared recipe 闭包：

```text
experiments/c4v8_rl_training_bridge/prepared-fast-v3/qwen30b_dapo_c4v8.py
experiments/c4v8_rl_training_bridge/prepared-fast-v3/run-c4v8-dapo100-every5.sh
experiments/c4v8_rl_training_bridge/prepared-fast-v3/run-c4v8-runtime.sh
experiments/c4v8_rl_training_bridge/prepared-fast-v3/recipe_support.py
experiments/c4v8_rl_training_bridge/prepared-fast-v3/numerical_profile.py
experiments/c4v8_rl_training_bridge/prepared-fast-v3/precision-policy.json
experiments/c4v8_rl_training_bridge/prepared-fast-v3/c4v8_validation_protocol.py
```

注意 prepared 的 `recipe_support.py` 是正式启动器实际导入的模块，不能仅提交顶层同名文件。门禁曾因组装器漏记这个文件而拒绝启动，已修复为使用真实 consumer 自检。

Slime 主仓库的必要实现／补丁至少包括实际 model provider 的 `--spec` 接入、FP32 hybrid Adam 预分配、成功更新计数及 every5 HF saver、disk weight update 与 SG runtime 的环境/生命周期接口。它们属于主仓库或 pinned 依赖，不要以实验目录替代：

```text
slime/backends/megatron_utils/model_provider.py
slime/backends/megatron_utils/optimizer_step_hf_checkpoint.py
slime/backends/megatron_utils/hf_checkpoint_saver.py
slime/backends/megatron_utils/update_weight/update_weight_from_disk.py
slime/utils/npu_optimizer_preallocation.py
slime/backends/sglang_utils/runtime.py
slime/backends/sglang_utils/sglang_engine.py
slime/rollout/rm_hub/math_utils.py
slime/rollout/rm_hub/__init__.py
```

同样保留最终运行依赖的 SGLang loader、model_runner、scheduler_update_weights_mixin 补丁，及固定 Megatron/MindSpeed/mbridge 版本。对外导出应记录基础提交和补丁，不复制整个环境或未经审查的第三方仓库。

## 3. 最小部署 setup 依赖

| 文件／配置 | 作用 | 是否为这次 fresh C4V8 所需 |
|---|---|---|
| `setup/preallocate-hybrid-adam.py` | 在首次前后向前预分配真实 FP32 Adam 状态 | 需要；通过环境变量由 `npu_optimizer_preallocation.py` 加载 |
| `activate-slime.sh` 或等价环境模板 | 激活固定 Python、CANN/ATB、库搜索路径 | SG 评测脚本需要；导出为不含主机信息的模板 |
| `experiments/c4v8-integration-probes/run-owned-probe.py` | 稳定子进程身份和有界任务组清理 | `run-pilot-then100.py` 直接加载 |
| `setup/compare-initial-qwen30b.py` | 相对官方 base 的真实样本权重变化 | completion 验收需要 |
| `setup/scan-hf-finite-sha256.py` | 全索引、全参数有限值、全分片 SHA | completion／评测需要 |
| `setup/base100_every5_resume_lineage.py` | 旧 BF16 continuation 分支 | 本次 fresh guard 禁止该分支，不属于最小运行依赖 |
| 旧 replay / thinking-eval / 定点 status 脚本 | 历史恢复或观察 | 本次新策略不使用，不因旧字符串引用就加入默认启动链 |

模型和训练／验证 JSONL 是外部输入。需要发布其来源、清理方法与内容 SHA，不能把名称相同视作内容相同。

## 4. 评测和完成流水线

```text
experiments/c4v8_rl_training_bridge/eval-c4v8-trajectory.py
experiments/c4v8_rl_training_bridge/eval-c4v8-client.py
experiments/c4v8_rl_training_bridge/eval_deadlines.py
experiments/c4v8_rl_training_bridge/eval_support.py
experiments/c4v8_rl_training_bridge/c4v8_eval_protocol.py
experiments/c4v8_rl_training_bridge/c4v8_eval_inherited.py
experiments/c4v8_rl_training_bridge/c4v8_eval_runtime_support.py
experiments/c4v8_rl_training_bridge/run-c4v8-eval-server.sh
experiments/c4v8_rl_training_bridge/run-pilot-then100.py
experiments/c4v8_rl_training_bridge/pilot_evidence.py
experiments/c4v8_completion/run_pipeline.py
experiments/c4v8_completion/plot_curve.py
experiments/c4v8_completion/plot-requirements.txt
```

completion 的独立图形环境不用提交。worker 未执行；提交后暂停，不能把 prepared 计划放入任何自动启动入口。

## 5. Guard／测试／生成器的额外闭包

原 `recipe_support.source_inventory()` 对三个目录顶层 `*.py/*.sh` 做 broad glob，并绑定部分主仓库文件。因此原部署的 86 项 SHA 清单包含测试、probe 和生成器，**它不等于运行时 import graph**。

两种合法导出方法：

1. 保存原清单引用的全部小型源码，保持原始来源报告可核对；README 说明其中的 audit-only 文件。
2. 在隔离 Git 导出中改成显式最小清单，并生成新 export identity。原 NPU 报告保留原始身份，仅说明基础数学实现的验证来源；新部署重新走 guard，不伪造旧报告的 SHA。

若提交生成器 `prepare_recipe.py` 或 `prepare_evaluation.py`，必须同时带上它们读取的 `recipe-source-snapshot` 输入及其来源 SHA。当前已生成、审查的 evaluator 是权威版本；旧 `prepare_evaluation.py` 会覆盖后来补充的 timeout 等实现，不应被当成一键重建最新运行文件的入口。也可以省略旧生成器，仅保留已生成配置和测试。

建议保留 `test_bridge.py`、`test_recipe.py`、`test_eval_support.py`、`test_eval_deadlines.py`、`test_pilot_evidence.py`，prepared 的 `assemble-launch-gate.py`／`test_gate_assembler.py`，以及 completion 的 `test_pipeline.py`。数值组件和 SG 的专项测试由各组件提交清单补齐。

## 6. 精简真实证据索引

所有路径相对部署工作区；它们是证据定位，不是公开环境的硬编码启动参数。

| 范围 | 原始证据路径 |
|---|---|
| NVFP4 fused-only 格式及无 fallback | `experiments/c4v8_fused_nvfp4/evidence/npu-fused-only-v4/report.json`、`backend-log-audit.json` |
| MXFP8/Vector 正式 60 项组件 | `experiments/c4v8_fused_qdq/evidence/npu-default-finite-v1/report.json` |
| 最终整体前后向后验 | `experiments/c4v8-integration-probes/evidence/owned-finalcore-postcheck-v1/validation/report.json` |
| 当前 grouped SG 与 Megatron attention adapter | `experiments/c4v8-integration-probes/evidence/owned-attention-adapters-grouped-q1024-v1/adapter/report.json` |
| 真实 tiny Megatron Adam | `experiments/c4v8-integration-probes/evidence/owned-tiny-staged-q1024-v1/tiny/report.json` |
| 单层代表长度性能 | `experiments/c4v8-integration-probes/evidence/owned-final-performance-q1024-v1/measurements/report.json` |
| 原版 30B、并发 4、短输出 | `experiments/c4v8-integration-probes/evidence/owned-30b-sg-concurrency4-grouped-v1/pilot/manifest.json` |
| 上述旧桥同条件对照 | `experiments/c4v8-integration-probes/evidence/owned-30b-sg-concurrency4-baseline-v1/pilot/manifest.json` |
| 评测时限／owned cleanup | `experiments/c4v8_rl_training_bridge/prepared-fast-v3/eval-deadlines-cpu-v1.log` |
| 修正后的真实门禁 consumer 自检 | `experiments/c4v8_rl_training_bridge/prepared-fast-v3/consumer-fixed-review-v1.json` |
| completion 十项 CPU fixture | `experiments/c4v8_completion/evidence/cpu-tests-v4.log` |
| completion 未执行计划 | `experiments/c4v8_completion/qwen30b-407f37-v2/plan.json` |

有界设备报告还需关联同目录 `result.json`、完整外层日志 SHA 和全 8 卡释放证据，不能只摘一个 `passed=true`。公开提交建议提取不含部署信息的证据摘要，并保留原报告 SHA；不要把脱敏后摘要标成原始报告原封不动。

不提交：认证材料、环境目录、模型权重、数据全集、编译缓存、进程控制套接字、全部历史大日志、失败候选重复树、测试示意图。`CPU FIXTURE` 图和合成数据不是实验结果。
