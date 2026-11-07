# Qwen3 W4A8 实验入口

当前主实验为 **Qwen3-4B Dense：BF16 对照与线性层 NVFP4 W4A8 数值模拟**。
新运行默认 50 次 rollout，每 5 次 rollout 进行一次 50 题 valid，并在训练开始前执行一次 valid。
rollout 使用 4 道题、每题 4 个回答；valid 使用旧研究相同的 50 道题、每题 1 个回答。
历史正式实验的 100 次更新及最终 512 题评估已完成。

完整说明见 [4B 实验指南](QWEN3_4B.md)，实际配置和结果见
[正式实验索引](experiments/qwen3_4b_20261030.json)。项目总览见
[仓库根目录的中文说明](../../README_4B.md)。

## 脚本职责

| 环节 | 入口 | 作用 |
| --- | --- | --- |
| 准备 | `prepare_qwen3_4b.py` | 生成匹配的 BF16/W4A8 计划，不启动训练 |
| 下载 | `download_model.py`、`download_parallel_shards.py` | 下载并校验初始权重 |
| 检查 | `preflight.py` | 解析后端参数，检查精度和运行环境 |
| 验证 | `validate_megatron.py`、`validate_rollout.py` | 小模型后端集成验证，支持 Dense/MoE |
| 调度 | `run_dense_experiment.py` | 空闲 GPU 检查、全尺寸验证和正式实验调度 |
| 训练 | `train_best_last.py` | 接入原生 Slime，启用每次更新保存的 best/last 策略 |
| 资源 | `observe_idle.py` | 观察并记录 GPU 连续空闲时间 |
| 初始化标记 | `gpu_claim.py`、`watch_gpu_claims.py` | 在容器初始化时显示 CUDA 进程，记录设备归属 |

`prepare_qwen3_30b.py` 和 `queue_validation.py` 用于早期 30B MoE 准备，
对应说明见 [MoE 文档](docs/moe_30b.md)。通用量化和后端适配器仍共用一套实现。

## 运行阶段与证据

准备计划 → 小模型后端验证 → 全尺寸短更新及恢复验证 → 100 次正式更新。
正式训练前必须检查权重完整性、磁盘空间和 GPU 当前进程。
已有任务启动后遇到其他进程竞争时，记录实际情况，不自动终止已有训练。

容器启动、初始评估和参数更新是不同阶段。实际进度依据优化器记录、
完整 last checkpoint 和评估结果确认。全尺寸验证使用 32 题、每题 4 个回答；
正式训练使用 4 题、每题 4 个回答。二者均使用 1024-token 回答上限。

每次参数更新后原子发布完整 `last`，每 25 次更新评估一次并按改善结果更新
权重版 `best`。保存失败时保留上一份完整 checkpoint，清理未被引用的旧快照。

启动器的运行镜像、数据目录和 Ray 临时目录目前使用本机约定，跨机器部署前
需要调整这些路径。完整冻结来源与实际 Docker 命令记录在每次实验的目录中。

## 监控与临时产物

TensorBoard 只展示正式 `BF16` 与 `W4A8`。训练和评估由不同事件文件写入，
查看曲线时启用 `--reload_multifile true`，以持续读取全部文件。

临时冒烟测试、缓存和已结束的容器在验证用完并确认无任务依赖后清理；
正式 best/last、日志、结果及对应代码版本保留。清理过测试产物的旧实验目录
不能再用依赖这些产物的 `--resume-after-validation` 快捷方式重新启动。
