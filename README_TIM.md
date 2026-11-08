# Train/Inference Mismatch baseline

`tim/base` 将 BF16、NVFP4 软件模拟和固定权重噪声实验放在同一份实现中，用配置区分实验。

- [配置、运行和复现说明](examples/tim/README.md)
- [配置目录](examples/tim/configs)
- [历史实验与当前入口的对应关系](examples/tim/HISTORICAL_PROTOCOL.md)

入口：`python3 examples/tim/run.py --help`。主机入口仅依赖 Python 标准库，训练依赖来自已有 Docker 镜像。

默认模型为 Qwen3-0.6B Dense，训练数据为冻结筛选的 GSM8K 子集，GRPO 每步16题×8回答。
主训练引擎固定到历史实验使用的 Slime 提交；本分支提供独立的实验扩展，避免将引擎升级混入对照实验。

本次整理只做 CPU 测试与配置预检；重构后的统一入口尚未重新执行 GPU rollout 或训练。
历史训练的完成情况不能替代新入口的 GPU 验证。
