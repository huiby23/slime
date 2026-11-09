# 在 8 张 A100 上做 Qwen3-4B MATH NVFP4 注意力后训练

本实验从原始 `Qwen/Qwen3-4B` 开始，以非思考模式做 100 次 GRPO 更新。W16A16C4V16 中的 C4 表示注意力 QK 和 PV 的输入操作数采用 NVFP4 数值模拟。训练、生成和评测共用同一个注意力模块。模型权重、线性层激活和原生分页 KV 存储均为 BF16。

## 实验协议

| 项目 | 设置 |
|---|---|
| 数据 | 全部 7,500 道 MATH 训练题，全部 500 道 MATH-500 留出题，与本项目昇腾实验的文件 SHA-256 一致 |
| 采样 | seed 1234，temperature 1，top-p 1，每题 8 个回答，回答上限 3,070 token，模型运行上下文 8,192 |
| 训练 | 每轮收齐 5 个奖励有方差的题组后更新一次，global batch 40，micro batch 1，Adam 一、二阶矩 FP32，学习率 1e-6 |
| 卡分配 | 同节点 8 张 A100 80GB，训练 DP8、TP1、PP1，采样 8 个 TP1 实例，训练和采样分时使用设备 |
| 评测 | 第 0、5、10 到 100 步共 21 次，greedy，每题一答，保存逐题回答、准确率、输出长度与截断率 |
| 保存 | 每 5 步保留独立的 Megatron DCP（参数、Adam、原生调度器与 RNG）、数据源游标和 HF 权重 |

Q、K 按每个 token/head 的特征轴分成连续 16 个值，每个块用 E2M1、E4M3FN 局部 scale 和 FP32 上层 scale。P 在每个查询行的 128-key 在线 softmax 块内，对未归一化指数值做一次 NVFP4 转换后送入 PV；分母使用转换前的概率。V 沿逻辑序列每 16 个位置分组，每个 head 的 16-token × head_dim 小矩阵共享一个上层 scale，每个特征列有独立的局部 scale。对角组的 scale 只看当前查询可见的值。该上层 scale 粒度是本实验的选择。

Vector 的指数输入、概率、在线重缩放、分母和分子更新有 BF16 舍入边界，矩阵乘累加与块内求和用 FP32。模拟使用确定性最近偶数舍入，训练对量化使用直通估计器，反向使用常规 autograd。CUDA 代码在 Ampere 上执行软件量化并重建操作数，沿用 BF16 KV；每次读取逻辑前缀后重新转换。测量吞吐时请记录实际输出 token 和整轮耗时。格式依据见 [NVIDIA NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)。

## 准备环境和数据

推荐使用之前两卡 Slime 实验的 CUDA 镜像 `slimerl/slime:v0.4.0-cu129`。已检查的本地 Docker image ID 为 `sha256:efbc54259ccfc247e24d36be3425a4d34fe421e2230dfafd234adf87b04dc657`。该镜像包含 Megatron-LM、SGLang、Triton、Ray 和数学奖励依赖。将此分支挂载到容器，例如 `/work/slime`；将模型、数据和实验输出挂载在持久目录，例如 `/work/weights` 和 `/work/experiments`。分配 8 张 GPU，并为 Ray 提供足够的共享内存（例如 Docker `--shm-size=16g`）。

以下命令在容器内、仓库根目录执行。更换 `MODEL` 和 `WORK` 即可使用其他挂载点。完整保留 20 份 4B DCP 和 HF 的容量预算约 **1.3 TB**，另留临时文件空间；实际大小取决于分布式优化器布局。

```bash
cd /work/slime
export PYTHONPATH="$PWD:/root/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export MODEL=/work/weights/Qwen3-4B
export WORK=/work/experiments
export DATA="$WORK/math7500-math500-3070"
export RUN="$WORK/qwen3-4b-math-nvfp4-01"

python -m examples.nvfp4_fa_math.data "$DATA"
python -m examples.nvfp4_fa_math.run plan --model "$MODEL" --data "$DATA" --run-dir "$RUN"
```

数据准备支持离线参数 `--train-parquet /path/train.parquet --test-jsonl /path/test.jsonl`，也可直接指向昇腾作业已生成的同一数据目录。准备程序验证固定版本的上游文件和最终 JSONL 的 SHA-256，包含 4 个上游训练答案格式修复。正式启动会校验原始模型 3 个 safetensors 分片及 tokenizer。

## 实卡测试和训练

先在已分配的 A100 上执行小张量测试；覆盖量化码点、GPU 前后向、分块与整段预填充、分页缓存隔离，以及 512、1,024、2,048、3,070 长度的固定输入解码。

```bash
python -c 'import torch; assert torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0)'
python tests/test_nvfp4_fa_math.py -v
python -m examples.nvfp4_fa_math.run train --model "$MODEL" --data "$DATA" --run-dir "$RUN"
python -m examples.nvfp4_fa_math.verify "$RUN"
```

训练使用私有本机 Ray 实例。进度在 `$RUN/driver.log`，TensorBoard 在 `$RUN/tensorboard`。每次评测的原始答卷在 `evaluation/step_NNN.json`。第 5 步对应 `hf/rollout_004`，第 100 步对应 `hf/rollout_099`，因为 Slime 的 rollout ID 从 0 开始。HF 导出保存 BF16 模型与 tokenizer；部署时要使用这个分支的注意力实现才能复现 C4V16 的数值行为。

## 恢复与验收

恢复时将原运行目录作为输入，指定新的输出目录。启动器读取最新一份已经完成校验的 `saved/step_NNN.json`，检查 DCP 中的 Adam 一、二阶矩、RNG、分片长度、游标和 HF 索引。随后由 Megatron 原生加载包括调度器在内的训练状态。如果上次已经保存权重但尚未写出该步评测，会先用恢复的权重补测该边界。

```bash
python -m examples.nvfp4_fa_math.run train --model "$MODEL" --data "$DATA" \
    --resume-from "$RUN" --run-dir "$WORK/qwen3-4b-math-nvfp4-02"
python -m examples.nvfp4_fa_math.verify "$WORK/qwen3-4b-math-nvfp4-02"
```

验收沿恢复链读取 100 次更新的有限正 grad norm、20 个完整保存点、21 个评测点，并全量扫描最终 HF 的形状、dtype、有限值和分片 SHA-256，再抽样对比原始权重。结果写入 `verification.json`。保留父运行目录以支持验收和复核。

此分支编写时已在历史 A100 镜像的 CPU 模式验证原生参数解析、Megatron spec 和 SGLang 模型注册，并为 SM80 离线编译量化和 PV 内核。真实 8 卡吞吐、峰值显存和完整训练恢复仍须通过实卡运行验证。
