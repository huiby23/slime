# TIM：用同一份代码管理不同实验

BF16、NVFP4 和噪声注入共享数据、训练循环、检查点和指标定义，适合保存在一个分支。
**实验差异写在配置里，代码开发用分支，运行结果绑定提交和完整配置。**

## 配置与分支怎么分工

| 内容 | 管理方式 |
| --- | --- |
| BF16 / NVFP4 rollout / 固定噪声，学习率、seed、batch | 版本控制中的 JSON 配置 |
| 新算法、算子替换、Score Centering 等尚在开发的实现 | 独立开发分支，验证后合并 |
| 一次运行的代码、参数、数据版本与硬件 | 输出目录中的冻结源码、manifest、resolved config |
| 训练权重、原始数据、完整日志和 TensorBoard events | 用户指定的实验输出目录，避免提交到 Git |
| 长期引用的 baseline | 提交 SHA；需要固定发布版本时另加 tag |

不必为每个学习率创建分支。只记录 `tim/base` 这个名字也不足以复现，因为分支会继续更新。
多次实验可以并行运行：入口会把本次使用的代码冻结到各自输出目录，再挂载到独立容器。

## 已提供的配置

| 配置文件 | 训练 | rollout | 在线评测 | 恒定 LR | GPU 数 |
| --- | --- | --- | --- | --- | --- |
| `bf16_lr1e6.json` | BF16 | BF16 | BF16 | 1e-6 | 1 |
| `bf16_lr1e5.json` | BF16 | BF16 | BF16 | 1e-5 | 1 |
| `bf16_lr1e3.json` | BF16 | BF16 | BF16 | 1e-3 | 1 |
| `nvfp4_rollout.json` | BF16 | NVFP4 模拟 W4A4 | NVFP4 模拟 W4A4 | 1e-6 | 1 |
| `fixed_noise_s005.json` | BF16 | BF16＋固定权重噪声 | 干净 BF16 | 1e-6 | 1 |
| `nvfp4_rollout4_fresh.json` | BF16 | 4个独立的模拟 sampler | NVFP4 模拟 W4A4 | 1e-6 | 5 |
| `nvfp4_train_rollout.json` | NVFP4 训练配方模拟 | NVFP4 模拟 W4A4 | NVFP4 模拟 W4A4 | 1e-6 | 1 |

NVFP4 使用普通 PyTorch 运算模拟线性层量化；不代表原生 FP4 的速度、显存或逐位精确行为。
注意力、归一化和 embedding lookup 保留原路径。模拟训练的参数存储仍为 BF16，具有 FP32 optimizer master/state。
`nvfp4_train_rollout.json` 是统一到当前数据和 batch 的新预设，不是早期模拟训练实验的完整复刻。
四 sampler 预设从原始权重开始训练，不会自动接入历史第111步或113步。

共同设置：Qwen3-0.6B Dense、500次更新、16题×8回答、1024新 token 上限、2048上下文、AdamW、weight decay0.01、β=(0.9,0.98)、seed42。
GRPO 不启用 TIS、RS、Score Centering、KL 或 entropy bonus；记录相同生成 token 的训推 log-prob 差异。
`noise.*` 仅在 `rollout.precision=fixed_noise` 时生效。

LR1e-5 预设另外评测并保留前10步的完整检查点。其他预设默认每25步评测。
要做严格的学习率对照，应让各组评测日程一致；相同全局 seed 不保证相同采样轨迹。
早期完整检查点包含模型、Adam/RNG和数据游标，需要为这些额外检查点预留空间。

## 依赖和本地资产

- 主机：Python3.8+、Git、Docker；训练时需要 NVIDIA Container Toolkit。
- CPU预检会自动查找并只读挂载主机的 `libcuda.so.1`，满足Transformer Engine导入；不暴露GPU设备，也不初始化CUDA。
- 使用已经安装依赖的镜像，默认 `slimerl/slime:v0.4.0-cu129`。入口只使用本地镜像并记录不可变的 image ID，不会拉取或安装。
- Slime 引擎提交：`0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e`，从本地 Git 历史归档；当前工作区的其他引擎修改不参与实验。
- 镜像中的 Megatron 默认位于 `/root/Megatron-LM`。使用不同镜像前，需重新验证路径和依赖。
- 本地 Qwen3-0.6B safetensors 权重及 tokenizer。支持 HF cache 的 snapshot/blob 相对链接，自动挂载整个模型 cache 根目录。
- 训练/验证 JSONL 均需 `prompt` 和 `label`。训练 metadata 需有 `screen_samples=8`、`screen_correct_count` 为2—6。
- 默认检查2784道训练题和512道验证题；筛选只作用于训练集，验证集不按正确率筛选。两者不得有重复 prompt 或相同 source_index。

不包含权重、训练题目内容或私人存储路径。替换机器时通过命令行指定位置。

## 先预检，再启动

在仓库根目录执行。替换以下路径；`preflight` 不申请 GPU：

```bash
python3 examples/tim/run.py preflight \
  --config examples/tim/configs/bf16_lr1e6.json \
  --model-path /path/to/Qwen3-0.6B \
  --train-data /path/to/train.jsonl \
  --eval-data /path/to/validation.jsonl \
  --output-dir outputs/tim-bf16-s42
```

`plan` 只打印解析后的计划；`prepare` 冻结代码和资产哈希；`preflight` 在不暴露 GPU 的容器中检查原生 Slime 参数、AdamW、hook、保存协议和数据。
输出目录必须是新目录；已有的同一次准备可以用于后续预检和训练，禁止修改其配置、输入或冻结源码。

正式运行使用相同参数，将 action 改为 `train` 并选择实际可用的卡：

```bash
python3 examples/tim/run.py train \
  --config examples/tim/configs/bf16_lr1e6.json \
  --model-path /path/to/Qwen3-0.6B \
  --train-data /path/to/train.jsonl \
  --eval-data /path/to/validation.jsonl \
  --output-dir outputs/tim-bf16-s42 \
  --gpus 0
```

`train` 会重新预检最终 bundle，再核实指定 GPU 无进程、显存≤64MiB且利用率为0；不占位、不排队、不结束其他任务。
容器独占自己的 Ray 生命周期，源码、模型和数据只读，新增日志、cache、checkpoint 和 TensorBoard 全部写入指定输出目录。
前台启动可配合自己的 tmux 会话使用。该入口只支持 fresh start，拒绝覆盖已有训练轨迹；历史续训协议见单独文档。

参数可以保存在新的 JSON 文件中，也可以使用有限的配置覆盖：

```text
--set optimizer.lr=1e-5
--set training.seed=43
--set training.updates=2 --set evaluation.interval=1
```

覆盖沿用类型和合法性检查；拼错字段会报错，不会静默忽略。
如果使用 LR1e-5 预设做2步 smoke，还需把 `evaluation.early_updates` 和 `evaluation.retain_early` 都设为0。
最后一步必须安排评测。减小验证题数或切换数据应同步修改 `data.eval_questions` / `data.train_questions`，并使用新输出目录。

## 固定噪声

从初始权重生成一次独立高斯 bank：`eta = Z * W0`（FP32）。每次同步都使用 `BF16(Wt + sigma * eta)`。
默认 sigma0.05、bank seed20261030；这是逐权重相对高斯标准差，扰动并不限于±5%。
bank 不随当前权重重算、不累计、不重采样；不同 canonical tensor 使用独立的按名称确定的 RNG，tied embedding/head 共用同一份。
训练端保持干净，评测从缓存的干净 BF16 权重恢复，不用“带噪 BF16 减噪声”近似恢复。
目前只支持一个 colocated sampler；多副本需先扩展控制与审计。

已有同源 bank 可以直接复用。准备新 bank 的工具只读本地模型，不下载，需在有 PyTorch 与 safetensors 的环境中执行：

```bash
python3 examples/tim/prepare_noise_bank.py \
  --model-path /path/to/Qwen3-0.6B \
  --output-dir /path/to/noise-bank \
  --seed 20261030 --sigma 0.05
```

工具存储未乘 sigma 的 eta；`--sigma` 用于说明所需运行幅度，实际幅度由实验配置控制。
有相同来源的完整 bank 时先验证再复用，拒绝覆盖不完整目录或不同来源的数据。
使用 `fixed_noise_s005.json` 时额外传入 `--noise-bank /path/to/noise-bank`。
同一 bank 配合 `--set noise.sigma=0.0` 可做同 wrapper 的零噪声对照。

## 输出和判断口径

- `resolved-config.json`：完整配置、原生 argv、代码版本、数据/模型哈希、精度和拓扑。
- `sources/`：实际运行的固定 Slime 源码和本次扩展；源文件清单及 bundle SHA 同时保存。
- `configuration-check.json` / `launch.json`：最终 bundle 预检、镜像 ID 与实际 GPU UUID。
- `status.json` / `optimizer-validation.jsonl` / `training-validation.json`：执行进度、逐次 optimizer 观察和结束验证。
- `evaluations-by-completed-updates.jsonl`：初始模型为0，第一次更新后的评测为1，避免原生 rollout_id=0 的歧义。
- `checkpoints/best`、`last`、可选 `early/`：完整模型/Adam/RNG/数据游标的原子发布。
- `tensorboard/`、`logs/train.log`：原生曲线和训练日志。TensorBoard 用 `tensorboard --logdir /path/to/output/tensorboard --port 6006` 启动；远程 SSH 需转发对应端口。
- NVFP4：`rollout-quantization.workers/worker-*.json(.jsonl)`，每个 worker 单独记录加载、缓存失效和重建，避免覆盖。
- 噪声：`noise-audit.json(.jsonl)`、`clean-evaluation.jsonl`，记录固定扰动和 clean/noisy 切换。

执行成功、梯度有限、模型答对率、回答长度/截断率、训推差异必须分别查看。
零梯度不自动等于数值故障；有梯度也不代表学得好。
平均 token log-prob 误差不是准确率、完整分布 KL，也不单独证明训推不一致导致训练崩坏。
NVFP4 在线评测使用模拟权重，不能替代同一 checkpoint 的干净 BF16 评测。

## 验证范围

共享训练循环固定在历史 Slime 上，新入口只切换配置与插件；CPU 测试检查参数、噪声生命周期、cache 和检查点协议。
历史实验及其 frozen 源码不被改写。统一入口仍需新的 GPU smoke 才能验证完整 rollout→重算→更新→同步路径，CPU 预检不能代替这一步。

```bash
python3 -m unittest discover -s tests -p 'test_tim_bf16_runtime.py' -v
python3 -m unittest discover -s tests -p 'test_tim_protocol.py' -v
# 在具有 PyTorch/pytest 的环境中：
python -m pytest -o addopts='' \
  tests/test_tim_fixed_noise.py tests/test_tim_nvfp4_runtime.py
```
