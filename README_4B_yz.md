# Qwen3-4B-Base：W4A8 下四条 attention 路线的设计与开发指南

本文面向需要运行、修改和验证四条 attention 路线的开发者，按 2026-11-07 的代码说明。路径均相对于本仓库；运行进度、GPU 分配不属于本文的静态保证。

> **研究范围：所有路线共同使用 Linear W4A8 软件数值模拟，只改变训练侧 attention。C16 不等于全模型 BF16；C4/C8 也不表示 rollout/validation 使用同精度 attention。**
>
> 本文讨论 **Qwen3-4B-Base**。其他文档中的非 Base 4B、BF16/W4A8 双臂、100 次更新或 4×4 batch 是历史实验，不能直接替代本协议。当前实现不是原生 FP4/FP8 kernel 性能实验。

## 1. 四条路线的区别

| 配置值 | Q/K/V 与概率 P 的前向量化 | 反向 D 使用的输出 | dV 使用的概率 | 训练执行 backend |
| --- | --- | --- | --- | --- |
| `c16` | 原生 backend 的精度规则，不安装软件 attention quantizer | 原生 backend | 原生 backend | Megatron / Transformer Engine 原生 attention spec |
| `naive-c8` | FP8 E4M3 | 量化前向输出 O | 未再次量化的 P | 自定义 PyTorch autograd / FP32 运算 |
| `naive-c4` | NVFP4 | 量化前向输出 O | 未再次量化的 P | 同上 |
| `attn-qat-c4` | NVFP4，与 naive-C4 相同 | 辅助输出 O′ | 前向实际使用的 Pᶠ | 同上 |

`naive C4` 与旧讨论中的 `naive QAT C4` 是同一类别，不增加第五条路线。`attn-qat-c4` 的目标是修正低精度前向与反向之间的两处一致性关系，不是预先保证其精度更高或梯度一定更稳定。

### 1.1 共同的 W4A8 线性层

- 权重：NVFP4 E2M1 payload、E4M3 block scale、FP32 tensor scale；线性权重采用 16×16 tile。
- 激活：FP8 E4M3；线性层反向输出梯度：FP8 E5M2，FP8 按行 block64、FP32 scale。
- GEMM：FP32 软件数值模拟，关闭 TF32，输出回到 BF16；主参数、主梯度与 Adam 状态保持 FP32。
- 覆盖 attention 的 QKV/output projection 与 MLP projection；embedding、norm、LM head 不因此全部变成 W4A8。
- 不模拟硬件 packing、swizzle、随机 bit stream 或精确硬件累加顺序，不据此宣称原生低精度加速。

线性层量化与 attention 量化是两个独立边界。尤其不能把线性权重的 tensor scale / tile 定义套到下面的 attention quantizer。

### 1.2 低精度 attention 的公式

记 F 为路线对应的 fake quantizer，d 为 head_dim。以下公式按一个 head、一个 packed sequence 表示；实际实现支持 GQA。

```text
Qᶠ = F(Q), Kᶠ = F(K), Vᶠ = F(V)
P  = softmax(Qᶠ Kᶠᵀ / sqrt(d) + causal_mask)
Pᶠ = F(P)
O  = Pᶠ Vᶠ                     # 对外输出再转换回输入 dtype
O′ = P Vᶠ                      # 仅 Attn-QAT 计算并保存

dP = dO Vᶠᵀ
D  = rowsum(dO ⊙ O)            # naive-C8 / naive-C4
D  = rowsum(dO ⊙ O′)           # Attn-QAT-C4

dS = P ⊙ (dP - D)
dQ = dS Kᶠ / sqrt(d)
dK = dSᵀ Qᶠ / sqrt(d)
dV = Pᵀ dO                    # naive-C8 / naive-C4
dV = (Pᶠ)ᵀ dO                 # Attn-QAT-C4
```

设计要点：

1. naive-C4 与 Attn-QAT-C4 在相同 Q/K/V 输入上应有相同前向输出，区别只在反向。
2. naive 的 D 使用保存的、转换回输入 dtype **之前的 FP32 O**，不是把 BF16 输出重新转 FP32。因此不能称其与某个 BF16 FlashAttention backward kernel 逐位一致。
3. quantizer 本身在 `torch.no_grad()` 下执行，scale 不求梯度；梯度由 `_SimulatedAttention.backward` 显式定义，不是自动微分穿过 round 得到的梯度。
4. GQA 在前向展开 KV head，在反向将共享同一个 KV head 的梯度相加。
5. “普通 FlashAttention backward”只能描述 naive 的算法规则；当前后三条路线**没有直接调用 FlashAttention backward kernel**。

### 1.3 attention 量化格式与尺度范围

| 属性 | C8 | C4（两个 C4 路线相同） |
| --- | --- | --- |
| payload | FP8 E4M3，最大幅值 448 | E2M1，最大幅值 6 |
| block size | 64 | 16 |
| block scale | FP32 | E4M3 |
| 外层 scale | 无额外行级 scale | 每行独立 FP32 global scale |
| fake-quant 输出载体 | BF16 | BF16 |
| attention 运算 | fake-quant 结果转 FP32 后运算 | 同左 |

尺度统计边界：

- Q/K/V 为 `[token, head, head_dim]`，沿最后一维分组；每个 token/head 独立，不能跨 token 或 head 取 amax。
- P 为每个 head/query 的一行，沿 key 维度分组；causal future 先归零，不能影响有效 amax。
- 每条 packed sequence 单独计算 attention，不能跨序列共享概率尺度。
- C8 的 zero block 用 scale=1，输出仍为零。C4 的全零行采用 global_encode=1；zero block 的 block scale/encode 为零，输出为零。
- C4 的“global”仅相对于**当前行的多个 block**，不是整个 packed tensor 的全局统计。
- Attn-QAT 的 dV 复用保存的同一份 Pᶠ，不能重新用不同尺度量化。

旧 0.6B 原型的 E2M1 + group32 + FP32 absmax scale 不等于这里的 NVFP4，不能直接替换。

## 2. 代码导航与实际接入

| 文件 | 开发职责 |
| --- | --- |
| [attention.py](slime_plugins/w4a8_sim/attention.py) | `ROUTES`、C8/C4 quantizer、packed forward、自定义 backward、四路线 spec |
| [megatron.py](slime_plugins/w4a8_sim/megatron.py) | `model_provider` 选择 attention spec，并安装 W4A8 线性层 |
| [recipe.py](slime_plugins/w4a8_sim/recipe.py) | W4A8 线性层的量化与数值运算 |
| [nvfp4_sim/recipe.py](slime_plugins/nvfp4_sim/recipe.py) | attention 复用的 `round_e2m1` / `round_e4m3` 等基础运算，发布时不可漏掉此包 |
| [sglang_linear.py](slime_plugins/w4a8_sim/sglang_linear.py) | SGLang 线性层量化适配 |
| [sglang_models/qwen3.py](slime_plugins/w4a8_sim/sglang_models/qwen3.py) | Qwen3 projection 替换、权重同步后的量化缓存失效 |
| [train_best_last.py](examples/w4a8_sim/train_best_last.py) | 当前训练入口，接入 Slime、同步 best/last 策略及显式 metrics 恢复 |
| [metrics.py](slime_plugins/w4a8_sim/metrics.py) | reward/valid 导出、连续性检查、`prepare_resume` |
| [training_audit.py](slime_plugins/w4a8_sim/training_audit.py) | optimizer success、梯度范数、采样主参数有限性与更新检查 |
| [checkpoint_policy.py](slime_plugins/w4a8_sim/checkpoint_policy.py) | 完整 last、HF weights-only best、评测 hook |
| [prepare_qwen3_4b.py](examples/w4a8_sim/prepare_qwen3_4b.py) | 生成计划，不启动训练；包含需要覆盖的历史默认路径 |
| [generate_attention_commands.py](examples/w4a8_sim/generate_attention_commands.py) | 本机历史四路线 Docker 命令生成器，不是通用部署工具 |

训练链路：

```text
train_best_last.py → Slime train.py → MegatronTrainRayActor
  → w4a8_sim.megatron.model_provider
  → args.w4a8_attention（若存在且非空）或 W4A8_ATTENTION（默认 c16）
  → get_c16_reference_spec / get_naive_c8_spec /
    get_naive_c4_spec / get_attn_qat_c4_spec
  → Megatron GPTModel / TransformerLayer / SelfAttention.core_attention
```

C16 返回未替换的 TE spec；其他路线只将其 `core_attention` 替换为 `SimulatedAttention`。C16 的底层实现来自环境中的 Megatron/TE，不是本仓库独立实现的 kernel。不要把 `attention.py` 中模块级 `ROUTE` 当运行开关，实际路线由 provider 和 spec 决定。

**推理侧边界：** `W4A8SimQwen3ForCausalLM` 只替换 `qkv_proj`、`o_proj`、`gate_up_proj`、`down_proj`，沿用 SGLang attention。因此四路线比较的是不同训练 attention 得到的权重，在共同的 W4A8-linear / SGLang-attention 推理路径上的表现。

## 3. 如何分别使用四条路线

### 3.1 最小切换接口

在一份已核实可用的 **W4A8** 启动配置中，四路线只切换下列变量，并为每条路线设置独立输出目录和运行名：

```bash
export W4A8_ATTENTION=c16          # 原生 TE attention 对照
export W4A8_ATTENTION=naive-c8     # C8 前向 + naive backward
export W4A8_ATTENTION=naive-c4     # C4 前向 + naive backward
export W4A8_ATTENTION=attn-qat-c4  # C4 前向 + 一致性修正 backward
```

以上是四个互斥示例，实际启动时只选一个；依次执行会只保留最后一个值。变量必须进入训练进程和 Ray workers 的环境，修改宿主 shell 不会自动改变既有容器。

共同配置不能遗漏：

```text
W4A8_TRAINING_MODE=w4a8
PYTHONPATH=/quant/slime_plugins:/workspace/slime:/root/Megatron-LM
SGLANG_EXTERNAL_MODEL_PACKAGE=w4a8_sim.sglang_models

--custom-model-provider-path w4a8_sim.megatron.model_provider
--sglang-json-model-override-args '{"architectures":["W4A8SimQwen3ForCausalLM"]}'
```

仅设置 `W4A8_ATTENTION` 而未安装 custom provider 不足以启用路线；也不要假定已有通用 CLI 参数 `--w4a8-attention`。准备脚本的 `--attention` 是**生成计划参数**。

### 3.2 环境和目录约定

既有实验环境：`slimerl/slime:v0.4.0-cu129`，固定镜像 ID：

```text
sha256:efbc54259ccfc247e24d36be3425a4d34fe421e2230dfafd234adf87b04dc657
```

这是本机镜像 ID，不是保证其他机器可直接拉取的 registry digest。迁移时核对镜像与依赖版本；不要仅凭同名 tag 声称环境完全相同。镜像 Python 为 `/usr/bin/python`，不默认使用 `/opt/venv` 或重装 PyTorch/CUDA。

建议每条路线使用独立容器、单 GPU、16 GiB shared memory，并建立以下挂载：

| 容器路径 | 内容 | 权限 |
| --- | --- | --- |
| `/quant` 与 `/workspace/slime` | 同一份本仓库源码 | 只读 |
| `/models/Qwen3-4B-Base` | 原始 Base 模型与 tokenizer | 只读 |
| `/datasets` | 本次冻结的 train/validation 数据 | 只读 |
| `/workspace/w4a8` | 当前路线独立输出目录 | 可写、持久化 |

宿主编辑源码，项目运行与测试在经核实的专用容器内执行。当前训练使用镜像内 `/root/Megatron-LM`；修改宿主旁边的 Megatron 仓库不会自动生效。单卡容器内通常是逻辑 GPU0，物理卡必须以 Docker DeviceRequests 核实。

### 3.3 从零准备一条路线：先生成，后审查，最后启动

下面示例在**已经正确挂载的专用容器内**执行，选择全新、空的路线输出目录。它不会替你创建容器。不要在正在训练的容器里执行，也不要将已有实验目录作为新输出目录。

先选路线并生成计划，显式指定独立验证集：

```bash
cd /workspace/slime
export ROUTE=naive-c8  # 分别换为 c16、naive-c4、attn-qat-c4
python examples/w4a8_sim/prepare_qwen3_4b.py \
  --model /models/Qwen3-4B-Base \
  --output /workspace/w4a8/plan \
  --checkpoints /workspace/w4a8/planned-checkpoints \
  --gpus 0 --attention "$ROUTE" \
  --updates 50 --eval-interval 5 \
  --eval-path /datasets/validation-50.jsonl
```

准备脚本同时生成 `bf16` 和 `w4a8` 两个 arm；**四条 attention 实验全部取 `arms.w4a8`，C16 也一样**。不要把 `arms.bf16` 当 C16。

该脚本仍有历史默认路径，下面显式修正数据、输出、Ray 和缓存路径，生成可审查的命令与环境文件；此步仍不启动训练：

```bash
python - <<'PY'
import json
import shlex
from pathlib import Path

root = Path('/workspace/w4a8')
plan = json.loads((root / 'plan/training-plan.json').read_text())
arm = plan['arms']['w4a8']
command = arm['training_command']
for flag, value in {
    '--prompt-data': '/datasets/train.jsonl',
    '--save': str(root / 'checkpoints'),
    '--tensorboard-dir': str(root / 'work/tensorboard'),
}.items():
    command[command.index(flag) + 1] = value

env = arm['environment']
env.update({
    'PYTHONPATH': '/quant/slime_plugins:/workspace/slime:/root/Megatron-LM',
    'W4A8_METRICS_DIR': str(root / 'work'),
    'W4A8_CHECKPOINT_ROOTS': json.dumps([str(root / 'checkpoints/.snapshots')]),
    'W4A8_TRAINING_AUDIT_DIR': str(root / 'work/training-audit'),
    'W4A8_ROLLOUT_AUDIT_DIR': str(root / 'work/rollout-audit'),
    'TENSORBOARD_DIR': str(root / 'work/tensorboard'),
    'TMPDIR': str(root / 'work/tmp'),
    'RAY_TMPDIR': str(root / 'raytmp'),
    'TRITON_CACHE_DIR': str(root / 'work/cache/triton'),
    'TORCHINDUCTOR_CACHE_DIR': str(root / 'work/cache/inductor'),
    'W4A8_RUN_NAMESPACE': 'qwen3-4b-base-' + plan['attention'],
})
# 使用独占创建，避免静默覆盖已准备的启动文件。
with (root / 'command.json').open('x') as f:
    json.dump(command, f, indent=2)
with (root / 'env.list').open('x') as f:
    f.write(''.join(f'{k}={v}\n' for k, v in env.items()))
with (root / 'run.sh').open('x') as f:
    f.write('#!/usr/bin/env bash\nset -euo pipefail\n')
    f.write(''.join(f'export {k}={shlex.quote(v)}\n' for k, v in env.items()))
    f.write('exec ' + shlex.join(command) + '\n')
print('Prepared route:', plan['attention'])
print('Review command.json, env.list and run.sh before launch.')
PY
```

这里保留原始 `plan/training-plan.json`，实际启动以修正后的 `command.json`、`env.list`、`run.sh` 为准。生成器的 `formal_launch_ready=false` 不是失败证据，更不能手改为 true 代替运行验证。

人工核对 GPU 占用、模型身份、数据交集、磁盘空间、挂载和全部参数后，才在此专用容器中执行：

```bash
bash /workspace/w4a8/run.sh
```

这一条会真正启动训练。四条路线分别重复准备流程，使用不同宿主输出目录，不能共用同一个 `/workspace/w4a8` 对应的宿主目录，也不要在运行中切换路线。

**不要直接追加调用历史 `generate_attention_commands.py`：** 它要求各路线已有 `command.json` 与 `env.list`，硬编码本机源码/模型路径，并会把 eval 路径重新写为 `validation-50-old-study.jsonl`。它不负责保证独立验证集，也不是准备脚本输出的自动下一步。

### 3.4 比较实验的共同参数和数据边界

| 项目 | 当前四路线协议 |
| --- | --- |
| 初始化 | Qwen3-4B-Base，start0，fresh Adam |
| 长度 | 总计 50 rollout，每轮一次 optimizer update |
| batch | 10 prompts × 2 answers，global batch20，micro batch1 |
| 优化器 | Adam，lr=1e-6，betas=0.9/0.98，eps=1e-8，weight decay=0.1，clip-grad=1.0 |
| 生成 | temperature=0.8，response 上限1024 |
| 上下文 | seq/context length2048 |
| 验证 | step0，然后每5次更新；固定50题，每题1答，top-k1，response 上限1024 |
| 模板 | chat template，`enable_thinking=false` |
| 并行 | 当前单路线单 GPU，TP=PP=CP=1 |

数据要求 `prompt`、`label` 字段，reward 使用 `math`；模型路径名不能替代权重来源与哈希验证。

**已知历史问题：** 现有父目录实验的 `validation-50-old-study.jsonl` 共50题，全部与6961题 train 重合。训练后成绩不能称为独立 holdout，也不是官方 GSM8K test 总体分数。后续新实验应使用已核对 prompt 与 source_index 均无交集的拆分，并记录 SHA256；本机准备过的 `next-run/` 数据不随 Git 仓库自动发布。不要在续跑中换验证集，也不要混合两套验证曲线。

在线 RL 中相同 seed 不保证各路线生成相同训练数据。算法隔离实验优先使用相同初始化、fresh Adam 和同一份 replay 数据；在线结果应明确生成分布也发生了变化。50题中2个百分点只有1题，不能将单次小差异直接解释为稳定优势。

## 4. 输入约束与后续开发

当前低精度 attention 是数值参考实现，而非任意布局/并行的通用 backend：

- THD packed causal self-attention，Q/K/V 的 token 数和 head_dim 匹配，query heads 可整除 KV heads。
- `cu_seqlens` 必须在相同设备上、连续 int32，Q 与 KV 共用 offsets；不支持 padded offsets。
- 显式 `ProcessGroupCollection`，TP=CP=1；不支持 dropout、windowed attention、attention bias 或显式 attention mask。
- 仅 vanilla softmax，不支持 QK clipping、max-logit logging、layer-dependent scaling。
- spec 要求 dense Qwen3 Q/K norm，不支持 MoE / MLA。W4A8 provider 的其他约束仍须满足。
- 实现显式构造每条序列的 `[heads, length, length]` 概率张量并保存反向数据，空间随长度平方增长；不能按 FlashAttention 的显存/速度预期扩大上下文。

推荐改动入口：

1. **修改量化格式/尺度：** 改 `fp8_fake_quant` / `nvfp4_fake_quant`，同时保持 causal、token/head/query、packed sequence 隔离。
2. **修改反向规则：** 改 `_SimulatedAttention.backward`，先写独立公式 reference；不要把“梯度不同”当数学正确性的充分证据。
3. **增加路线：** 同时更新 `ROUTES`、spec factory、provider 路由、准备脚本 choices、相关测试和说明；若继续使用历史 Docker 生成器，也需更新其 `ROUTES`。
4. **扩展到推理侧低精度 attention：** 需要额外接入 SGLang attention，不能只增加 `W4A8_ATTENTION` 环境变量。
5. **替换为高效 kernel：** 先与软件 reference 对齐前向和 dQ/dK/dV，再单独声明数值容差和性能结果。
6. **改结构或参数名：** 检查 checkpoint 转换及 Megatron→SGLang 权重映射；保留权重更新后的量化缓存失效。

每次实验 manifest 至少明确：`attention_mode`、`qkv_format`、`p_format`、`payload_format`、`block_size`、`scale_format`、`scale_scope`、`accumulator_format`、`output_format`、`backward_rule`。当前 `AttentionRoute` / 计划文件只包含其中一部分，不能声称生成器已自动记录完整 manifest。

## 5. 验证与验收

### 5.1 已有测试与覆盖边界

- [test_w4a8_attention.py](tests/test_w4a8_attention.py)：低精度前向/梯度形状与有限性、两个 C4 路线前向相同但梯度不同、causal future 归零与 packed 隔离、naive-C8 的 dV 公式、基本量化输出及路线 metadata。
- [test_w4a8_resume.py](tests/test_w4a8_resume.py)：恢复时归档超界指标、幂等整理、恢复边界不匹配时不改文件。属于手动 CPU 测试，不是真实模型/optimizer/RNG 恢复测试。
- [test_w4a8_dense_plan.py](tests/test_w4a8_dense_plan.py)：计划和相关策略检查。

在已经确认挂载本仓库的**开发容器**中进行定向 CPU 测试，隐藏 GPU，禁用只读挂载中的字节码与 pytest 缓存：

```bash
# DEV_CONTAINER 必须由使用者指定为可用于测试的自有容器
: "${DEV_CONTAINER:?Set a verified development container}"
docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
  -w /workspace/slime "$DEV_CONTAINER" \
  python -m pytest -p no:cacheprovider tests/test_w4a8_attention.py tests/test_w4a8_resume.py
```

这些是执行说明，不代表本文编写时已执行或通过。不要使用裸 `pytest` 启动仓库全部测试；部分测试会启动 Ray、下载数据或进行 GPU 训练。

后续算法改动按需验证：独立 quantizer/边界 → C16-reference 与原生 TE 差异 → C8/C4 前向和梯度 → Attn-QAT 独立公式/STE reference → full-model 接入及 checkpoint 恢复。现有 CPU 测试**不等于**已覆盖完整 TE 数值对齐或独立 QAT 公式证明。

### 5.2 判断训练是否真正成功

| 证据 | 能说明什么 |
| --- | --- |
| `work/train_reward.csv` | 本轮生成的 reward，早于 optimizer 成功写入，行数不能作为完成数 |
| `work/training-audit/optimizer-rank-0.jsonl` | optimizer success、grad_norm、采样主参数是否有限及更新幅度；采样检查不是全参数证明 |
| `checkpoints/checkpoint-policy.json` 的 `last.completed_updates` | 已发布的完整 checkpoint 进度 |
| `work/valid_metrics.csv` | 实际验证点，包括 step0；应与日志和保存策略分别核对 |
| 容器和 Ray 日志 | 实际加载路径、恢复、异常与退出状态；`Up` 不等于训练健康 |

目标50次运行的结束验收：50次连续且成功的更新、last50、step50验证以及正常退出。pending 保存不算提交；rollout_id 从0开始，`step_0000009` 表示完成10次更新。

建议记录 reward、valid、global grad norm、optimizer success、首个 NaN/Inf、checkpoint；诊断时再增加各层 dQ/dK/dV、更新/参数范数比、`||O-O′||` 与 `||D(O)-D(O′)||`，这些诊断项不是当前都已采集。不要把“naive-C4 必须梯度爆炸”写成验收条件。

结果图使用每10轮非重叠分块平均 reward 和真实 validation 点，缺失不补造；纵轴按数据范围留白，并标明数据是否与 train 重合。

## 6. 同一路线中断后如何恢复

`last` 是含 optimizer/RNG 等状态的完整训练 checkpoint；4B 约56.3 GB（52.45 GiB）。`best` 是 HF BF16 weights-only，不能完整恢复训练；导出为 BF16 权重也不意味着评测应移除 W4A8 插件。

恢复前必须确认旧任务不再写同一目录，检查 `checkpoint-policy.json`、`last` 指向和实际文件完整性。使用**已提交 last**，不要选 `.pending-*`，也不要把 best 当 last。

保持原模型身份、路线、数据、总更新目标、batch 和优化器参数，调整原启动配置：

```text
--load <已提交 last 所指向的完整 snapshot>
--start-rollout-id <last.completed_updates>
--num-rollout 50                         # 总目标，不是额外再跑50次
--tensorboard-dir /workspace/w4a8/work/tensorboard-resume-<边界>

W4A8_RESUME_METRICS=1
W4A8_METRICS_DIR=/workspace/w4a8/work
W4A8_TRAINING_AUDIT_DIR=/workspace/w4a8/work/training-audit
```

`--save` 仍为原运行的 checkpoints 根目录；`--hf-checkpoint` 和 tokenizer 仍指向原 Base。不添加 `--finetune`、`--no-load-optim`、`--no-load-rng`，不擅自改变 step0/eval 协议。

`train_best_last.py` 在 Ray 启动前调用 `metrics.prepare_resume`：

1. 校验 load 与该运行已提交 last 一致，tracker、policy、start 对齐，并存在 rollout 数据游标。
2. 保留恢复边界之前的连续 reward、验证记录。
3. 若日志超前于 checkpoint，先将被修改文件的完整原文统一归档到 `work/resume-before-<start>-<时间戳>.json`，再移出超界 reward/audit；不靠放松连续性检查掩盖缺失。
4. 真正恢复 optimizer/RNG、模型和数据游标仍由 Slime/Megatron 完成，metrics 整理不能替代 native 恢复。

例如完成25次的 checkpoint tracker 为24，下一 rollout_id 是25。需要在日志中确认 checkpoint 成功加载，再确认首次新 optimizer 更新；仅有容器 running 或 metrics 归档成功不够。重新生成的第26轮回答不保证逐 token 重现失败尝试，失败尾部不得重复计入曲线。

## 7. 发布与协作边界

- 四路线的项目侧实现都在本仓库，运行仍依赖固定环境中的 Megatron、TE、SGLang；不需要复制旧 0.6B Slime 原型。
- 提交时显式检查未跟踪的 attention、metrics、测试和命令生成器，不能只提交 tracked diff 而漏掉核心文件。
- 模型、checkpoint、运行目录、缓存与本机 `/data4` 配置不应不加筛选地进入 Git；需要复现时发布经过审查的配置、数据来源/哈希和环境信息。
- 本文示例不修改历史生成器，也不覆盖既有实验配置。源码只读挂载不等于冻结版本：宿主后续修改仍可能被新进程读到，已运行 actor 则不会自动热更新。
- 共享 GPU 不意味着可以清理他人的进程。启动、停止、删除、重建容器或清理数据需要对应授权；失败只报告，不自动重启，不使用全局清理命令。
