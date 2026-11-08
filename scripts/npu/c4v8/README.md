# fa/30B-W16A16C4V8

Qwen3-30B-A3B 的 FlashAttention 低精度正向模拟及 GRPO 接入。主干保持 BF16，注意力 Cube 路径使用 NVFP4 编码与反量化，Vector 路径采用 MXFP8 量化边界；反向使用高精度 STE。

本提交保存实现、配置及分层验证证据，**不表示已完成 100 步 C4V8 训练**。导出快照时，30B 作业仍在首轮采样，实际成功 GRPO 更新为 0。用户随后要求提交后暂停全部任务；恢复运行需要用户明确说开始。独立完成流水线只完成了准备，没有执行，也没有创建定时任务或自动恢复。

## 精度定义

| 标记 | 本实现的含义 |
|---|---|
| W16A16 | BF16 主干权重与主干计算输入；保留原训练栈的 FP32 主参数、Adam 状态及必要累加。 |
| C4 | 注意力 QK、PV 相关操作数的 NVFP4 正向 QDQ：E2M1、nearest-even 舍入、16 元素分块、E4M3FN block scale，**FP32 per-row upper scale**。 |
| V8 | E4M3FN 数据、E8M0 power-of-two scale 的 MXFP8，32 元素分组；在明确的 Vector 边界量化，并在实验定义的 max/sum 二叉归约树各层重新量化。 |
| 反向 | identity STE 和高精度自动微分／闭式导数；查询块重算以控制保存的中间状态内存。不是低精度 GEMM 反向。 |

这里的 C/V 表示注意力中的 Cube/Vector 计算，**不是 KV cache 的 K/V 位数**。存储中的 KV cache 保持 BF16。W16A16 也不表示把全部主干激活量化成 4/8 位。

本实现选择 per-row upper scale，使独立请求、GQA 头分组和查询块重算遵循同一缩放规则。它不能被描述为所有 NVFP4 工具链默认使用的 per-tensor upper-scale 配方。矩阵累加、基本算术和 STE 在 FP32 中模拟，在规定边界执行量化与反量化；这不是 910B3 原生 NVFP4 Tensor Core 加速，也没有实现完整的低精度 FA 反向。

当前数学策略固定为 causal、query block 1024、key block 128。结构性 mask 在分数量化前施加，并独立保留；逻辑 key 分块从 token 0 开始，不依赖物理 KV 页。实现 ID 为 `fast_attention_fused_forward_ste_staged_decode_v2`，数学策略与实现／工具链 SHA 分开记录。

## 实现范围

- Megatron 通过独立 `--spec` 替换 core attention，覆盖 SBHD、packed THD、GQA 与因果边界，保持 BF16 主干。
- SGLang 通过独立 Python shim，在 HTTP parent 和每个 scheduler 显式安装同一数值 backend；prefill、decode、缓存前缀和逻辑 KV 映射有实际适配验证。
- 融合量化组件与 staged decode 减少运行开销。相同 KV 长度的多个 decode 请求可以合批；请求间缓存仍独立。
- `whole_decode.py` 被现有源码指纹引用，但其 monolithic 候选**未被路由到运行路径**。不要把它当作已启用的生产优化。
- 标准 SG offload、FP32 Adam、编译服务数量限制及已有 NPU 生命周期修复保留。训练和推理分别使用经过验证的 allocator 配置。

## 训练与评测配置目标

| 项目 | 配置目标，尚未完成 |
|---|---|
| 初始模型 | 原版 Qwen3-30B-A3B，从 0 开始，不沿用 BF16 已训练权重 |
| 训练数据 | 清理后的 DAPO-Math-17k，17,182 道题 |
| 方法 | 真实数学奖励的 GRPO，保留非零组内奖励方差筛选 |
| 模式 | 非思考模式，最大回答 4,096 tokens，总序列上限 8,192 |
| 并行 | 8 张 910B3，训练 TP4 × PP2、EP4；SG TP8 |
| 更新 | 5 次 rollout，每次 5 题 × 16 回答，global batch 4，即每轮 20 次实际更新，目标共 100 次 |
| 保存 | 每 5 次成功 optimizer 更新保存完整 HF 权重，共 20 个训练检查点 |
| 验证 | AIME2024 全部 30 题，step 0、5、…、100，共 21 点，统一非思考 greedy 配置 |

采样量、生成调用和前向次数不能计作 optimizer 更新。第一轮需先收集 80 条通过分组筛选的真实回答，才开始更新。

每 5 步检查点是 HF-only，不含 Adam 状态、RNG 或完整数据源游标。它们不能精确恢复原优化轨迹。失败后如果选择从 HF 恢复，必须另行记录优化器重建、游标重启和实际已持久化更新；不能把丢失或不同模型的更新相加。

## 已验证到哪一层

| 层级 | 已有证据 | 不能据此声称 |
|---|---|---|
| 数值组件 | 真实 NPU 的 NVFP4 payload/scale/value 检查，MXFP8/Vector 正式 API 60 项检查；最终路径另核无意外 Tensor CPU fallback | 整个 30B RL 已训练完成 |
| 注意力整体 | Q1024 前向、STE 梯度、查询块重算及 staged/generic 结果检查 | 所有 CPU/NPU BLAS 在任意边界逐位相同 |
| 训练适配 | 真实 pinned Megatron 单层 MoE，SBHD/THD，有限前后向及一次 Adam 参数变化 | 这次小模型更新是 30B 的 GRPO 更新 |
| 训练／推理接口 | 相同 Q/K/V 经过实际 SG adapter 与 Megatron adapter，因果、前缀、GQA、请求隔离检查 | 两个完整模型的 logprob 已逐项一致 |
| 原版 30B SG | TP8 实际 prefill/decode、24 个短请求、有限且 token 对齐的 logprob、实际 backend 与资源释放审计 | 30B 训练反向或收敛已经通过 |
| 评测／收尾 | 30 题 CPU 奖励检查；时限与 owned cleanup 测试；完成流水线 CPU 10 项测试 | 已经得到 21 个真实验证点 |

历史 eager 路径曾出现 Tensor CPU fallback；旧严格检查失败的证据保留。最终通过的证据只支持其实际测过的源码、策略与范围，不把早期失败重新标成成功。当前完整模型 trainer-versus-SG logprob 一致性仍标记为 unknown。

## 性能数字的边界

原版 30B、并发 4、短 16-token 输出的对照测量中，合批后聚合吞吐约 **5.116 tokens/s**，相同短测旧桥约 **3.050 tokens/s**。这些是四个请求合计的吞吐，不是每个请求的速度。

较长生成／采样另有约 **2.8 tokens/s** 的观测，其长度和阶段与上述短测不同。不能把两者混成一套同条件基准，也不能把 5.116 tokens/s 宣称为整个训练过程的持续吞吐。完整训练还包含严格奖励筛选、prefill、反向、优化器、磁盘导出及权重重载。单层 attention 的微基准同样不能当成完整模型速度。

## 对照实验的限制

旧 BF16 100 步实验在第 40 步后曾从 HF 权重重建 Adam，并重启数据源游标。该变化是比较时必须披露的混杂因素。本次 C4V8 配置从官方原始模型开始，不能把旧 BF16 曲线直接解释为唯一变量只有 C/V 精度的严格对照。README 不从聊天硬编码旧准确率，也不预先宣称量化收益或 AIME 提升。

## 源码、证据与后续执行

最小功能依赖、原始哈希绑定的额外文件和必要 setup 脚本见 `MINIMAL_DEPENDENCIES.md`；原部署 86 项审计清单见 `SOURCE_INVENTORY.snapshot.tsv`。路径迁移和导出适配会产生新的源码身份，原始证据不能改写成新版本已验证。

独立 completion 代码可以在用户明确恢复后，用同一运行身份执行完成检查、全量有限值／SHA 验收、21 点评测和绘图。它目前仅 prepared，未执行；旧 v1 计划已保留并标记 superseded。测试图片带有 `CPU FIXTURE — NOT MODEL RESULTS` 水印，不能作为训练曲线交付。

本提交不包含模型权重、数据全集、虚拟环境、编译缓存、SSH 配置或认证信息。原始部署路径与运行 PID 仅属于内部审计，不应作为公开默认配置。用户提交后暂停的要求优先于所有计划中的后续步骤。

## 归档结构与恢复前提

这是当前开发状态的源码归档，尚非一键部署发行版。主仓库变更包含 Ascend 运行、HF 保存、真实更新记录和内存生命周期支持；`workspace/` 下的路径对应原 `/root/hby/` 布局。不得直接覆盖正在运行的工作区。

- `dependency-patches/` 保存每个 pinned HEAD 的完整补丁。先读其 README，禁止与旧 docker/NPU 补丁重复叠加。
- `environment-versions.json` 是实际 Python 包版本快照，不是完整 wheel 锁或 CANN 安装包；二进制环境与模型、数据须另备。
- `activate-slime.example.sh` 是环境模板，部署时应审核后生成工作区的 `activate-slime.sh`；通信地址通过 `HCCL_IF_IP` 提供。
- 只提交 `launch-gate.pending.json`。旧 accepted gate、PID 绑定计划和训练运行目录不作为默认配置；新部署必须针对实际源码、数据和依赖重新验收组装 gate。
- 旧生成器 `prepare_recipe.py`、`prepare_evaluation.py` 及对应生成器测试未导出，以免覆盖后来修复的权威 prepared recipe/evaluator；不要用历史 86 文件清单冒充新导出身份。
- `original-source-export-manifest.json` 与 `SOURCE_INVENTORY.snapshot.tsv` 均为原部署来源记录。导出只对通信地址做环境变量化，数值核心逐文件保持原 SHA；归档后的启动代码仍需重新核验。
- `evidence/npu-evidence-summary.json` 是去除运行身份、主机路径和请求内容后的字段摘要，保留原报告 SHA 及 owned cleanup 结果，包括旧对照退出失败。摘要不替代原始报告。

如用户后续明确恢复，应先确认模型和 DAPO/AIME 文件的实际 SHA、可用资源、依赖补丁与新的门禁，再手动选择新的运行目录；不要复用旧 PID 或自动恢复训练。当前完成流水线和 21 点评测均未执行。

提交前本地 CPU 回归：共 62 项，58 项通过、4 项因 macOS 不具备 Linux `/proc`/pidfd 条件而跳过；无失败。详情见 `evidence/export-cpu-checks.json`。NPU 历史证明与本地 CPU 检查的范围分别记录。
