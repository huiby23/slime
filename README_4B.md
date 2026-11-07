# Qwen3-4B 低精度训练研究

本项目在 Slime 中实现线性层 NVFP4 W4A8 数值模拟，并与原生 BF16 训练对照。
新运行默认 50 次 rollout，每 5 次 rollout 进行一次 50 题 valid，并在训练开始前执行一次 valid。
rollout 使用 4 道题、每题 4 个回答；valid 使用旧研究相同的 50 道题、每题 1 个回答。
历史正式实验的 100 次更新和最终评估已完成。正式实验的原始代码版本保存在标签
`qwen3-4b-w4a8-v1-20261030`；后续整理使用新的提交，保留完整历史。

## 从哪里开始

- [4B 实验指南](examples/w4a8_sim/QWEN3_4B.md)：精度范围、训练设置、结果和 checkpoint 保存恢复。
- [脚本索引](examples/w4a8_sim/README.md)：准备、验证、训练、监控各入口的用途。
- [正式配置与结果索引](examples/w4a8_sim/experiments/qwen3_4b_20261030.json)：
  实际命令、环境、数据哈希、原始代码版本和五次评估。

## 代码结构

| 目录或文件 | 用途 |
| --- | --- |
| `slime_plugins/w4a8_sim/recipe.py` | FP4/FP8 量化与线性层前向、反向计算 |
| `slime_plugins/w4a8_sim/megatron.py` | 接入 Megatron 训练层 |
| `slime_plugins/w4a8_sim/sglang_linear.py` | Dense/MoE 共用的推理线性层接口 |
| `slime_plugins/w4a8_sim/sglang_models/qwen3.py` | Qwen3 Dense 推理接入与权重缓存失效 |
| `slime_plugins/w4a8_sim/checkpoint_policy.py` | 每次更新保存完整 last，评估改善后更新 best |
| `examples/w4a8_sim/` | 实验准备、检查、启动和资源观察脚本 |
| `tests/test_w4a8*.py` | 数值、适配器、保存恢复及两组配置的一致性检查 |
| `slime/`、`train.py` | Slime 原有训练框架 |

`slime_plugins/nvfp4_sim/` 提供共用的 NVFP4 量化和 checkpoint 工具，
`examples/nvfp4_sim/export_diagnostic_hf.py` 提供 best 权重导出；这些是当前 4B
实现的实际依赖。30B MoE 的准备说明位于
[单独的 MoE 文档](examples/w4a8_sim/docs/moe_30b.md)。

## 精度范围

线性层权重使用 NVFP4，激活使用 FP8 E4M3，反向输出梯度使用 FP8 E5M2。
矩阵乘法在 A100 上以 FP32 模拟，输出回到 BF16；主参数和 Adam 状态保持 FP32。
默认 FA 的 QK、softmax、PV 保持 C16 基线精度；可选的 `--attention`
选择 `naive-c8`、`naive-c4` 或 `attn-qat-c4`，分别启用 FP8 E4M3 或 NVFP4
E2M1 的 Q/K/P/V fake quant 与对应 backward。四条路线仍是 A100 上的数值模拟，
不是原生 FP8/FP4 kernel；详细格式和实验字段见仓库根目录的 `说明.md`。

## 日常版本管理

在仓库根目录执行：

```bash
git status
git diff
git log --oneline -10
```

提交前通过 `git diff` 确认本次变更范围。
