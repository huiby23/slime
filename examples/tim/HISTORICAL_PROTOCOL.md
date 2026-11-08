# 历史实验与统一入口

本文件说明协议来源和边界。历史输出、冻结源码和已有报告保留原样；新入口创建新的实验目录。

| 历史实验 | 可用的统一配置 | 必须保留的区别 |
| --- | --- | --- |
| 0.6B BF16 GRPO 500步，LR1e-6 | `bf16_lr1e6.json` | 2784训练题、512 holdout、16×8、每25步评测 |
| BF16 LR1e-3 | `bf16_lr1e3.json` | 历史作业由用户在300步停止；500是原计划，不能标为完成500步 |
| BF16 LR1e-5 | `bf16_lr1e5.json` | 额外评测/保存前10步；采样时间和 RNG 轨迹不保证与旧 LR1e-6 逐题配对 |
| BF16训练、模拟NVFP4 rollout | `nvfp4_rollout.json` | 在线评测同为模拟NVFP4；不等于干净BF16能力评测 |
| 固定相对高斯噪声 sigma0.05 | `fixed_noise_s005.json` | clean trainer，固定偏移只加到 sampler；评测恢复 clean weights；未启用 Score Centering |
| NVFP4四sampler续训 | `nvfp4_rollout4_fresh.json` 仅提供新训拓扑 | 历史从111步恢复，经112/113步验证后续训；路由/副本/采样 RNG 变化，不能视为单卡连续轨迹 |
| 更早的NVFP4配方训练 | 原有 `examples/nvfp4_sim` 保留；新 `nvfp4_train_rollout.json` 可作扩展 | 新预设统一到当前16×8/500/筛选数据，不声称等于旧训练配方实验的所有条件 |

目前发布的统一入口只处理从初始模型开始的新训练。
没有将历史多卡迁移的专用容器、GPU编号、checkpoint路径和自动调度逻辑混入默认baseline。
真正的续训需要共同验证模型、Adam moments/step、调度器、trainer RNG、数据游标；sampler RNG和batch布局变化应另行记录。

新代码从已有运行实现提取：

- BF16：LR1e-5运行时的best/last/early保存及完成步数评测轴，optimizer审计改为读取参数。
- NVFP4：四sampler运行时的加载/缓存失效/重建审计，量化算术继续使用共享recipe。
- 固定噪声：完成500步实验的bank与clean-evaluation路径，去掉私人路径和固定seed断言。

固定噪声控制文件增加了原子替换的inode识别，防止共享文件系统上两次快速切换的mtime相同而漏恢复。
这属于整理后修复，有独立CPU回归测试；历史实验原件没有被修改，也未声称历史结果因此有误。

历史seed42单次运行及不同在线评测精度不足以完成因果归因。
后续若要把性能退化归因于训推不一致，应增加同wrapper零噪声、固定checkpoint共用BF16评测、相同生成token重算、独立seed等控制。
