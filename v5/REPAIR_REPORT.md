# Redstone 修复结果

代码与数据链路修复已部署，纠正训练和评测已完成。完整状态分支在给定真实几何时有明确提升；端到端建筑生成仍标记为实验状态，不能据此认定建筑或红石电路已经可靠。

## 已完成的工作

- 修复完整 blockstate 表达、旋转、版本规范化和导出，新增独立状态分支；旧 VAE 的五字段输出不再作为最终状态。
- 修复大调色板、跨字位流、多区域与负尺寸、Sponge v3、旧 metadata、非零空气和错误坐标输入。
- 统一训练/推理图片预处理；用来源、几何、旋转和重复裁块分组隔离留出集；保留原件、未知状态和问题分类。
- 修复已知空气保持、未知值泄漏、边界指标、图片索引，以及微调时 EMA 未同步加载权重的缺陷。
- 更新 18 个原有源码文件，部署 `/data/redstone_v5/code/mcfix`。修复前源码、数据及完整 32,000 步训练检查点均有备份。

逐项原因和测试对应见 [修复清单](/home/sensen/Desktop/redstone_full_fix/修复清单.md)，运行方式见 [架构与运行说明](/home/sensen/Desktop/redstone_full_fix/架构与运行说明.md)。

## 数据与回归验证

| 检查 | 结果 |
|---|---|
| 原始来源去重归档 | 15,398 份；逐份 SHA-256 一致 |
| 成功解析 / 隔离 | 14,505 / 893；隔离原件保留 |
| 64³ 裁块 | 73,941；没有缺失文件 |
| train / validation / test 裁块 | 71,961 / 1,141 / 839 |
| 来源位置抽检 | 200 个裁块、186 份来源、33,060,512 个原始单元，零差异 |
| 注册状态抽检 | 52,124,244 个单元，零差异 |
| 自动回归 | 新模块 27 项 + TRELLIS 12 项 = 39 项通过 |
| 完整注册表测试 | 27,866 个状态往返、旋转及文件导出 |

隔离中包含 766 份方块字符串不合法的来源、45 份超过裁块资源上限的来源，其余为不支持、损坏或结构不合法的容器。没有把它们静默改成空气。全量来源被归档和分组，但源位置逐单元对照是 200 个裁块抽检，不能写成全量逐格独立复验。

## 模型结果

状态分支训练 3,000 步，使用 72 个训练来源组的 288 张图，验证使用 12 个来源组的 48 张图。在 29,583 个未知且有实测属性的状态单元上：

- 默认状态基线：63.87%
- 完整状态分支：73.42%，提高 9.55 个百分点

以上是在真实几何条件下的状态预测。几何 VAE 与 Flow 的误差在端到端评测中另计。

EMA 初始化缺陷修复后重新训练 1,000 步，并使用相同 8 个验证来源组、50 步采样比较：

| 几何检查点 | 未知区域 128 类编号准确率 | 占据 IoU |
|---|---:|---:|
| baseline | 5.97% | 21.01% |
| ema | 6.27% | 21.20% |
| raw | 5.70% | 21.89% |

本轮选择 `ema`，选择依据只来自验证集。最终独立测试的 8 个来源组：平均未知类别编号准确率 2.83%、完整状态准确率 2.96%、占据 IoU 28.99%。所有测试样本的已知方块和状态均严格保持。

这些均为逐样本平均；类别编号与完整状态指标使用不同分母，而且旧类别中存在重复名称，不能直接相减。完整状态按规范注册表 ID 比较。样本量较小，且没有游戏服务器物理或红石功能验证。

## 交付与实际入口验证

模型清单位于服务器 `/data/redstone_v5/active.json`；完整数据位于 `/data/redstone_v5/dataset`。生成命令：

```bash
export PYTHONPATH=/home/linux/redstone_ai/TRELLIS:/data/redstone_v5/code
export ATTN_BACKEND=sdpa
export HF_HUB_OFFLINE=1
/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python -m mcfix.generate \
  --active /data/redstone_v5/active.json \
  --image /absolute/path/input.png --out /absolute/path/output.litematic
```

补全已知结构时同时传 `--known-litematic` 和布尔 `--known-mask`。实际公开入口已输出并重新读取一个 64³ Litematic：131,072 个已知单元零状态变化，全部 262,144 个输出单元可由目标注册表解析，DataVersion 为 4189。该检查证明入口及导出契约，不额外证明生成质量。

新导出范围是方块及状态；实体、容器物品、告示牌文本等 NBT 留在原始快照中，不由本模型生成。旧几何头仍为 128 类，仅对应 123 个不同的已注册方块名称并包含 OTHER 代理，尚未覆盖整个 Minecraft 方块集合。

## 实际渲染检查

公开入口样例中，已知半幅保持正确，但新增半幅把参考玻璃面生成了大量灰色和杂色方块，与低类别准确率一致。不能将文件可读取、状态合法或形状大致连续当作生成内容正确。

[参考结构](/home/sensen/Desktop/redstone_full_fix/reference.png) · [已知部分](/home/sensen/Desktop/redstone_full_fix/known.png) · [实际生成](/home/sensen/Desktop/redstone_full_fix/completion.png) · [生成的 Litematic 文件](/home/sensen/Desktop/redstone_full_fix/completion.litematic)

## 可追踪证据

- [全量数据统计](/home/sensen/Desktop/redstone_full_fix/evidence/dataset_summary.json)
- [来源位置抽检](/home/sensen/Desktop/redstone_full_fix/evidence/data_verification.json)
- [最终模型清单](/home/sensen/Desktop/redstone_full_fix/evidence/active.json)
- [状态验证](/home/sensen/Desktop/redstone_full_fix/evidence/state/best_validation.json)
- [最终测试结果](/home/sensen/Desktop/redstone_full_fix/evidence/test_final_eval.json)
- [公开入口与导出验证](/home/sensen/Desktop/redstone_full_fix/evidence/public_cli_verification.json)
- [新模块回归日志](/home/sensen/Desktop/redstone_full_fix/evidence/regression_tests.log)、[TRELLIS 回归日志](/home/sensen/Desktop/redstone_full_fix/evidence/trellis_regression_tests.log)
- [原有源码修复前后哈希](/home/sensen/Desktop/redstone_full_fix/production_changes.json)、[已部署源码核对](/home/sensen/Desktop/redstone_full_fix/evidence/installed_verification.json)、[新模块逐文件哈希](/home/sensen/Desktop/redstone_full_fix/evidence/package_verification.json)
- [实际运行依赖版本](/home/sensen/Desktop/redstone_full_fix/evidence/runtime_versions.json)

原训练备份：`/data/redstone_v5/backup_20260913`；原源码备份：`/data/redstone_v5/production_backup`。旧 supervisor 已退役，不会自动重启旧数据训练。
