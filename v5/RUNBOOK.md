# Redstone v5 修复架构与运行说明

这次修复将完整方块状态从旧 VAE 的五字段输出中独立出来，并重建输入数据与评测链路。旧几何 VAE 和 Flow 仍是 128 类、64³ 体素模型；本次新增的状态分支根据几何、图片和已知状态预测注册表中的合法状态。合法状态预测不等于电路功能正确，当前没有 Minecraft 服务器物理执行验证。

## 数据与模型边界

| 环节 | 输入与输出 | 约束 |
|---|---|---|
| 原始文件归档 | 原始字节 → SHA-256 快照、来源清单 | 不改原件；实体和方块实体 NBT 保存在原件 |
| 文件解码 | Litematic、Sponge、旧 Schematic、Java Structure → 完整调色板和位置 | 多区域、负尺寸、跨 long 位流、VarInt；歧义或损坏文件明确隔离 |
| 状态规范化 | 原始名称、属性、DataVersion → 完整注册表 ID | 固定 Java 1.21.4；保留原始字符串；未知记为 -1 |
| 训练样本 | 原始体素 → 64³ 裁块 | 保存绝对来源坐标、padding、有效和实测属性掩码；大型输入有资源上限 |
| 数据分组 | 来源、几何、旋转、重复裁块 → train/validation/test | 历史训练来源不得进入真正留出集 |
| 图片条件 | 规范化状态 → 旋转渲染 → DINO | 输入、资源包、渲染器、视角、尺寸、PNG 均有校验；训练和推理共享归一化 |
| 几何生成 | 图片、已知空间 → Flow latent → 冻结 VAE 的方块类 | 保留预训练 latent 坐标系；已知空气也受约束 |
| 完整状态预测 | 几何、图片、已知完整状态 → StateRefiner | 只预测对应方块合法的状态；未知空间的状态值不得泄漏 |
| 导出 | 完整状态 ID → Litematic | 已注册状态按目标版本规范化；未知的已知状态保留原文并列入结果清单 |

旧五字段状态只用于已训练的几何条件器兼容输入，不再承担完整状态存储或最终生成。旧 VAE 输出中的状态 logits 在新管线中不使用。OTHER 63/127 作为代理类别单独标注；无法表达的新方块不会伪装成受支持标签用于状态监督。

## 服务器位置

- 新代码：`/data/redstone_v5/code/mcfix`
- 全量数据：`/data/redstone_v5/dataset`
- 阶段状态：`/data/redstone_v5/pipeline_status.json`
- 原始训练备份：`/data/redstone_v5/backup_20260913`
- 生产源码备份：`/data/redstone_v5/production_backup`
- 通过检查后才生成的模型选择清单：`/data/redstone_v5/active.json`

本地 `production_changes.json` 记录远程改动路径和修复前后哈希，`production/` 保存修复源码。`mcfix/` 与 `tests/` 是可独立安装的新模块和回归用例。

## 运行与验证

远程 Python：`/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python`。调用模型前应设置：

```bash
export PYTHONPATH=/home/linux/redstone_ai/TRELLIS:/data/redstone_v5/code
export ATTN_BACKEND=sdpa
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
```

查看执行状态：

```bash
cat /data/redstone_v5/pipeline_status.json
```

回归测试：

```bash
cd /data/redstone_v5/code
/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python -m unittest discover -s tests -v
cd /home/linux/redstone_ai/TRELLIS
/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python -m unittest test_mc_inpaint -v
```

单独校验源数据映射（从原始快照重新解码并核对位置）：

```bash
/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python -m mcfix.verify_data \
  --dataset /data/redstone_v5/dataset --limit 200 \
  --out /data/redstone_v5/data_verification.json
```

生成入口支持直接读取当前模型选择清单：

```bash
/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python -m mcfix.generate \
  --active /data/redstone_v5/active.json \
  --image /absolute/path/input.png --out /absolute/path/output.litematic
```

也可显式提供几何配置、几何检查点、状态检查点和 VAE decoder，详见 `python -m mcfix.generate --help`。需要保留已有结构时，同时提供 64³ 的 `--known-litematic` 和布尔 `.npy` 的 `--known-mask`；掩码采用 `[X,Y,Z]`，必须明确哪些空气格已知。不能把“当前是空气”直接当作“未知”。

输出是方块状态文件。容器物品、告示牌文本、实体、命令方块 NBT 等没有通过神经网络生成；原始来源快照保留这些信息。未注册的 mod 方块若被保留，需要相应游戏环境才能解释。

## 修复训练的验收

本轮纠正训练抽取 96 个来源组尽量多样的样本，每个 4 个旋转；这不是全量语料训练。全量重建用于完整归档、分组和核对。状态分支先训练 3000 步，若未超过默认状态基线则延长到 6000 步；仍不达标则停止发布。几何分支使用纠正后的图像预处理和来源隔离，以较小学习率训练 1000 步，并在同一留出集上和旧检查点比较；旧检查点更好则保留旧几何模型。训练器的 EMA 必须在微调加载权重后同步初始化；本次已修复并以失败转通过的回归验证。纠正重跑同时比较原始权重与 EMA 权重。

最终模型选择清单包含各检查点、数据哈希、状态与几何验证分数和独立测试结果。状态预测的独立准确率是在给定真实几何条件下测量；端到端准确率包含几何生成误差，两者必须分别解读。已知空间严格保持的测试也不等于未知结构质量达标。

不要直接恢复旧 v4 supervisor：旧数据及其五字段监督已标记退役。需要回退某项代码时，先比对 `production_changes.json` 的当前哈希，再从对应备份恢复，以免覆盖后续人工修改。历史检查点与数据仍在，未删除。

新管线将生成质量标为 experimental_generation。状态分支通过的验收仅针对给定真实几何的完整状态预测，不能据此声称整栋建筑或红石功能已经可靠。最终完整状态指标按注册表 ID 比较，避免旧类别中重复名称造成虚假的状态错误；旧 128 类编号准确率仍单独保留。
