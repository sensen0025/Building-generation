# ARCHITECTURE — `/data/redstone_v4` v4 Inpaint 训练流程

> 服务器 `175.155.64.171:22173` · 更新 2026-09-11
> 历史/基线资产见 `/home/linux/redstone_ai/PROJECT_MAP.md`。

---

## 1. 目标

在已有 v2 小建筑模型（`ss_flow_mc_rot_400k`，400k steps）基础上，**微调**一个 v4 inpaint 模型：
新增 **6062 个从大型建筑切割出的切片**样本，使其支持**局部补全（inpaint）**。
总训练 **200,000 steps**（batch 4，3090 24GB）。

---

## 2. 目录结构

```
/data/redstone_v4/
├── sources/            9421 个上传的源文件（.litematic/.schem）+ qualified.json    639M  [输入]
├── qualified.json      源文件筛选结果（MD5 + 元数据）                              30M
├── code/               从 TRELLIS 移植的确定性代码（勿改）                        168K
│   ├── decode.py          体素解码：输出索引 [y,z,x]，dims (W,H,L)
│   ├── nbt.py             NBT 解析
│   ├── classifier.py      方块 → 128 类
│   ├── cutter.py          大建筑切片
│   ├── legacy_names.py    旧版方块名兼容
│   └── _build_v4_ingest.py  ★ ingest 主脚本（含轴转置，见 §5）
├── work/               管线脚本 + 中间产物 + 配置                            ~9G
│   ├── _redo_v4.sh        ★ 全流程编排（7 个 Phase）
│   ├── _compact_ingest.py     压缩数组（只保留 ok 样本）
│   ├── _export_new_litematics.py  回写 litematic 供渲染
│   ├── _encode_merge_v4.py    VAE 编码 + 4x 旋转增强 + 合并旧数据
│   ├── _assemble_render_all.sh    symlink 组装条件图目录
│   ├── _completion_smoke.py / _eval_v4_inpaint.py   补全评测
│   ├── _rh_flow_v4_supervisor.py  ★ 200k 训练守护进程
│   ├── _make_v4_config.py     生成训练配置
│   ├── ss_flow_mc_v4_comb.json  ★ 200k 训练配置（已修正 normalize）
│   ├── ss_flow_mc_v4_smoke.json 4-step 冒烟配置
│   └── v4_init.pt          ★ 微调起点（400k EMA + 14 个 voxel_conditioner 张量）197M
├── litematics_new/     6062 个重建 litematic（供渲染）                             57M
├── render_new/         6062 张条件渲染图（PyMinecraft/node 渲染）                 ~554M
├── render_all/         symlink 汇总：旧 rot 21296 + 新 6062×4
├── train/              ★ 合并后的训练数组                                         39G
│   ├── ss64_all.npy        (45604,64,64,64) uint8   体素
│   ├── ss64_state_all.npy  (45604,64,64,64) uint16  状态
│   ├── ss_latent_all.npy   (45604,8,16,16,16) f32   latent
│   └── latent_norm_all.npz 合并后重算的 mean/std
└── out_v4/             训练输出（ckpts + log.txt），由 supervisor 创建
```

---

## 3. 流程（`_redo_v4.sh`，7 个 Phase）

| Phase | 动作 | 产物 | 耗时 |
|---|---|---|---|
| 0 | 清理旧错误产物 | — | 1s |
| 1 | **ingest**：解码源 → 切割 → bbox 重拟合 → 居中 64³ → alias 映射 → 打包状态 | `work/ss64_new.npy` `ss64_state_new.npy` + manifest | ~12min |
| 2 | **compact**：只保留 ok 样本 | `work/ss64_new_c.npy` `ss64_state_new_c.npy` `ingest_manifest_ok.json` | ~12s |
| 3 | **export + encode/merge（并行）**：回写 litematic；VAE 编码 + 4x 旋转 + 合并旧数据 + 重算 norm | `litematics_new/` ；`train/*.npy` | ~35min |
| 4 | **render**：3 分片 node 渲染 | `render_new/` 6062 张 | ~2h |
| 5 | **assemble**：symlink 旧 rot 图 + 新图（每个新样本 4 个旋转复用同一张 0° 图） | `render_all/` | ~1min |
| 6 | **smoke**：4-step 训练冒烟 | `smoke_out/` | ~3min |
| 7 | **baseline eval + 启动 supervisor** | `baseline_eval.json` → `out_v4/` | 立即 |

启动命令：
```bash
cd /data/redstone_v4/work && nohup bash _redo_v4.sh > redo.log 2>&1 &
```

---

## 4. 训练配置要点（`ss_flow_mc_v4_comb.json`）

| 项 | 值 |
|---|---|
| 模型 | `SparseStructureFlowModel`，resolution 16，in/out 8ch，512 model_channels |
| 条件 | `cond_channels` 1024（DINOv2-large），`voxel_cond_classes` 128 |
| 数据 | `MCFlowImageDataset`，`up_axis: 1`，`inpaint_probability: 0.3` |
| 图像 | `image_size: 518`，`image_dir: /data/redstone_v4/render_all` |
| 训练 | `max_steps 200000`，`lr 5e-5` AdamW，CosineAnnealing，EMA 0.9999 |
| CFG | `p_uncond 0.1`，`inpaint_image_drop_probability 0.25` |
| 起点 | `finetune_ckpt.denoiser = /data/redstone_v4/work/v4_init.pt` |
| 保存 | `i_save 2000` |

**normalize 必须是合并后统计量**（旧版 `ss_flow_mc_inpaint_v4.json` 误用旧 VAE 统计量，已修复）。

---

## 5. ⚠️ 轴约定（曾导致严重 bug）

- `code/decode.py` 输出体积索引为 **`[y,z,x]`**，dims 为 `(W,H,L)`
- 训练数据必须为 **`[x,y,z]`（y = 垂直轴）**
  ```python
  # 正确
  sl = np.transpose(vol, (2, 0, 1))   # [y,z,x] -> [x,y,z]
  # 错误（曾使用）→ 得到 [x,z,y]，高度与长度互换，建筑"躺倒"、相机拍到顶/底
  sl = np.transpose(vol, (2, 1, 0))
  ```
- 验证方法：`EreborTower` y 向内容跨度应为 **31**（错误时为 43）；litematic 高度 37
- `litemapy.Region(width, height, length)` 索引顺序即 `(x, y, z)`
- 旋转增强：`rot_y_64(a) = np.transpose(a[:, :, ::-1], (2,1,0))`，绕 y 轴（作用于 `[x,y,z]`）

---

## 6. 类别映射

128 类词表由**冻结 VAE** 定义，**不可扩展**（否则需重训 VAE ~4.3h）。
新数据中的陌生方块通过 **alias 映射** 归入已有 128 类，覆盖率 **98.7%**；
443 个样本因覆盖率过低或体量过小而跳过。

---

## 7. 数据规模

| 来源 | 样本数 | 4x 旋转后 |
|---|---|---|
| 旧（v2 小建筑） | 5339 | 21356 |
| 新（大建筑切片） | 6062 有效 | 24248 |
| **合计** | — | **45604** |

> 注：新数据每个样本的 4 个旋转**复用同一张 0° 条件图**（旧数据有真实旋转渲染），
> 存在条件图与体素朝向不匹配的情况，属已知取舍。

---

## 8. 监控与运维

```bash
# 进度
tail -f /data/redstone_v4/work/redo.log
ls /data/redstone_v4/render_new | wc -l          # 渲染进度 /6062
tail -5 /data/redstone_v4/work/render_shard{0,1,2}.log

# 训练
tail -f /home/linux/redstone_ai/logs/trellis_flow_v4.log
tail -1 /data/redstone_v4/out_v4/log.txt         # 当前 step
ls /data/redstone_v4/out_v4/ckpts/               # supervisor 仅保留最后 5 个
```

supervisor 行为：训练进程死亡自动重启（`--auto_retry 3`）、裁剪旧 ckpt（每类保留 5 个）、
到达 200k 后退出。守护期上限 14 天。

环境变量：`ATTN_BACKEND=sdpa`（3090 必须）、`HF_HUB_OFFLINE=1`（离线 DINOv2）。

---

## 9. 已知修复记录

| 问题 | 修复 |
|---|---|
| ingest 轴转置错误 `(2,1,0)` | 改为 `(2,0,1)`，全量重做新数据 |
| 配置 normalize 用旧 VAE 统计量 | 改用合并后统计量 |
| `run_snapshot` 中 `deepcopy(dataset)` 读 43GB mmap → OOM | `mc_flow.py` 改为浅拷贝 |
| `np.flush` 不存在 | 改用 `memmap.flush()` |
| `pack_state_from_props` 传入字符串非 dict | 改用 `parse_props(nm)` |
| litemapy `Region` 轴序错误 | 改为 `Region(0,0,0,D,H,W)` |
| 源文件名 cp866 乱码 | `.encode("cp866").decode("utf-8")` 修复 |
| ⚠️ 内联 `pkill -f` 误杀编排脚本自身 | 改用 PID 精确操作 |
