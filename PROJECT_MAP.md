# PROJECT_MAP — redstone_ai 项目地图（历史/基线资产）

> 服务器 `175.155.64.171:22173` · 更新 2026-09-11
> 本文档描述 **历史与基线资产**（v1/v2 数据集、模型 checkpoint）。
> v4 inpaint 全流程见 `/data/redstone_v4/ARCHITECTURE.md`。

---

## 1. 项目概览

Minecraft 3D 体素生成模型：TRELLIS **稀疏结构 Flow Matching** + **DINOv2-large** 图像条件，
128 类方块词表 + 5 字段打包状态。

硬件：RTX 3090 24GB / 12 core / 31GB RAM；`/` 197G，`/data` 197G（独立盘）。

---

## 2. 目录结构与职责

| 路径 | 大小 | 职责 | 状态 |
|---|---|---|---|
| `TRELLIS/` | 8.6G | 模型代码（`trellis` 包）+ `outputs/` checkpoint | ✅ 核心 |
| `TRELLIS/outputs/` | 8.5G | 各代模型 checkpoint | ✅ 保留 |
| `dataset/v2/` | 19G | **v2 规范训练数据**（4x 旋转增强） | ✅ 规范 |
| `dataset/litematics/` | 102M | 源 litematic | 📦 输入 |
| `dataset/render_schem/` | 1.2G | 渲染用 schematic | 📦 输入 |
| `render_output_rot/` | 3.6G | **v2 条件图（4x 旋转，21296 张）** | ✅ 规范 |
| `render_mc/` | 4.9G | Resource-Hub 构建数据（`builds.rhb` 等） | 📦 保留 |
| `logs/` | 724K | 训练日志 | 📋 |
| `_tool__rh_*.py`（40+） | — | 一次性探查/调试脚本 | 🗄 可归档 |
| `gen_rot_v3.py` `_gen_v3*.py` | — | v3 生成脚本 | 🗄 可归档 |

---

## 3. 关键数据（**不可删**）

### 3.1 v2 规范训练数据 `dataset/v2/`

| 文件 | 形状 | 说明 |
|---|---|---|
| `ss64_rot.npy` | (21356, 64,64,64) uint8 | 体素方块 ID（128 类），4x 旋转 |
| `ss64_state_rot.npy` | (21356, 64,64,64) uint16 | 打包状态（5 字段） |
| `ss_latent_mc64_v2_rot.npy` | (21356, 8,16,16,16) f32 | 冻结 VAE latent |
| `latent_norm_rot.npz` | mean/std ×8 | latent 归一化统计 |

打包状态字段：乘数 `[1, 7, 21, 126, 2016]`，维度 `[7, 3, 6, 16, 3]`。

### 3.2 关键 checkpoint

| 路径 | 说明 |
|---|---|
| `TRELLIS/outputs/ss_vae_mc_v2/ckpts/encoder_step0050000.pt` | **冻结 VAE 编码器** — 定义 128 类语义，**不可替换**（替换需重训 ~4.3h） |
| `.../decoder_step0050000.pt` | 冻结 VAE 解码器 |
| `TRELLIS/outputs/ss_flow_mc_rot_400k/ckpts/denoiser_ema0.9999_step0400000.pt` | **v4 微调基点**（400k steps EMA） |

> 128 类词表**不能扩展**而不重训 VAE；v4 新数据改用 **alias 映射**（覆盖率 98.7%）。

---

## 4. 数据血缘

```
litematic / .schem 源文件
  │
  ├─ v1: voxel96 (96³) ......................... → 已归档 /data/archive_v3/datasets_v1/
  │
  └─ v2: 64³ 体素 + 5 字段打包状态, 4x 旋转 ..... → dataset/v2/*_rot.npy      【规范】
        └─ 冻结 VAE 编码 → latent ............... → ss_latent_mc64_v2_rot.npy
              └─ 训练 ss_flow_mc_rot_400k (400k)  → outputs/ss_flow_mc_rot_400k/
                    └─ v4: + 6062 大型建筑切片 .... → /data/redstone_v4/     【进行中】
```

---

## 5. 轴约定（**重要教训**）

- VAE `decode.py` 输出体积索引为 `[y,z,x]`，dims 为 `(W,H,L)`
- **下游训练数据必须是 `[x,y,z]`，y 为垂直轴** → 转置须为 `transpose(2,0,1)`
- 曾误写 `transpose(2,1,0)` → 得到 `[x,z,y]`，高度与长度互换，建筑"躺倒"（相机拍到顶部/底部）
- `litemapy.Region(width, height, length)` 索引顺序即 `(x, y, z)`
- `rot_y_64(a) = np.transpose(a[:, :, ::-1], (2,1,0))` 绕 y 轴旋转（作用于 `[x,y,z]`）

---

## 6. 归档区

已移入 `/data/archive_v3/`（**可恢复**，非删除），详见该目录 `RESTORE.md`：

| 归档子目录 | 内容 |
|---|---|
| `datasets_v1/` | `voxel96_id.dat` + `voxel96_state.dat`（13.2G） |
| `v2_preaug/` | 未旋转版 `ss64.npy` / `ss64_state.npy` / `ss_latent_mc64_v2.npy` |
| `oldvae/` | 冻结 VAE 之前的 `*_oldvae.npy` / `latent_norm_oldvae.npz` |
| `checkpoints/` | `ss_flow_mc_img_v3.old68k/` |
| `renders/` | `render_output/`（rot 集 0° 子集）、`render_output_placeholder_bak/` |
| `v4_intermediate/` | ingest/compact 中间数组（已合并进 `train/`） |
| `logs/` | 历史日志（含 transpose bug 时期的记录） |

---

## 7. 运维要点

- 远端操作走本地 `paramiko` helper：`_ssh_run.py`（`cmd` / `put` / `get`）
- Python 环境：`/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python`
- Node：`/home/linux/node-v22.14.0-linux-x64/bin/node`
- 训练环境变量：`ATTN_BACKEND=sdpa HF_HUB_OFFLINE=1`
- ⚠️ 避免内联 `pkill -f`：会匹配到命令自身包装进程，曾误杀编排脚本；请按 PID 操作
