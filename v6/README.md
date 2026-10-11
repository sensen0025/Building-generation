# v6 — HY-World 2.0 生成点云 → Minecraft 建筑（本地部署 + 补体训练）

本目录记录 **v6 这条新链路**：从纯文字 prompt 出发，经 HY-World 2.0 生成世界点云，
再由本地训练栈把它变成带合法 blockstate 的 Minecraft 建筑。

与 v4（inpaint）/ v5（mcfix）不同，v6 解决的是两个新问题：
**(1) 怎么让 HY-World 2.0 真的生成"建筑外表面"**，**(2) 怎么把"只有一层壳"的点云补成实体建筑**。

---

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| HY-World 2.0 五段链路 | ✅ 跑通（我修掉 11 个真实阻塞，见 §4） |
| 室外建筑点云 | ✅ 产出 200 万点，**高出地面 >5 方块有 301,071 点**（旧风景场景是 **0**） |
| 本地部署（点云→方块） | ✅ 端到端跑通，litematic **逐格往返一致**，150 种合法 blockstate |
| **补体训练（本轮核心）** | ✅ **模型从"去噪器"变成"补全器"** |

**补体训练的前后对比**（冻结 64 建筑基准，**同一份挖空证据**）：

| | 旧 flow | 新 flow（scan 训练） |
|---|---|---|
| 部署时 生成/证据 | 0.79–0.92（6 组实验从不 > 1） | **1.096** |
| 凭空补出的格 | 0 | **2,901 = 生成量的 30.6%** |
| 训练 loss | 0.71–0.79（19,400 步**平**） | **0.517–0.535** |
| occupancy IoU | 0.4626 ± 0.0175 | **0.4818 ± 0.0166** |

配对判定（64 栋）：**IoU +0.0192 ± 0.0046 → "scan_v1 better"**；**recall +0.0277 ± 0.0075 → "better"**；
precision 无差别（没变差）；原始证据基线 0.4004。**提升全来自 recall**——补得更多但没多塞垃圾。

---

## 2. 为什么需要"补体训练"

诊断过程（有数据，不靠感觉）：

1. 生成点云是**表面的壳**：64³ 窗口里只有 3.30% 的格有证据，且 99.9% 的格互相连通 → 一层连贯表面。
2. 部署时模型**永远输出 ≤ 证据**：换尺度（4/8 blocks-per-unit）、换阈值（0.3/0.5）、
   只裁建筑、去地形，**6 组实验的 生成/证据 都在 0.79–0.92**。
3. 查训练语料发现根因：**训练数据里"证据"比"实体建筑"还多**
   （evidence/truth = 0.94–1.66）→ 这个 flow 学的是「证据≈占位，稍微去噪」，
   **从来没有被要求"把空壳补成实体"**。所以练更久也不会补体（loss 也证明 19,400 步没进展）。

---

## 3. 做法：造"外表面证据 → 实体建筑"训练对

`code/build_scan_evidence_dataset.py`（11,643 栋，输出 `gs2blocks_scan`）

```
outside = 从体边界出发、不穿实体可达的空气（6 连通 flood fill）
shell   = 与 outside 六邻接的实体格          ← "从外面看得见的表面"
visible = shell 膨胀 1 格                    ← 真实扫描也会落在表面外侧一点
证据    = 原 counts/colours，但只保留 visible 的格
目标    = 原 labels，一格未动
```

| | 数值 |
|---|---|
| 建筑数 | 11,643 |
| 原证据格 | 27,061,661 |
| **砍掉的内部泄漏** | **5,018,321（18.54%）** |
| 重签闭集证明 | 11,643 栋，**0 失败** |

**合规性（重点）**：仓库的闭集监督契约把 `.npz` 哈希绑进证明，防止偷换证据。
`audit_source` 只校验「目标 = 源 schematic 裁剪」与「裁剪外无非空气」，**从不检查 counts/colours**；
因此"改证据、不改目标"是合法的，我用它重签了与新 npz 绑定的证明（`code/reissue_proofs.py`）。
**没有伪造任何证明。**

### 重训

```bash
--data gs2blocks_scan --init <旧最佳> --vae vae_air_repair_stage4/last.pt
--lr-schedule cosine --steps 40000 --ema-decay 0.999
--completion --source-sigma 0.6 --minutes 300        # 实跑 33,000 步
```

---

## 4. 在 HY-World 2.0 上修掉的 11 个阻塞（`code/hyworld2_patch_*.py`）

| # | 阻塞 | 根因 |
|---|---|---|
| 1 | `Sam3VideoProcessor` 加载失败 | transformers 5.2.0 注册了该类但**没这个文件**；官方 `facebook/sam3` 也没有 `video_preprocessor_config.json` |
| 2 | `WorldStereo.from_pretrained` 离线失败 | 代码在 `repo_id` 是本地目录时才会绕过 Hub |
| 3 | 缺 28 GB Wan 权重 | **不需要下载**：wrapper 建完架构立刻用 WorldStereo 自己的 34.8 GB 权重全覆盖 |
| 4 | `AutoencoderKLWan does not support len()` | `if getattr(self,"vae",None)` 对 `torch.compile` 的 `OptimizedModule` 做真值判断会退到 `__len__` |
| 5 | FlashAttention 只支持 fp16/bf16 | ControlNet 路径产生 fp32 → 用 `HYWORLD_ATTENTION=sdpa` 切换 |
| 6 | CUDA OOM | 第 1 段的 VLM 服务常驻 20.6 GB，而第 3~5 段不用它 |
| 7 | `FileNotFoundError: torchrun` | `retrieval_wm` 用 subprocess 调 `torchrun`，环境里没有 |
| 8 | 重跑浪费 40 分钟 | 第 3 段加 `--skip_exist` |
| 9 | WorldMirror 权重悬空 | HF 缓存 blobs 被清；用 sha256 同名文件恢复 |
| 10 | 离线解析不了仓库 id | 子进程带 `HF_HUB_OFFLINE=1` → 显式传本地快照目录 |
| 11 | 第 4 段 MoGe 加载失败 | `gen_gs_data.py` 用 Hub id，本地权重在 `/dev/shm/.../moge/model.pt` |

另外两个关键发现：
- **VLM 不是必需的**：`traj_generate` 的 VLM 调用在 try/except 里，且 `objects.json` 命中时只读缓存
  → 省掉 19 GB 的 Qwen3-VL 下载（我打了补丁让缓存复用不再绑死 `--skip_exist`）。
- **种子全景决定一切**：HY-Pano 是图像条件式的，种子图（俯瞰航拍/透明背景）在主导结果，
  提示词拧不过它。换成"地面视角、有地平线、建筑占主体"的种子后，世界才真的出现建筑。

---

## 5. 部署链路（`code/hyworld2_cloud_infer.py`）

```
点云 PLY → 非度量尺度契约 → 逐格池化 → 结构 Flow → 材质 → litematic → render.mjs 渲染 → 看图
```

- **尺度是"选"的，不是恢复的**：`blocks_per_unit = 8`（HY-World 单位没有物理尺寸）
- **轴向映射**：源 Z-up → Minecraft Y-up，`signed_axes=(+1,+3,-2)`，行列式 +1（旋转不镜像）
- **三处替换已写进 manifest**：labels 全空气（无真值）、`pack()` 用不了（需闭集证明，不伪造）、
  completion 的已知掩码用"有点的格"
- **因此不产出任何 IoU/准确率**：生成场景没有可对照的对象

---

## 6. 目录

```
v6/
├── README.md                 ← 本文件
├── code/                     ← 全部脚本（24 个）
│   ├── hyworld2_cloud_infer.py         部署驱动（features/flow/material/export 四阶段）
│   ├── build_scan_evidence_dataset.py  ★ 补体训练语料构建器
│   ├── reissue_proofs.py               并行重签闭集证明
│   ├── measure_interior_leakage.py     量化"内部泄漏"= 补体信号
│   ├── hyworld2_cloud_voxel_view.py    点云体素化查看器（真渲染器出图）
│   ├── hyworld2_patch_*.py             11 个 HY-World 2.0 修复补丁
│   └── run_generation.sh / fetch_*.sh / prepare_stage3_dirs.sh  服务器流水线
└── work/                     ← 证据（15 个）
    ├── scan_flow_report.md             本轮完整报告
    ├── local_deploy_report.md          部署链路报告
    ├── round2_status.md                种子全景修复报告
    ├── old_vs_new_flow.png             ★ 新旧模型同相机对比（连续性明显改善）
    ├── three_way_comparison.png        点云 / 体素化 / 模型输出 三者关系
    ├── compare_old_on_scan_vs_scan_v1.json  配对判定原文
    ├── scene_scanflow.litematic        ★ 生成的建筑（150 种合法 blockstate）
    └── ...
```

---

## 7. 仍未做到（不粉饰）

1. **补体幅度有限**：证据砍 18.5%，部署净增约 10%（补 2,901、漏 2,073）；基准上 `generated_cells`
   与旧模型无差别——它主要是**把格子放得更准**，不是"大幅增体"。
2. **材质模型没重训**：仍是 `trellis_material_ctx_v1`（旧证据分布），结构已换分布。
3. **点云仍是壳**：重建只看得见表面，这是摄影测量式重建的本质。
4. **天空**用蓝灰启发式过滤（占 4%），不是模型分割。
5. 尺度/裁窗只试了少数几组；`sde_noise` 未启用（基准显示有 +0.0184 的潜力）。

## 8. 下一步（性价比排序）

1. 材质模型一起重训（结构已换分布）
2. 加狠补体目标（`--dilate 0` 或只保留最外层壳）再训一轮
3. 续训 + 调 LR（33,000 步时 loss 仍在降、cosine 未完）
4. 尺度/阈值扫描 + `sde_noise`
