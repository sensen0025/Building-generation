# 本地部署：HY-World 2.0 点云 → Minecraft 方块（实验结果，非成品）

日期：2026-10-11　输入：`new2_aligned_pcd.ply`（新全景生成的室外点云，200 万点，含建筑）

## 1. 这一轮真正跑通了什么

**整条产品链在本地跑通了**：生成的点云 → 非度量尺度契约 → 逐格池化 → 结构化 flow（训练好的最佳权重）→ 材质（147 种合法 blockstate）→ **litematic 导出（往返逐格一致）** → 真渲染器出图。

```
[features] observed cells 8656  窗口 64³  counts_max 2359  origin [-27,-6,-39]
[flow]     condition built (5×16³) with pooled_condition
[flow]     generated cells 7846  (evidence 8656)       ← 25 步，与训练时一致
[material] non-air cells 7846  distinct states 147
             254298  minecraft:air
               2500  minecraft:grass_block[snowy=false]
                847  minecraft:lime_wool
                443  minecraft:stone
                415  minecraft:stone_bricks
[export]   hyworld2_scene.litematic  shape (42,47,59)  palette 147  roundtrip exact
[render]   4 视图，覆盖率 0.078–0.111
```

## 2. 用的权重（按冻结基准 BENCHMARK_V1.md 选）

| 阶段 | 权重 | 依据 |
|---|---|---|
| VAE | `vae_air_repair_stage4/last.pt` | flow checkpoint 的 `args.vae` 字段（必须是配套的那一个） |
| 结构 flow | `trellis_flow_reg_v1/best.pt` | 64 建筑冻结基准里最好的结构结果 0.4614 IoU（> 原始证据 0.4508） |
| 材质 | `trellis_material_ctx_v1/best.pt`（step 112000，自带邻域 trunk） | 邻域 trunk 是唯一被配对比较判定为真实提升的材质改动（+0.0163 IoU） |

采样参数照训练时：`steps=25, guidance=1.0, source_sigma=0.6, completion=True, seed=0, threshold=0.5`。

## 3. 关键契约（都写进了 manifest）

- **尺度是"选"的，不是恢复的**：HY-World 单位没有物理尺寸。本次 `blocks_per_unit = 8`（场景最大边长 11.9 单位 → 约 95 格）。
- **轴向映射**：源是 Z-up，目标是 Minecraft 的 Y-up XYZ。`signed_axes = (+1, +3, -2)`（X=+x, Y=+z, Z=−y），行列式 = +1 → **是旋转不是镜像**（保手性）。
- **裁窗**：在水平面上滑动 64³ 窗口，取"高出地面"点数最多的那个（建筑优先）。
- **天空**：蓝灰启发式规则剔除（该场景占 4.4%）。

## 4. 三个必须说清的替换（因为没有真值）

1. `labels` 全为空气——生成的场景没有参考建筑，**没有任何 GT**。
2. `pack()` **用不了**：它内部要求"闭集监督证明"（`require_closed_asset`），我没有伪造它；条件改用 `pooled_condition()`，这与 `pack()` 给模型的 token 是同一套池化（仓库测试逐格验证一致）。
3. completion 用的"已知证据掩码"是**有点的格子**（`count>0`），而不是 GT 的已知空气图。

**因此本轮不产出任何 IoU / 准确率数字**——没有可对照的对象。manifest 里 `not_claimed` 明确列了这条。

## 5. 看图得到的诚实结论

四视图（`deployed_building_4views.png`，我打开看过）：**是方块风格的 Minecraft 结构**——绿色草地基底 + 灰色石砖墙体竖立其上，还夹杂些红/黄/棕方块。

但**不是一栋漂亮建筑**：
- 生成格 7,846 ≈ 证据格 8,656，说明 flow 基本是**在复现点云的表面格子**，没有把它"实心化"成一栋楼。
- 材质里 grass 占了 2,500 格（把大片地面读成草），stone/stone_bricks 才 858 格。
- 结构很薄（7,846 / 42×47×59 = 6.8% 填充率）。

原因（诚实归因，不推给运气）：
- 输入是**表面点云**，只有壳没有体，模型没理由凭空补出实心体；
- 材质模型是在**数据集建筑**（WorldMirror 重建）上训练的，HY-World 点云是**域外**输入；
- 裁窗与尺度我只试了 `8 blocks/unit` 一组，没有调优。

## 6. 本地产物（绝对路径）

| 文件 | 内容 |
|---|---|
| `/home/sensen/Desktop/hyworld2_exterior_ref/new2_aligned_pcd.ply` | 输入的室外点云（200 万点） |
| `/home/sensen/Desktop/hyworld2_exterior_ref/hyworld2_scene.litematic` | **生成的 Minecraft 建筑**（42×47×59，147 种合法 blockstate） |
| `/home/sensen/Desktop/hyworld2_exterior_ref/deployed_building_4views.png` | 真渲染器四视图 |
| `/home/sensen/Desktop/hyworld2_exterior_ref/hyworld2_scene.manifest.json` | 完整清单（契约、替换、权重、不声称什么） |
| `/run/media/sensen/Data1/minecraft/AI_3D_World_Repos/tools/cloudbuild/hyworld2_cloud_infer.py` | 新写的部署驱动（四阶段） |
| `.../outputs/cloudbuild/hyworld2_deploy_v1/` | 全部中间产物（npz / flow.npz / states.npy / render/） |

## 7. 下一步能改进的点（按性价比排序）

1. **裁窗与尺度扫描**：`blocks_per_unit` 试 4/6/8/12，并比较不同窗口——现在只有一组。
2. **提高阈值敏感度**：threshold 0.3/0.4/0.5 扫一遍，看能否得到更实心的体量。
3. **域内微调**：材质模型在 HY-World 点云上没有训练过；用几份 HY-World 云做少量适配可能明显改善材质。
4. **多 seed / SDE 噪声**：基准显示 `sde_noise=1.0` 有 +0.0184 IoU 的提升，可作为采样选项。
