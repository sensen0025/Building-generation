# v8 — 结论：**仍然未达标**（视觉检查否决）

> 验收顺序（从 v7 起固定）：**先看图 → 图不合格就直接判不合格 → 指标只用来解释原因，不许翻案。**
> 本文件里所有数字都是**诊断**，不是合格依据。

任务目标：从 HY-World 2.0 生成的点云里，输出**一栋完整的建筑外表面**（不是内部房间、不是一面墙）。

本轮做了什么：把 scan-evidence 语料上的结构 Flow **再训 10 小时**（34,000 步，cosine，从 scan_v1 的 33,102 步热启），然后在**同一个点云、同一份证据、同样的步数/种子**上重新部署，和上一版逐图对比。

---

## 一、看了图：和上一版**没有本质区别**

左 = scan_v1（5 h），右 = scan_v2（10 h）——同一输入、同一流程、只是模型换了：

![v1 vs v2](work/v1_vs_v2_side_by_side.png)

四个视角（scan_v2）：

![view000](work/scanv2_view000.png)
![view001](work/scanv2_view001.png)
![view002](work/scanv2_view002.png)
![view003](work/scanv2_view003.png)

图上是什么：**一片绿色地面边缘 + 一整面灰色薄墙 + 墙上一个粉色团（樱花色羊毛）+ 墙顶几根悬空黑色薄板 + 一根细尖塔**。
没有闭合的墙体、没有屋顶、没有"一栋楼"的体量。**和 v1 是同一个东西**，只是略微变厚。

## 二、体量剖面：说明它为什么不是"一栋楼"

![layer profile](work/layer_profile_v1_vs_v2.png)

把 64³ 体素逐层拆开（`y` 为层，`cells` 为该层格子数，`dx/dz` 为水平跨度）：

| y 层 | 是什么 | v1 | v2 |
|---|---|---|---|
| 11–21 | 绿色地面**薄边**（1–2 格厚，填满率 0.06–0.56） | 6,239 格（含全部） | — |
| 22–37 | **一整面墙**：`dz` 一路涨到 59（整个裁剪长度），但 `dx` 只有 6–8 | 每层 162–248 格 | 每层 224–300 格，填满率 +0.1 |
| 38–44 | 墙顶**薄板**：`dx` 突然涨到 30，填满率 0.22–0.65，只有 3–7 层厚 | 1,004–1,058 格/层 | 1,019–1,194 格/层 |
| 45–52 | 细尖塔：`dx=1–2` | 40 格 | 42 格 |

关键一行：**在墙的那 16 层里 `dz=59`（沿观测面方向铺满），`dx` 只有 6–8（垂直于观测面方向不展开）。**
也就是说 Flow 学到的是"**把看到的那面墙沿着墙面自己延伸**"，**不是**"围出一个闭合的建筑轮廓"。

连通性上没有碎块问题：v1 14 个连通块 / 最大块占 99.1%，v2 13 个 / 99.3%（6-连通）。v6 那轮的"消除零星方块"是真的，这一轮也没退化。

## 三、数字（仅作诊断，不作为合格依据）

| 指标 | scan_v1（5 h） | scan_v2（10 h） | 变化 |
|---|---|---|---|
| 生成 / 证据 格数 | 9,484 / 8,656 = **1.096** | 11,517 / 8,656 = **1.331** | +2,033 格（+21%） |
| 导出尺寸 (x,y,z) | 42×31×59 | 43×33×59 | 各 +1 / +2 / 0 |
| 验证集 flow IoU | 0.4466 | 0.4488 | **+0.0022（噪声级）** |
| 最佳 val IoU | 0.5094 | 0.4993 | **−0.0101** |
| 训练 mse（尾段） | — | 0.4797 | 平 |
| 材质状态种类 | 150 | 149 | 平 |

**读法**：多出来的 21% 格子全部落在"同一面墙 + 同一片地面边"上（见上表每层 +40~60 格），
验证集 IoU 只动了 0.002、最佳 val 反而更低。**多训的 5 小时买到的只是"同一堵墙更厚一点"。**

## 四、根因（查代码得出，不是猜）

训练语料的输入证据，定义在 `tools/cloudbuild/build_scan_evidence_dataset.py`：

```python
outside = ndimage.binary_propagation(seed, mask=air)   # 从体积边界能走到的空气
shell   = solid & ndimage.binary_dilation(outside)     # 与外部空气 6-邻接的实心格
visible = ndimage.binary_dilation(shell, iterations=dilate)  # 默认 1
```

`shell` 的定义是"**所有与外部空气相邻的实心格**"——**那就是这栋楼完整的、四面闭合的外表面**。
所以训练时模型面对的永远是：

- **输入 = 完整闭合外表面（四面都在）→ 输出 = 该实心体**

模型因此只学会了 **"把已经闭合的外表面变成实心体"**（shell → solid）。
而部署时输入是**局部观测**：只有相机拍到的一个转角/一面墙（8,656 / 262,144 = 3.3% 覆盖）。
**"从一张开口的局部表面补出一个闭合外表面"这件事，语料从来没要求过它做。**

这解释了本轮观察到的一切，也解释了为什么再加时间没用：
两次训练的**任务定义是同一个**，所以两张图是同一个东西；多出来的格子只是同一面墙变厚。

> 这条是**假设**，但它有明确的、可证伪的检验方式——见下节。**不要在没检验前当结论用。**

## 五、下一步（按性价比排序，都还没做）

1. **做"遮挡证据"语料（针对上面那条根因的最小改动）**
   在 `visible_mask` 之后再加一步：按随机视角方向只保留可见的 shell（或直接随机丢掉 shell 的 60–90%），
   让证据变成**开口的局部表面**，目标仍是同一栋楼的实心体。
   这样训练任务就从 "shell → solid" 变成 "**局部表面 → 闭合外表面 + solid**"，才是部署真正需要的映射。
   检验方式：同一份点云证据重新部署，看图——墙必须开始在 `dx` 方向围合（上表那一行 `dx` 必须涨上去），
   否则假设被证伪。
2. **修点云侧（v7 已定位、未执行）**：v7 的相机 `--move_dist` 用反了（它是"总路程 = 多少倍中位深度"，
   我把它从 8 调到 12 = 让相机飞得更远）。v8 方案：全景相机离建筑 ~12 单位、`--move_dist 1.0`
   （路程 ≈ 0.5 倍建筑距离），需要用户开机服务器才能跑。
3. 材质模型在 scan 证据分布上重训（结构已重训，材质还是 `trellis_material_ctx_v1`）。
4. 采样侧还没扫：`sde_noise`、guidance、threshold。

## 六、交付物（绝对路径）

- 本次模型输出（成品文件，可直接导入游戏）：`/run/media/sensen/Data1/minecraft/AI_3D_World_Repos/outputs/cloudbuild/hyworld2_deploy_scanv2/scene_scanv2.litematic`
  （副本：`/home/sensen/Desktop/hyworld2_exterior_ref/scanv2_deploy/scene_scanv2.litematic`）
- 渲染图 4 视角：`/run/media/sensen/Data1/minecraft/AI_3D_World_Repos/outputs/cloudbuild/hyworld2_deploy_scanv2/render/`
- 对比图：`/home/sensen/Desktop/hyworld2_exterior_ref/scanv2_deploy/v1_vs_v2_side_by_side.png`
- 剖面图：`/home/sensen/Desktop/hyworld2_exterior_ref/scanv2_deploy/layer_profile_v1_vs_v2.png`
- 上一版（v1）渲染：`/run/media/sensen/Data1/minecraft/AI_3D_World_Repos/outputs/cloudbuild/hyworld2_deploy_scanflow/render/`
- 训练记录：`/run/media/sensen/Data1/minecraft/AI_3D_World_Repos/outputs/cloudbuild/trellis_flow_scan_v2/report.json`

渲染用 `tools/schematic_render_local/render.mjs` + 原版资源包，1280×960，覆盖度检查通过（最小 0.0489）。

## 不声称

- 生成场景**没有 ground truth**，这里没有任何精度/质量分数，所有数字都只是形状描述。
- 方块尺度是**选定的非度量映射**（8 格/单位），不是还原出来的物理尺寸。
- 这条链（点云 → 体素 → 结构 Flow → 材质）是**我们自己搭的实验实现**，不是成品。
- 本轮没有 VLM：caption/标签是手写的，已记录在 manifest 里。
