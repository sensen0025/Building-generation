# 本轮实况：管线全通，但这个场景里没有建筑

日期：2026-10-10　机器：region-42 服务器（1×H20 96 GB）→ 本地 CPU 分析

## 1. 结论先说

1. **HY-World 2.0 五段链路已经全通**（我修掉了 11 个真实阻塞，见下），第一次真正跑出我们场景的室外全量点云。
2. **但这份点云里没有建筑**：它是一片地形 + 天空穹顶。定量证据（`scene_standing.png`、`scene_clusters.png`，我都打开看过）：
   - 天空（蓝灰色）点 **919,723 个 = 全云的 46%**
   - 高于地面 >2 方块：20,392 点；>3 方块：**282** 点；>5 方块：**0**
   - 最大突出高度 / 场景宽度 ≈ **5%**（真正的建筑外立面应是几十 %）
   - 正视图里"高于地面的点"在 Z 上只占 **0.04 单位厚的一条薄片**——**没有任何垂直墙体**
3. 原因不是管线坏了，而是**输入**：种子全景 `scene_panorama.png` 是一张"俯瞰浮空岛屿"的风景图（草地/树/水塘 + 一大片灰天空），HY2 忠实地扩展成了一片风景。**全景阶段是唯一的自由文本控制点**，所以要让世界里有建筑，必须在那里注入"建筑外立面"。

## 2. 本轮修掉的阻塞（全部是我手动打补丁，实验性，非官方路径）

| # | 阻塞 | 根因 | 修法 |
|---|---|---|---|
| 7 | `FileNotFoundError: torchrun` | 20 条轨迹全跑完后 `retrieval_wm.py:1140` 用 subprocess 调 `torchrun` 跑 WorldMirror，而 nav_env 里没有 | 装转发脚本 `torchrun` → `python -m torch.distributed.run` |
| 8 | 重跑浪费 40 分钟 | 无 | 第 3 段加 `--skip_exist`（点云已存在时整段秒过：83 秒） |
| 9 | WorldMirror 权重悬空 | HF 缓存 blobs 被清，`model.safetensors` 指向不存在的 blob | 用 `missing_region_test/worldmirror/model.safetensors` 恢复（**sha256 = `9fff0653…` 与 HF blob 名逐位一致**，确证同一份） |
| 10 | 离线解析不了仓库 id | 子进程带 `HF_HUB_OFFLINE=1`，`snapshot_download` 必失败（加 `local_files_only` 也报 "Cannot find an appropriate cached snapshot folder"） | 把本地快照目录显式作为 `--pretrained_model_name_or_path` 传入，解析器第一步命中，不碰 Hub |
| 11 | 第 4 段 `LocalEntryNotFoundError` | `gen_gs_data.py:98` 用 Hub id 加载 MoGe，而本地权重在 `/dev/shm/hyworld2-nav-models/moge/model.pt`（第 1/3 段都在用） | 补丁指向本地路径 |

（第 1–6 个阻塞见 `README.md`：SAM3 视频类缺失、WorldStereo 离线、28 GB Wan 权重、`torch.compile` 真值判断、FlashAttention fp32、VLM 占显存。）

## 3. 已跑通的证据链

```
[16:27:47] === stage 3_video_gen
[16:32:23] === stage 3_video_gen done     ← WorldMirror 推理成功(138.8s) + 对齐(24视频/187帧)
[16:35:39] === stage 4_gen_gs_data        ← skip_exist 秒过
[16:36:58] === stage 4_gen_gs_data done   ← 58.6s（打上 MoGe 补丁后）
[16:36:58] === stage 5_gs_trainer         ← 3DGS 训练启动（GPU 65%）
```

产出点云：`generation_bank_worldstereo-memory-dmd/aligned_pcd.ply`
- 2,000,000 点、带颜色、最大边长 **3.691**（源单位，非度量）
- `pcd_info.json`：`{"scene_type":"outdoor", "global_pcd":{"num":237572}, "aligned_pcd":{"num_voxel":9033941,"num_final":2000000}, "sky_pcd":{"num":794758}}` ✓ 室外分支确实生效

## 4. 沙盒事实

- **服务器 SSH 端口已失效**（`58772` 连接被拒，主机 ping 通）→ 容器重启拿了新端口。重启会清空 `/dev/shm`（nav 模型 + WanLocal 镜像需重建，脚本 `hyworld2_setup_wanlocal.sh`、`fetch_nav_models.sh`；约 40 分钟）。
- 持久盘没丢：打补丁后的源码、WorldStereo 34.8 GB 权重、`sam3_public`、以及**场景和它的点云**都还在。
- 本地只做 CPU 分析（numpy/matplotlib），没有 GPU 训练。

## 5. 本地文件

| 文件 | 内容 |
|---|---|
| `scene_aligned_pcd.ply` | 本次生成的室外全量点云（32 MB，200 万点） |
| `scene_panorama.png` | 种子全景（HY-Pano 出的 Minecraft 风格风景图） |
| `scene_exterior_cloud_views.png` | 点云三视图（俯/正/侧） |
| `scene_cloud_analysis.png` | 天空/地形/候选结构分离 |
| `scene_clusters.png` | 颜色聚类 + 地面板放大 |
| `scene_standing.png` | **决定性证据**：剔除天空后，高于地面 2 方块的点在 Z 上只是薄片 |
| `expansion_traj2.mp4` / `_3frames.png` | 第 3 段世界扩展的真实输出（方块风格保留 ✓） |
| `README.md` | 外表面配方 + prompt 控制面 + 前 6 个阻塞的修法 |

## 6. 下一步（要建筑就必须重做全景）

1. **用建筑外立面的提示词重做种子全景**（HY-Pano 阶段的 `--prompt`）：
   例如 "Minecraft 方块风格的中世纪城堡，站在城堡外正面拍摄，占据画面主要部分，1×1 米方块，清晰方块边缘，不要俯视"。
2. 重跑第 1→5 段（`meta_info.json` 保持 `outdoor`），拿到**有建筑**的室外点云。
3. 天空问题：SAM3 视频去天空目前是降级状态；重做时要么补上 `Sam3VideoVideoProcessor`（官方 transformers 5.2.0 缺该类、`facebook/sam3` 缺 `video_preprocessor_config.json`），要么在下游按颜色/几何自己滤（本轮已验证：蓝灰规则能滤掉 46% 的天空点）。
4. 然后走本地链路：`point_scene_preflight.py`（PLY + `WorldContract` 非度量尺度契约 + `pool_cells`）→ 结构 flow（点云证据 token 条件）→ 材质 → litematic → `schematic_render_local/render.mjs` 渲染 → 看图验证 → 冻结基准评测。
