# HY-World 2.0 室外建筑外表面生成 —— 配方与证据（实验性质，非成品）

日期：2026-10-10　机器：region-42 服务器（1×H20 96 GB）

## 1. 你要的"建筑外表面而不是内部房间"怎么控制

**唯一开关：场景目录里的 `meta_info.json`**

```json
{ "scene_type": "outdoor" }
```

- 第 1 段 `traj_generate.py` 会**直接读取已存在的 `meta_info.json`**（约 253 行），不再用 VLM 重新判定场景类型。
- 写 `outdoor` 后走室外分支：`scene_type == "outdoor"`（288/316/855 行）→ 计算天空掩码（ZiM + Grounding-DINO）、走室外轨迹（环绕 + 航拍）、把天空点单独存成 `sky_pcd`。
- **不写就是 indoor**，那就会生成"内部房间"（墙内壁、地板、天花板的房间壳）。

### 硬证据：官方室外参考场景 `braustuble_alley`

路径：`/root/autodl-tmp/hyworld2/official_worldgen/braustuble_alley/`
- `meta_info.json` = `{"scene_type": "outdoor"}`
- `render_results/generation_bank_worldstereo-memory-dmd/aligned_pcd.ply`：**2,000,000 点，最大边长 15.95 m，带颜色**
- `pcd_info.json`：`{"scene_type": "outdoor", "global_pcd": {"num": 1433096}, "aligned_pcd": {"num_voxel": 45547373, "num_final": 2000000}, "sky_pcd": {"num": 360129}}`

三视图（已亲眼看图确认）：`cloud_views.png`
- 俯视：一条斜向延伸的**街巷**
- 正视/侧视：y≈0~2 是**两层建筑立面**，上方是 sky_pcd 天空点，下方地面
- **没有内部房间、没有天花板**

对比 case000（`scene_type: indoor`）最大边长仅 6.243，是"甜甜圈房间壳"。

### 第二个要点：种子全景必须是"站在屋外"的视角

`scene/panorama.png` 是第 1 段的输入。相机在屋外 → 扩展出来才全是外表面。

## 2. 他支持 prompt 输入控制吗？—— 只有全景阶段支持自由文本

| 阶段 | 是否吃自由文本 | 实际控制手段 |
|---|---|---|
| 1 `traj_generate.py` 轨迹规划 | **不支持** | `meta_info.scene_type`(indoor/outdoor) + 几何旋钮：`--rotation_deg 120`、`--rotation_up 45`、`--up_right 60`、`--move_dist 8.0`、`--radius_threshold 4.0`、`--nframe 21`、`--apply_up_route`、`--apply_nav_traj`、`--apply_recon_iteration`、`--force_vlm`。VLM 的提示词（`get_navigation_instruction`）是**写死的导航助手指令**，不是用户输入 |
| 2 `traj_render.py` 多视角渲染 | 无 | 由第 1 段轨迹决定 |
| 3 `video_gen.py` WorldStereo 扩展 | 无 | 用第 1 段 VLM 生成的 caption + 模型 config 里的 `negative_prompt` |
| 4 `gen_gs_data.py` | 无 | 由 3 的输出决定 |
| 5 `world_gs_trainer` 3DGS | 无 | 训练超参 |

**HY-Pano（全景）阶段才有 `--prompt` / `--negative-prompt`**（Qwen-Image-Edit-2509 后端）。

结论：**"方块风格"和"室外视角"都必须在种子全景那一步注入**，后面几段没有文本接口。

## 3. 方块风格：已经验证端到端保留

`scene_panorama.png` = HY-Pano 生成的 **Minecraft 方块风格室外全景**（1920×960，方块草地/树/房子/水塘）。

`expansion_traj2_3frames.png` / `expansion_traj2.mp4` = 第 3 段 WorldStereo 扩展的**真实输出**（21 帧 832×480）。

**看帧确认：方块草地、土块侧面、体素树、青色水面全部保留，Minecraft 美学没有被平滑掉。**
（之前"HY2 会把方块抹平"的结论是错的——那是我用自造 seed 图试出来的假象。）

## 4. 本轮打通的 6 个真实阻塞（都是我手动实施的，属实验性改动）

| # | 阻塞 | 根因 | 修法 |
|---|---|---|---|
| 1 | 第 3 段 `Sam3VideoVideoProcessor` 加载失败 | transformers 5.2.0 在 `auto/video_processing_auto.py` 注册了 `("sam3_video","Sam3VideoVideoProcessor")`，但 `models/sam3_video/` 里**根本没有 `video_processing_sam3_video.py`**；官方 `facebook/sam3` 也没有 `video_preprocessor_config.json` | `hyworld2_sam3_video_shim.py` 实现该类（`BaseVideoProcessor` 子类，逐帧调用真 SAM3 图像处理器）；`hyworld2_patch_sam3_degrade.py` 让去天空失败时降级为 warning 而不是中断 |
| 2 | `WorldStereo.from_pretrained` 离线失败 | HF 缓存快照的 blob 悬空，代码走 `hf_hub_download` 要联网 HEAD | `hyworld2_patch_worldstereo_local.py`：代码里 `repo_id` 是**本地目录**时完全绕过 Hub |
| 3 | 缺 Wan2.1 transformer 权重（28 GB） | 本地快照 `transformer/` 只有 config.json | `hyworld2_setup_wanlocal.sh`：**不需要下载**——wrapper 建完架构后立刻用 WorldStereo 自己的 34.8 GB 权重全覆盖（`load_state_dict(strict=False)`，1799 个无前缀张量）。把 `transformer/diffusion_pytorch_model.safetensors` 指向该文件即可 |
| 4 | `TypeError: AutoencoderKLWan does not support len()` | `_pipeline_common.py:82-83` 写 `if getattr(self,"vae",None)`，而 VAE 被 `torch.compile` 包成 `OptimizedModule`，Python 真值判断会退到 `__len__`，dynamo 的 `__len__` 抛错 | `hyworld2_patch_pipeline_truthiness.py` 改成 `is not None` |
| 5 | `RuntimeError: FlashAttention only support fp16 and bf16` | `models/controlnet.py` 路径产生 fp32 张量 | 加环境变量 **`HYWORLD_ATTENTION=sdpa`** 切到 PyTorch SDPA（CUDA 上内部仍是 flash 内核），零代码改动 |
| 6 | `torch.OutOfMemoryError` | 第 1 段的 `vlm_service.py` 常驻占 20.6 GB 显存，而第 3~5 段不用它 | `hyworld2_patch_runner_vlm.py`：SKIP_STAGES 含 1 时不再启动 VLM；另加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

脚本归档：`/run/media/sensen/Data1/minecraft/AI_3D_World_Repos/tools/cloudbuild/hyworld2_*.py|sh`

## 5. 复现命令

```bash
# 第 1-2 段（全景 + 轨迹渲染），需 VLM 服务
SCENE=/root/autodl-tmp/hyworld2/town_compare/scene bash /root/run_generation.sh

# 第 3-5 段（世界扩展 + 3DGS），不需要 VLM
cd /root
SKIP_STAGES=12 SCENE=/root/autodl-tmp/hyworld2/town_compare/scene \
  HYWORLD_ATTENTION=sdpa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup bash /root/run_generation.sh </dev/null >/root/gen.log 2>&1 &
```

注意 `/dev/shm` 是内存盘，容器重启后需重跑 `hyworld2_setup_wanlocal.sh`。

## 6. 沙盒事实

- 无训练：全部是**推理/生成**，沙盒里没有真实外部环境。
- 本轮所有改动都是**我自己手动打补丁**得到的，属**实验性**，不是官方支持路径。
- 产物是**生成结果**，未经真实世界验证；下游喂给方块模型的精度仍待评测。
