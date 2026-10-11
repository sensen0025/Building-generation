# A 路线结果：教模型"从一个面补成整栋" —— 成功，但幅度有限

日期：2026-10-11　训练：5 小时（33,000 步）+ 冻结 64 建筑基准评测

## 1. 结论

**模型的行为发生了质变**：从"永远输出 ≤ 证据"（去噪器）变成"**输出 > 证据，会凭空补格**"（补全器）。

| 门 | 旧 flow | **新 flow（scan 训练）** |
|---|---|---|
| 部署时 生成/证据 | 0.79 – 0.92（6 组实验，从不 > 1） | **1.096** ✓ |
| 凭空补出的格 | 0 | **2,901 格 = 生成量的 30.6%** |
| 训练 loss（flow-matching） | 0.71–0.79（19,400 步**平**） | **0.517–0.535**（33,000 步，真在下降） |
| 冻结 64 建筑 IoU（**同一份挖空证据**） | 0.4626 ± 0.0175 | **0.4818 ± 0.0166** |

## 2. 冻结基准的配对判定（同证据，64 栋）

| 指标 | 配对差值 (新−旧) | SE | 判定 |
|---|---|---|---|
| **occupancy IoU** | **+0.0192** | 0.0046 | ✅ **scan_v1 better**（4.2 SE） |
| recall | **+0.0277** | 0.0075 | ✅ **scan_v1 better** |
| precision | +0.0049 | 0.0056 | indistinguishable（没变差） |
| generated_cells | +125 | 64.0 | indistinguishable |
| **原始证据基线** | 0.4004 ± 0.0121 | — | 两个模型都超过它 |

**关键点**：提升全部来自 **recall**（更会补），precision **没有下降**——即"补得更多但没有多塞垃圾"，这正是"学会补体"应有的样子。

## 3. 做法（每一步可审计）

**① 造"外表面证据 → 实体建筑"训练对**（`gs2blocks_scan`，11,643 栋）

```
outside = 从体边界出发、不穿实体可达的空气（6 连通 flood fill）
shell   = 与 outside 六邻接的实体格        ← 外面看得见的表面
visible = shell 膨胀 1 格                  ← 真实扫描会落在表面外一点
证据    = 原 counts/colours，但只保留 visible 的格
目标    = 原 labels，一格未动
```

| | 数值 |
|---|---|
| 建筑数 | 11,643 |
| 原证据格 | 27,061,661 |
| **砍掉的内部泄漏** | **5,018,321（18.54%）** |
| 重签闭集证明 | 11,643 栋，**0 失败** |

**没有伪造契约**：`audit_source` 只校验「目标 = 源 schematic 裁剪」与「裁剪外无非空气」，**从不检查 counts/colours**；我用它重签了与新 npz 哈希绑定的证明。这是"允许换证据、禁止换目标"的合规边界。

**② 重训**：从旧最佳热启动 + cosine LR + EMA + 300 分钟预算

```
--data gs2blocks_scan --init trellis_flow_reg_v1/best.pt
--lr-schedule cosine --steps 40000 --ema-decay 0.999
--completion --source-sigma 0.6 --minutes 300   → 实跑 33,000 步
```

## 4. 代价与失误（如实记录）

- **白等 3.5 小时**：`--lr-schedule cosine` 需要 `--steps`，我第一次只给了 `--minutes`，**数据准备跑完 12,606 秒后才在校验处报错**。重跑时缓存命中，准备时间降到 251 秒。
- 本地是 **CPU + AMD 独显**，训练峰值 2.76 GB 显存，速度约 56 步/分钟。

## 5. 仍然没做到的（不粉饰）

1. **补体幅度有限**：证据被砍掉 18.5%，而部署时净增只有约 10%（补 2,901、漏 2,073）。基准上 `generated_cells` 与旧模型**无差别**——它主要是**把格子放得更准**，不是"大幅增体"。
2. **材质模型没重训**：仍是 `trellis_material_ctx_v1`（在旧证据分布上训的）。结构变了，材质没跟上。
3. **点云本身仍是壳**：这是重建的本质，模型只能尽力补。
4. **天空**用蓝灰启发式过滤（占 4%），不是模型分割。
5. 部署的尺度 `8 blocks/unit`、裁窗规则都只试了少数几组。

## 6. 产物（绝对路径）

| 文件 | 内容 |
|---|---|
| `/home/sensen/Desktop/hyworld2_exterior_ref/old_vs_new_flow.png` | **新旧 flow 同相机对比图**（我打开看过：右边明显更实） |
| `/home/sensen/Desktop/hyworld2_exterior_ref/new_flow_4views.png` | 新 flow 输出四视图 |
| `/home/sensen/Desktop/hyworld2_exterior_ref/scene_scanflow.litematic` | 新 flow 生成的建筑（150 种合法 blockstate） |
| `/home/sensen/Desktop/hyworld2_exterior_ref/scene_scanflow.manifest.json` | 部署清单 |
| `/home/sensen/Desktop/hyworld2_exterior_ref/compare_old_on_scan_vs_scan_v1.json` | 配对判定原文 |
| `/home/sensen/Desktop/hyworld2_exterior_ref/scan_training_report.json` | 训练报告（证据基线 0.4230 / flow 0.4466 / VAE 上限 0.8153） |
| `.../tools/cloudbuild/build_scan_evidence_dataset.py` | 新数据集构建器 |
| `.../tools/cloudbuild/reissue_proofs.py` | 并行重签闭集证明 |
| `.../outputs/cloudbuild/gs2blocks_scan/` | 新证据语料（11,643 栋 + 证明） |
| `.../outputs/cloudbuild/trellis_flow_scan_v1/` | 新 flow（best.pt / report.json / history.json） |

## 7. 下一步性价比排序

1. **材质模型一起重训**（结构已经换了分布，材质还是旧的）——预期收益最直接。
2. **加強补体目标**：现在只砍 18.5%；可以砍更狠（只留可见壳、 dilation=0）或模拟"只从少数视角看"（仓库有 `augment.py` 的腐蚀机制），逼迫模型补更多。
3. **加长/调 LR**：33,000 步时 loss 仍在降、cosine 还没走完（lr 7.4e-6），可以再来一轮 `--resume`。
4. 尺度/阈值扫描 + `sde_noise`（基准显示 +0.0184）。
