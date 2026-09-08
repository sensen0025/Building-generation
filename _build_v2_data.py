"""构建 v2 数据集: 96^3 -> 64^3 中心裁剪(1:1, 不降采样), 类映射优先保留有state的方块, 对齐保存 state.
输出:
  dataset/v2/ss64.npy          (N,64,64,64) uint8  新类 id
  dataset/v2/ss64_state.npy    (N,64,64,64) uint16 5字段state(原编码)
  dataset/v2/block_map.npy     (N_CLASSES,) int32  新id -> 旧vocab(202)id
  dataset/v2/build_report.txt  报告
"""
import numpy as np
import os, time

BASE = '/home/linux/redstone_ai/dataset'
OUT = f'{BASE}/v2'
os.makedirs(OUT, exist_ok=True)
G = 96
C = 64
OFF = (G - C) // 2  # 16
N_CLASSES = 128  # 0=air + 126 方块 + 127=other

meta = np.load(f'{BASE}/voxel96_meta.npz')
N = int(meta['n'])
print(f'N={N}, 裁剪 {G}^3 -> {C}^3 offset={OFF}', flush=True)

flat_id = np.load(f'{BASE}/voxel96_id.dat', mmap_mode='r')    # (N,96,96,96) npy 格式
flat_st = np.load(f'{BASE}/voxel96_state.dat', mmap_mode='r')  # (N,96,96,96) npy 格式

# ---- pass 1: 频率统计 (裁剪后) ----
id_counts = np.zeros(203, dtype=np.int64)
id_stated = np.zeros(203, dtype=np.int64)
total = 0
stated = 0
for i in range(N):
    v = flat_id[i][OFF:OFF + C, OFF:OFF + C, OFF:OFF + C]
    s = flat_st[i][OFF:OFF + C, OFF:OFF + C, OFF:OFF + C]
    nz = v > 0
    total += int(nz.sum())
    ui, inv = np.unique(v[nz], return_inverse=True)
    id_counts[ui] += np.bincount(inv, minlength=len(ui))
    stnz = s[nz] != 0
    stated += int(stnz.sum())
    if stnz.any():
        id_stated[ui] += np.bincount(inv[stnz], minlength=len(ui))
print(f'pass1 done: solid={total:,} stated={stated:,} ({100.0*stated/total:.2f}%)', flush=True)

# ---- 类选择: 有state的类优先, 其余按频率 ----
stated_ids = [c for c in range(1, 201) if id_stated[c] > 0 and id_counts[c] >= 500]  # 排除201(other)
plain_ids = [c for c in range(1, 202) if c not in stated_ids and id_counts[c] > 0]
plain_ids.sort(key=lambda c: -id_counts[c])
print(f'有state的类(独立保留): {len(stated_ids)}: {stated_ids}', flush=True)

new_ids = np.zeros(N_CLASSES, dtype=np.int32)
new_ids[0] = 0  # air
n_slots = N_CLASSES - 1  # 方块类槽位 1..N_CLASSES-2, 最后一个是 other
pick = (stated_ids + plain_ids)[:n_slots]
for ni, oi in enumerate(pick, start=1):
    new_ids[ni] = oi
new_ids[N_CLASSES - 1] = 201  # other
old2new = np.full(203, N_CLASSES - 1, dtype=np.int32)  # 默认 other
for ni, oi in enumerate(new_ids):
    old2new[oi] = ni
old2new[0] = 0  # air
print(f'N_CLASSES={N_CLASSES}, 独立类数={len(np.unique(new_ids))-1} (含other)', flush=True)

vocab = [l.strip() for l in open(f'{BASE}/BLOCK_VOCAB_FULL.txt', encoding='utf-8') if l.strip()]
top200 = meta['top200'].astype(int)
id2name = ['minecraft:air'] * 203
for cls in range(1, 201):
    o = int(top200[cls - 1])
    id2name[cls] = vocab[o] if 0 <= o < len(vocab) else '?'
id2name[201] = 'minecraft:other'

# ---- pass 2: 构建输出 ----
ss = np.lib.format.open_memmap(f'{OUT}/ss64.npy', mode='w+', dtype=np.uint8, shape=(N, C, C, C))
ss_st = np.lib.format.open_memmap(f'{OUT}/ss64_state.npy', mode='w+', dtype=np.uint16, shape=(N, C, C, C))
t0 = time.time()
for i in range(N):
    v = flat_id[i][OFF:OFF + C, OFF:OFF + C, OFF:OFF + C]
    s = flat_st[i][OFF:OFF + C, OFF:OFF + C, OFF:OFF + C]
    ss[i] = old2new[v].astype(np.uint8)
    ss_st[i] = s
    if (i + 1) % 2000 == 0:
        print(f'  {i+1}/{N} ({time.time()-t0:.0f}s)', flush=True)
ss.flush()
ss_st.flush()
np.save(f'{OUT}/block_map.npy', new_ids)

# ---- 报告 ----
rep = [f'N={N} crop={C}^3 offset={OFF} N_CLASSES={N_CLASSES}',
       f'solid_cells={total:,} stated={stated:,} ({100.0*stated/total:.2f}%)',
       f'独立类(含other): {len(np.unique(new_ids))}']
rep.append('--- 新类映射 (new_id: old_id: name: cells: stated%) ---')
sel_counts = id_counts[new_ids]
sel_stated = id_stated[new_ids]
for ni in range(1, N_CLASSES):
    oi = int(new_ids[ni])
    rep.append(f'{ni:3d}: {oi:3d}: {id2name[oi]:<32} cells={sel_counts[ni]:>10,} stated={100.0*sel_stated[ni]/max(sel_counts[ni],1):5.1f}%')
# 保留率: 各类在裁剪后总cells的占比 vs 未映射 other
other_cells = int(id_counts[201])
rep.append(f'--- other(未选类聚合) cells={other_cells:,} ({100.0*other_cells/total:.2f}%) ---')
with open(f'{OUT}/build_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(rep))

# ---- class_weights (128类, 逆频率, 平滑) ----
counts = np.zeros(N_CLASSES, dtype=np.float64)
cnt_full = np.bincount(ss[:].reshape(-1), minlength=N_CLASSES)[:N_CLASSES].astype(np.float64)
counts = cnt_full
cw = np.zeros(N_CLASSES, dtype=np.float32)
nonz = counts > 0
med = np.median(counts[nonz]) if nonz.any() else 1.0
for c in range(N_CLASSES):
    if counts[c] > 0:
        cw[c] = np.clip(med / (counts[c] + 1.0), 0.01, 10.0)
    else:
        cw[c] = 1.0
cw[0] = 0.02  # air 权重极小 (避免全预测空气)
np.save(f'{OUT}/class_weights.npy', cw)
print(f'class_weights saved ({N_CLASSES})', flush=True)

print('\n'.join(rep[:20]))
print(f'DONE -> {OUT} ({time.time()-t0:.0f}s)', flush=True)
