
import numpy as np
BASE = '/home/linux/redstone_ai/dataset'
# ss64 127 位置
ss = np.load(f'{BASE}/v2/ss64.npy', mmap_mode='r')
vid = np.load(f'{BASE}/voxel96_id.dat', mmap_mode='r')
N, C = ss.shape[0], 64
OFF = 16
vocab = [l.strip() for l in open(f'{BASE}/BLOCK_VOCAB_FULL.txt', encoding='utf-8') if l.strip()]
print('vocab lines:', len(vocab), flush=True)
# 统计 ss64==127 的 cell 在 voxel96 的 old id
old_cnt = {}
n127 = 0
step = max(1, N // 8)
for i in range(N):
    v = np.asarray(ss[i])
    m = v == 127
    if not m.any():
        continue
    seg = np.asarray(vid[i][OFF:OFF+C, OFF:OFF+C, OFF:OFF+C])
    ids = seg[m]
    n127 += len(ids)
    ui, inv = np.unique(ids, return_inverse=True)
    for u, c in zip(ui, np.bincount(inv, minlength=len(ui))):
        old_cnt[int(u)] = old_cnt.get(int(u), 0) + int(c)
    if (i+1) % step == 0:
        print(f'  {i+1}/{N} n127={n127}', flush=True)
print('n127 total:', n127, flush=True)
print('--- old-id 构成 (top 40) ---', flush=True)
for oi, c in sorted(old_cnt.items(), key=lambda kv:-kv[1])[:40]:
    nm = vocab[oi] if 0 <= oi < len(vocab) else f'?({oi})'
    print(f'  old {oi}: {nm}  {c:,}  ({100.0*c/n127:.2f}%)', flush=True)
named = sum(c for oi, c in old_cnt.items() if 0 <= oi < len(vocab) and oi != 201)
print(f'named(可还原): {named:,} ({100.0*named/max(n127,1):.2f}%)', flush=True)
