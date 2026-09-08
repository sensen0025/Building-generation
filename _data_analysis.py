import numpy as np

BASE = '/home/linux/redstone_ai/dataset'
G = 96
C = 64
meta = np.load(f'{BASE}/voxel96_meta.npz')
N = int(meta['n'])
bb = np.load(f'{BASE}/bb96.npy')  # (N,6) x0,y0,z0,x1,y1,z1
print(f'N={N}')

# 1) bbox 大小分布
sizes = bb[:, 3:6] - bb[:, 0:3] + 1
maxsz = sizes.max(axis=1)
valid = bb[:, 0] >= 0
print(f'valid bboxes: {valid.sum()}/{N}')
print(f'max size >64: {(maxsz > 64).sum()} ({(maxsz>64).mean()*100:.1f}%)')
print(f'max size >48: {(maxsz > 48).sum()} ({(maxsz>48).mean()*100:.1f}%)')
print(f'size hist: <=32: {(maxsz<=32).sum()}, 33-48: {((maxsz>32)&(maxsz<=48)).sum()}, 49-64: {((maxsz>48)&(maxsz<=64)).sum()}, >64: {(maxsz>64).sum()}')

# 2) bbox 中心在 96^3 中的位置 (居中 pad 检查)
centers = (bb[:, 0:3] + bb[:, 3:6]) / 2.0
cmin = centers.min(axis=0)
cmax = centers.max(axis=0)
print(f'bbox center ranges: x[{cmin[0]:.1f},{cmax[0]:.1f}] y[{cmin[1]:.1f},{cmax[1]:.1f}] z[{cmin[2]:.1f},{cmax[2]:.1f}]')

# 中心裁剪后, bbox 超界情况: bbox 是否包含在 [16,80) 内
inside = (bb[:, 0:3] >= 16) & (bb[:, 3:6] < 80)
fully = inside.all(axis=1)
print(f'bbox fully inside center-64: {fully.sum()} ({fully.mean()*100:.1f}%)')

# 3) 裁剪后类别频率 (全量 10559, 中心 64^3) + state 频率
flat_id = np.memmap(f'{BASE}/voxel96_id.dat', dtype=np.uint8, mode='r', shape=(N * G * G * G,))
flat_st = np.memmap(f'{BASE}/voxel96_state.dat', dtype=np.uint16, mode='r', shape=(N * G * G * G,))
id_counts = np.zeros(203, dtype=np.int64)
id_stated = np.zeros(203, dtype=np.int64)
total = 0
stated = 0
for i in range(N):
    v = flat_id[i * G**3:(i + 1) * G**3].reshape(G, G, G)[16:80, 16:80, 16:80]
    s = flat_st[i * G**3:(i + 1) * G**3].reshape(G, G, G)[16:80, 16:80, 16:80]
    nz = v > 0
    total += int(nz.sum())
    ui, inv = np.unique(v[nz], return_inverse=True)
    id_counts[ui] += np.bincount(inv, minlength=len(ui))
    stnz = s[nz] != 0
    stated += int(stnz.sum())
    if stnz.any():
        id_stated[ui] += np.bincount(inv[stnz], minlength=len(ui))
    if (i + 1) % 3000 == 0:
        print(f'  {i+1}/{N}', flush=True)

vocab = [l.strip() for l in open(f'{BASE}/BLOCK_VOCAB_FULL.txt', encoding='utf-8') if l.strip()]
top200 = meta['top200'].astype(int)
id2name = ['minecraft:air'] * 203
for cls in range(1, 201):
    o = int(top200[cls - 1])
    id2name[cls] = vocab[o] if 0 <= o < len(vocab) else '?'
id2name[201] = 'minecraft:other'

print(f'\n=== 中心裁剪64^3后: solid cells={total:,} stated={stated:,} ({100.0*stated/total:.2f}%) ===')
print('=== 类别频率 top 50 (含state标记) ===')
rows = [(c, id2name[c], int(id_counts[c]), int(id_stated[c])) for c in range(203) if id_counts[c] > 0]
rows.sort(key=lambda r: -r[2])
print(f'{"id":>4} {"block":<35} {"cells":>10} {"%":>7} {"stated":>8} {"st%":>6}')
for c, nm, t, s in rows[:50]:
    print(f'{c:>4} {nm:<35} {t:>10,} {100.0*t/total:6.2f}% {s:>8,} {100.0*s/max(t,1):5.1f}%')
print(f'... 共 {len(rows)} 个非空类')
