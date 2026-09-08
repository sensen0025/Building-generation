import numpy as np

OUT = '/home/linux/redstone_ai/dataset/v2'
ss = np.load(f'{OUT}/ss64.npy', mmap_mode='r')
st = np.load(f'{OUT}/ss64_state.npy', mmap_mode='r')
bm = np.load(f'{OUT}/block_map.npy')
print('ss64:', ss.shape, ss.dtype)
print('state:', st.shape, st.dtype)
print('block_map len:', len(bm))

N = ss.shape[0]
total = 0
other = 0
stated = 0
cls_counts = np.zeros(len(bm), dtype=np.int64)
for i in range(0, N, 500):
    v = np.asarray(ss[i:i+500]).reshape(-1)
    s = np.asarray(st[i:i+500]).reshape(-1)
    nz = v > 0
    total += int(nz.sum())
    other += int((v == len(bm)-1).sum())
    stated += int((s[nz] != 0).sum())
    cls_counts += np.bincount(v, minlength=len(bm))[:len(bm)]
print(f'sampled: solid={total:,} other={other:,} ({100.0*other/total:.2f}%) stated={stated:,} ({100.0*stated/total:.2f}%)')

# state 与 id 对齐验证: 无 state 类的 cell 是否 state==0
vocab = [l.strip() for l in open('/home/linux/redstone_ai/dataset/BLOCK_VOCAB_FULL.txt', encoding='utf-8') if l.strip()]
meta = np.load('/home/linux/redstone_ai/dataset/voxel96_meta.npz')
top200 = meta['top200'].astype(int)
id2name = ['minecraft:air'] * 203
for cls in range(1, 201):
    o = int(top200[cls - 1])
    id2name[cls] = vocab[o] if 0 <= o < len(vocab) else '?'
id2name[201] = 'minecraft:other'
# stated class 列表 (从报告推): block_map 里哪些类 old_id 在 stated 集合
print('\n=== 128 类映射摘要 ===')
for ni in range(1, len(bm)):
    oi = int(bm[ni])
    nm = id2name[oi]
    c = int(cls_counts[ni])
    if c > 0:
        flag = 'STATE' if oi in [24,26,31,32,34,35,43,47,48,49,51,52,54,56,58,59,60,63,64,65,68,78,84,85,90,94,96,98,99,100,103,106,112,113,121,122,124,125,131,132,134,136,139,140,141,144,146,149,150,153,158,159,161,162,164,165,166,168,172,173,175,178,179,183,184,186,187,188,190,197,199,201] else ''
        print(f'{ni:3d} {nm:<32} cells={c:>10,} {flag}')
