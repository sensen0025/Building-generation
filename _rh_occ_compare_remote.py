# -*- coding: utf-8 -*-
"""远程对比: 对若干 idx, 比较 ss64(voxel)占用 / rebuild_full 重建后占用 / litematic 源占用.
用于判断 '一条方块' 是源数据如此, 还是裁剪/重建丢块. 一次远程运行, 集中对比."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import litemapy

BASE = '/home/linux/redstone_ai/dataset'
IDXS = [0, 6, 46, 250, 600, 1200, 1800, 2200]

ss = np.load(f'{BASE}/v2/ss64.npy', mmap_mode='r')
st = np.load(f'{BASE}/v2/ss64_state.npy', mmap_mode='r')
names = [l.strip() for l in open(f'{BASE}/valid_files_list.txt', encoding='utf-8') if l.strip()]
print('N names =', len(names), flush=True)

def litematic_occ(path):
    s = litemapy.Schematic.load(path)
    r = list(s.regions.values())[0]
    W, H, L = abs(r.width), abs(r.height), abs(r.length)
    n = 0
    for x in range(W):
        for y in range(H):
            for z in range(L):
                try:
                    b = r[x, y, z]
                except IndexError:
                    continue
                if b.id != 'minecraft:air':
                    n += 1
    return n

for i in IDXS:
    if i >= len(names):
        print(i, 'out of range'); continue
    v = np.asarray(ss[i])
    sarr = np.asarray(st[i])
    occ = int((v > 0).sum())
    nz = np.argwhere(v > 0)
    if len(nz):
        mn, mx = nz.min(0), nz.max(0)
        sp = (mx - mn + 1).tolist()
    else:
        sp = None
    fn = names[i]
    path = os.path.join(f'{BASE}/litematics', fn)
    try:
        l_occ = litematic_occ(path)
    except Exception as e:
        l_occ = f'ERR {e}'
    print(f'idx{i}: ss64_occ={occ} spans={sp} litematic_occ={l_occ} file={fn}', flush=True)
print('DONE', flush=True)
