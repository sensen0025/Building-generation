# -*- coding: utf-8 -*-
"""远程: 对比 ss64 非零坐标 vs 新 rhb 实际坐标(几个 idx), 判断重建坐标是否与 ss64/voxel 对齐."""
import sys, io, os, struct
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

BASE = '/home/linux/redstone_ai/dataset'
RHB = '/home/linux/redstone_ai/render_mc/builds.rhb'
IDXS = [0, 6, 46, 250, 600, 1200, 1800, 2200]

ss = np.load(f'{BASE}/v2/ss64.npy', mmap_mode='r')

# 解析 rhb offsets
with open(RHB, 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    npal = struct.unpack('<I', f.read(4))[0]
    off = 8
    for _ in range(npal):
        nl = struct.unpack('<H', f.read(2))[0]; f.read(nl)
        pl = struct.unpack('<H', f.read(2))[0]; f.read(pl)
        off += 4 + nl + pl
    N = struct.unpack('<I', f.read(4))[0]
    off += 4
    offsets = [off]
    for _ in range(N):
        nb = struct.unpack('<I', f.read(4))[0]
        off += 4 + nb * 8
        offsets.append(off)
    print('rhb N =', N, 'npal =', npal, flush=True)

def rhb_cells(idx):
    with open(RHB, 'rb') as f:
        f.seek(offsets[idx])
        nb = struct.unpack('<I', f.read(4))[0]
        cells = []
        for _ in range(nb):
            x, y, z, p = struct.unpack('<HHHH', f.read(8))
            cells.append((x, y, z))
    return cells

for i in IDXS:
    v = np.asarray(ss[i])
    nz = set(map(tuple, np.argwhere(v > 0)))
    rc = set(rhb_cells(i))
    print(f'idx{i}: ss64={len(nz)} rhb={len(rc)} 相同={len(nz & rc)} 仅ss64={len(nz-rc)} 仅rhb={len(rc-nz)}', flush=True)
    if len(nz - rc) <= 5 and len(rc - nz) <= 5:
        print('   ss64-only:', sorted(nz-rc), ' rhb-only:', sorted(rc-nz), flush=True)
    else:
        a = sorted(nz-rc)[:5]; b = sorted(rc-nz)[:5]
        print('   ss64-only sample:', a, ' rhb-only sample:', b, flush=True)
print('DONE', flush=True)
