# -*- coding: utf-8 -*-
"""远程: 读取新 rhb idx 的实际坐标, 与 ss64 同 idx 非零坐标对比 (判定放置坐标是否错位)."""
import sys, io, os, struct
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

BASE = '/home/linux/redstone_ai/dataset'
RHB = '/home/linux/redstone_ai/render_mc/builds.rhb'
IDXS = [250, 600, 1200, 1800, 2200, 46]

ss = np.load(f'{BASE}/v2/ss64.npy', mmap_mode='r')

# 解析 header: magic(4) npal(4) 每条目 nl(2)name pl(2)props  N(4)
with open(RHB, 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    npal = struct.unpack('<I', f.read(4))[0]
    pos = 8
    for _ in range(npal):
        nl = struct.unpack('<H', f.read(2))[0]; f.read(nl)
        pl = struct.unpack('<H', f.read(2))[0]; f.read(pl)
        pos += 4 + nl + pl
    N = struct.unpack('<I', f.read(4))[0]
    pos += 4
    # 建筑偏移表
    offsets = [pos]
    for _ in range(N):
        nb = struct.unpack('<I', f.read(4))[0]
        pos += 4 + nb * 8
        offsets.append(pos)
    print('N=', N, 'npal=', npal, 'header_end=', offsets[0], flush=True)

def rhb_cells(idx):
    with open(RHB, 'rb') as f:
        f.seek(offsets[idx])
        nb = struct.unpack('<I', f.read(4))[0]
        cells = []
        for _ in range(nb):
            x, y, z, p = struct.unpack('<HHHH', f.read(8))
            cells.append((int(x), int(y), int(z)))
        return cells

for i in IDXS:
    v = np.asarray(ss[i])
    nz_ss = set(map(tuple, np.argwhere(v > 0).tolist()))
    try:
        rc = rhb_cells(i)
    except Exception as e:
        print(f'idx{i}: rhb read ERR {e}', flush=True)
        continue
    nz_rh = set(rc)
    both = len(nz_ss & nz_rh)
    print(f'idx{i}: ss64={len(nz_ss)} rhb={len(nz_rh)} 相同={both} 仅ss64={len(nz_ss-nz_rh)} 仅rhb={len(nz_rh-nz_ss)}', flush=True)
    if len(nz_ss - nz_rh) <= 3:
        print('  ss64-only:', sorted(nz_ss - nz_rh), flush=True)
    else:
        print('  ss64-only sample:', sorted(nz_ss - nz_rh)[:3], flush=True)
    if len(nz_rh - nz_ss) <= 3:
        print('  rhb-only:', sorted(nz_rh - nz_ss), flush=True)
    else:
        print('  rhb-only sample:', sorted(nz_rh - nz_ss)[:3], flush=True)
print('DONE', flush=True)
