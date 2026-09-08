# -*- coding: utf-8 -*-
"""远程: 正确构建 offsets(跳过每建筑数据), 对比新 rhb idx250/600 坐标 vs ss64."""
import sys, io, struct, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

BASE = '/home/linux/redstone_ai/dataset'
RHB = '/home/linux/redstone_ai/render_mc/builds.rhb'
META = '/home/linux/redstone_ai/render_mc/builds_meta.json'
IDXS = [0, 6, 46, 250, 600, 2200]

ss = np.load(f'{BASE}/v2/ss64.npy', mmap_mode='r')

with open(RHB, 'rb') as f:
    f.read(4)  # magic
    npal = struct.unpack('<I', f.read(4))[0]
    for _ in range(npal):
        nl = struct.unpack('<H', f.read(2))[0]; f.read(nl)
        pl = struct.unpack('<H', f.read(2))[0]; f.read(pl)
    N = struct.unpack('<I', f.read(4))[0]
    offsets = [f.tell()]
    counts = []
    for _ in range(N):
        nb = struct.unpack('<I', f.read(4))[0]
        counts.append(nb)
        f.read(nb * 8)
        offsets.append(f.tell())
    print('N', N, 'counts idx0-3:', counts[:4], 'idx250:', counts[250], 'file_end:', f.tell(), flush=True)

def rhb_cells(idx):
    with open(RHB, 'rb') as f:
        f.seek(offsets[idx])
        nb = struct.unpack('<I', f.read(4))[0]
        cells = []
        for _ in range(nb):
            x, y, z, p = struct.unpack('<HHHH', f.read(8))
            cells.append((int(x), int(y), int(z)))
        return cells

meta = json.load(open(META))['meta']
for i in IDXS:
    v = np.asarray(ss[i])
    nz_ss = set(map(tuple, np.argwhere(v > 0).tolist()))
    rc = rhb_cells(i)
    nz_rh = set(rc)
    both = len(nz_ss & nz_rh)
    m = meta[i]
    print(f'idx{i}: meta_n={m["n"]} ss64={len(nz_ss)} rhb={len(nz_rh)} 相同={both} '
          f'仅ss64={len(nz_ss-nz_rh)} 仅rhb={len(nz_rh-nz_ss)}', flush=True)
    print(f'   meta_bbox={m["bbox_min"]}..{m["bbox_max"]}', flush=True)
    if len(nz_ss - nz_rh):
        print('   ss64-only sample:', sorted(nz_ss - nz_rh)[:4], flush=True)
    if len(nz_rh - nz_ss):
        print('   rhb-only sample:', sorted(nz_rh - nz_ss)[:4], flush=True)
print('DONE', flush=True)
