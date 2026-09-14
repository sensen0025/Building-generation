# -*- coding: utf-8 -*-
"""为旋转槽位 (k=1,2,3) 补建 litematic, 供 4 旋转条件图渲染.

严格遵守生产约定:
  - 数据源 = train/ss64_all.npy 的 slot (已是 rot_y_64^k 后的体素, 与训练完全一致)
  - 复用 _export_new_litematics.to_litematic / unpack_state (与生产同一套 id->name 与 state->props)
  - 裁到内容 bbox 后写 Region(0,0,0,D,H,W)  (与生产一致, 保证取景一致)

用法:
  python _build_rot_litematics.py <slot_start> <slot_end> [--all]
    --all : 连 k=0 槽位一起建 (仅用于验证复现生产)
  输出: /data/redstone_v4/litematics_rot/{slot:05d}.litematic
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/data/redstone_v4/work')
import _export_new_litematics as E  # noqa: E402  生产同款导出函数 (无 torch 依赖)

N0 = 21356
OUT = '/data/redstone_v4/litematics_rot'
os.makedirs(OUT, exist_ok=True)

start = int(sys.argv[1])
end = int(sys.argv[2])
include_k0 = '--all' in sys.argv

ss = np.load('/data/redstone_v4/train/ss64_all.npy', mmap_mode='r')
st = np.load('/data/redstone_v4/train/ss64_state_all_rotfix.npy', mmap_mode='r')

t0 = time.time()
built = 0
skipped = 0
for slot in range(start, min(end, ss.shape[0])):
    k = (slot - N0) % 4
    if k == 0 and not include_k0:
        continue
    outp = os.path.join(OUT, f'{slot:05d}.litematic')
    if os.path.exists(outp) and os.path.getsize(outp) > 200:
        skipped += 1
        continue
    vox = np.asarray(ss[slot])
    stv = np.asarray(st[slot])
    nzc = np.argwhere(vox > 0)
    if len(nzc) == 0:
        continue
    mn, mx = nzc.min(0), nzc.max(0)
    vox = vox[mn[0]:mx[0] + 1, mn[1]:mx[1] + 1, mn[2]:mx[2] + 1]
    stv = stv[mn[0]:mx[0] + 1, mn[1]:mx[1] + 1, mn[2]:mx[2] + 1]
    try:
        E.to_litematic(vox, stv, f'slot{slot}', outp)
    except Exception as e:
        print(f'ERR slot {slot}: {str(e)[:120]}', flush=True)
        continue
    built += 1
    if built % 200 == 0:
        el = time.time() - t0
        print(f'[{start}-{end}] built={built} skip={skipped} {el:.0f}s ({built/max(el,1e-9):.2f}/s)', flush=True)

print(f'DONE range[{start},{end}) built={built} skipped={skipped} in {time.time()-t0:.0f}s', flush=True)
