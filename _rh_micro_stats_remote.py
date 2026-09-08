# -*- coding: utf-8 -*-
"""远程: (1) 统计 ss64 微型建筑(n<200)占比; (2) 对比 out(新) vs out_v2compressed(旧) 同 idx 图像 sky/fg 差异."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from PIL import Image

BASE = '/home/linux/redstone_ai/dataset'
ss = np.load(f'{BASE}/v2/ss64.npy', mmap_mode='r')
N = ss.shape[0]
import collections
cnt = collections.Counter()
for i in range(N):
    v = np.asarray(ss[i])
    occ = int((v > 0).sum())
    if occ < 50: cnt['<50'] += 1
    elif occ < 200: cnt['50-200'] += 1
    elif occ < 1000: cnt['200-1k'] += 1
    else: cnt['>1k'] += 1
print('occ distribution:', dict(cnt), 'N=', N, flush=True)

def skyfg(p):
    im = np.asarray(Image.open(p).convert('RGB')).astype(np.float32)
    sky = np.array([177., 205., 255.])
    d = np.abs(im - sky).mean(2)
    return (d < 25).mean() * 100

print('\nidx  new_sky%  old_sky%  (out vs out_v2compressed)', flush=True)
for idx in [0, 6, 46, 250, 600, 1200, 1800, 2200]:
    np_ = f'/home/linux/redstone_ai/render_mc/out/bld_{idx:04d}.png'
    op = f'/home/linux/redstone_ai/render_mc/out_v2compressed/bld_{idx:04d}.png'
    ns = skyfg(np_) if os.path.exists(np_) else float('nan')
    os_ = skyfg(op) if os.path.exists(op) else float('nan')
    print(f'{idx:4d}  {ns:7.1f}  {os_:7.1f}', flush=True)
print('DONE', flush=True)
