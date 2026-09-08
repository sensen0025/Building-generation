# -*- coding: utf-8 -*-
"""远程: 统计当前 out/ 全部图的 sky 占比分布 (判断批量渲染天空图比例)."""
import sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from PIL import Image

OUT = '/home/linux/redstone_ai/render_mc/out'
files = sorted(glob.glob(os.path.join(OUT, 'bld_*.png')))
print('files:', len(files), flush=True)
sky = np.array([177., 205., 255.])
buckets = {'sky<10': 0, '10-30': 0, '30-50': 0, '50-70': 0, '70-90': 0, '90-100': 0}
bad = []
for f in files:
    try:
        im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
        d = np.abs(im - sky).mean(2)
        s = (d < 25).mean() * 100
        if s < 10: buckets['sky<10'] += 1
        elif s < 30: buckets['10-30'] += 1
        elif s < 50: buckets['30-50'] += 1
        elif s < 70: buckets['50-70'] += 1
        elif s < 90: buckets['70-90'] += 1
        else:
            buckets['90-100'] += 1
            bad.append(os.path.basename(f))
    except Exception as ex:
        print('ERR', f, ex, flush=True)
print('sky distribution:', buckets, flush=True)
print('sky>=90 count:', len(bad), 'sample:', bad[:20], flush=True)
# 也看每个 idx 的 sky(前 100)
print('\nidx sky% (first 120):', flush=True)
for f in files[:120]:
    try:
        im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
        d = np.abs(im - sky).mean(2)
        print(os.path.basename(f), round(float((d < 25).mean() * 100), 1), flush=True)
    except Exception:
        pass
print('DONE', flush=True)
