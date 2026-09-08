# -*- coding: utf-8 -*-
"""全量 5339 PNG 数值终检: 平均亮度直方图/暗帧/空帧/尺寸一致性."""
import os, glob, hashlib, collections
import numpy as np
from PIL import Image

OUT = '/home/linux/redstone_ai/render_mc/out'
files = sorted(glob.glob(os.path.join(OUT, 'bld_*.png')))
print('files:', len(files))
means = []
sizes = collections.Counter()
dark = []
skyonly = []
hs = {}
for f in files:
    try:
        im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
    except Exception as e:
        print('unreadable', f, e)
        continue
    sizes[im.shape[:2]] += 1
    m = float(im.mean())
    means.append(m)
    if m <= 45:
        dark.append((os.path.basename(f), round(m, 1)))
    h = hashlib.md5(open(f, 'rb').read()).hexdigest()
    hs.setdefault(h, []).append(os.path.basename(f))
means = np.array(means)
print('size set:', dict(sizes))
print('mean lum: p1=%.0f p50=%.0f p99=%.0f min=%.0f' % (np.percentile(means, 1), np.percentile(means, 50), np.percentile(means, 99), means.min()))
print('dark(<=45):', len(dark), dark[:10])
dups = {h: v for h, v in hs.items() if len(v) > 1}
print('dup groups:', len(dups), 'files in dups:', sum(len(v) for v in dups.values()))
for h, v in list(dups.items())[:5]:
    print('   dup:', v[:6])
# 直方图桶
hist, edges = np.histogram(means, bins=[0, 45, 70, 95, 120, 145, 170, 200, 256])
print('lum histogram:', dict(zip([f'{int(e)}-{int(edges[i+1])}' for i, e in enumerate(edges[:-1])], hist.tolist())))
