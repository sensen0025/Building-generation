# -*- coding: utf-8 -*-
"""中间质检: 抽样30张已渲染 png 做统计(sky占比/建筑包围盒/多样性)."""
import os, sys, glob, random, hashlib
import numpy as np
from PIL import Image

OUT = '/home/linux/redstone_ai/render_mc/out'
files = sorted(glob.glob(os.path.join(OUT, 'bld_*.png')))
print('total so far:', len(files))
random.seed(7)
samp = random.sample(files, min(30, len(files)))
sky = np.array([177.0, 205.0, 255.0])
bad = []
for f in samp:
    im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
    d = np.abs(im - sky).mean(2)
    skyf = (d < 25).mean()
    fg = d >= 25
    ys, xs = np.nonzero(fg)
    if len(ys) == 0:
        bad.append((os.path.basename(f), 'NOFG', float(skyf)))
        continue
    H, W = im.shape[:2]
    cov = len(ys) / (H * W)
    cx = (xs.min() + xs.max()) / 2 / W
    cy = (ys.min() + ys.max()) / 2 / H
    if not (0.2 <= cov <= 0.9) or not (0.25 <= cx <= 0.75) or not (0.25 <= cy <= 0.75):
        bad.append((os.path.basename(f), f'cov={cov:.2f} c=({cx:.2f},{cy:.2f}) sky={skyf:.2f}'))
hs = set()
for f in files:
    hs.add(hashlib.md5(open(f, 'rb').read()).hexdigest())
print('distinct md5:', len(hs), '/', len(files))
print('outliers in sample:', len(bad))
for b in bad[:15]:
    print('  ', b)
