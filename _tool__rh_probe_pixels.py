# -*- coding: utf-8 -*-
import os, glob
import numpy as np
from PIL import Image
OUT = '/home/linux/redstone_ai/render_mc/out'
for idx in [0, 1, 4, 5, 100, 500, 3000, 5338]:
    f = os.path.join(OUT, f'bld_{idx:04d}.png')
    im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
    H, W, _ = im.shape
    sky = np.array([177.0, 205.0, 255.0])
    d = np.abs(im - sky).mean(2)
    skyfrac = (d < 25).mean()
    tl = im[10:50, 10:50].mean(0)
    tr = im[10:50, -50:-10].mean(0)
    mid = im[H // 2 - 5:H // 2 + 5, W // 2 - 5:W // 2 + 5].mean(0)
    print(f'{idx:4d} skyfrac={skyfrac*100:5.1f}% cornerTL={np.round(tl).astype(int).tolist()} cornerTR={np.round(tr).astype(int).tolist()} center={np.round(mid).astype(int).tolist()}')
