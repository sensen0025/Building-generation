# -*- coding: utf-8 -*-
import os, glob
import numpy as np
from PIL import Image
OUT = '/home/linux/redstone_ai/render_mc/out'
for idx in [19, 29, 44, 48, 109, 187, 259, 274, 298, 4, 0]:
    f = os.path.join(OUT, f'bld_{idx:04d}.png')
    if not os.path.exists(f):
        print(idx, 'MISSING')
        continue
    im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
    H, W, _ = im.shape
    print(idx, 'mean', np.round(im.mean((0, 1))).astype(int).tolist(),
          'min', np.round(im.min()).astype(int), 'max', np.round(im.max()).astype(int),
          'std', round(float(im.std()), 1),
          'cornerTL', np.round(im[8:40, 8:40].mean((0, 1))).astype(int).tolist(),
          'cornerBL', np.round(im[-40:-8, 8:40].mean((0, 1))).astype(int).tolist(),
          'center', np.round(im[H//2-8:H//2+8, W//2-8:W//2+8].mean((0, 1))).astype(int).tolist())
