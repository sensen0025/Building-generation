# -*- coding: utf-8 -*-
import os, sys
import numpy as np
from PIL import Image
OUT = '/home/linux/redstone_ai/render_mc/out'
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1795
f = os.path.join(OUT, f'bld_{idx:04d}.png')
if not os.path.exists(f):
    print('missing', f)
    sys.exit(1)
im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
H, W, _ = im.shape
g = im.reshape(8, H // 8, 8, W // 8, 3).mean(axis=(1, 3))
for row in g:
    print(' '.join(f'[{int(c[0])},{int(c[1])},{int(c[2])}]' for c in row))
