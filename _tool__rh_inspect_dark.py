# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
OUT = '/home/linux/redstone_ai/render_mc/out'
im = np.asarray(Image.open(os.path.join(OUT, 'bld_0281.png')).convert('RGB')).astype(np.float32)
H, W, _ = im.shape
# 8x8 网格均值
g = im.reshape(8, H // 8, 8, W // 8, 3).mean(axis=(1, 3))
print('8x8 grid means (R,G,B):')
for row in g:
    print(' '.join(f'[{int(c[0])},{int(c[1])},{int(c[2])}]' for c in row))
# 顶部/底部颜色分布
print('top rows uniq-ish:', np.unique(im[:30].reshape(-1, 3).astype(np.uint8), axis=0)[:5].tolist())
print('bottom rows mean:', np.round(im[-30:].mean((0, 1))).astype(int).tolist())
# 亮像素分布(>100)
lum = im.mean(2)
ys, xs = np.nonzero(lum > 90)
print('bright px', len(ys), 'bbox rows', (ys.min(), ys.max()) if len(ys) else None, 'cols', (xs.min(), xs.max()) if len(xs) else None)
