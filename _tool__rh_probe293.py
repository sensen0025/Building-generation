# -*- coding: utf-8 -*-
import json, os
import numpy as np
from PIL import Image
meta = json.load(open('/home/linux/redstone_ai/render_mc/builds_meta.json'))['meta']
m = meta[293]
print('293 meta:', m)
im = np.asarray(Image.open('/home/linux/redstone_ai/render_mc/out/bld_0293.png').convert('RGB')).astype(np.float32)
print('mean', np.round(im.mean((0,1))).astype(int).tolist(), 'std', round(float(im.std()),1), 'min', int(im.min()), 'max', int(im.max()))
print('TL', np.round(im[8:40,8:40].mean((0,1))).astype(int).tolist(), 'TR', np.round(im[8:40,-40:-8].mean((0,1))).astype(int).tolist())
print('center', np.round(im[im.shape[0]//2-6:im.shape[0]//2+6, im.shape[1]//2-6:im.shape[1]//2+6].mean((0,1))).astype(int).tolist())
# 亮度>120 占比
lum = im.mean(2)
print('bright frac:', round(float((lum > 120).mean()), 4), ' dark frac:', round(float((lum < 40).mean()), 4))
