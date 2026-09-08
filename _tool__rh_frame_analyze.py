# -*- coding: utf-8 -*-
"""取景分析 v2: 以四角均值为背景, 前景=距背景远; 报 fg bbox/中心/覆盖率/背景色."""
import os, sys, glob
import numpy as np
from PIL import Image

OUT = '/home/linux/redstone_ai/render_mc/out'
files = sorted(glob.glob(os.path.join(OUT, '*.png')))
print('files:', len(files))
for f in files:
    im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
    H, W, _ = im.shape
    corners = np.concatenate([im[:40, :40].reshape(-1, 3), im[:40, -40:].reshape(-1, 3),
                              im[-40:, :40].reshape(-1, 3), im[-40:, -40:].reshape(-1, 3)])
    bg = corners.mean(0)
    dist = np.abs(im - bg).mean(2)
    # 前景 = 距背景显著
    th = max(28.0, dist.std() * 0.6)
    fg = dist > th
    cov = fg.mean()
    ys, xs = np.nonzero(fg)
    if len(ys) == 0:
        print(os.path.basename(f), 'bg', np.round(bg).astype(int).tolist(), 'NO FG')
        continue
    bb = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    cx = (bb[0] + bb[1]) / 2 / W
    cy = (bb[2] + bb[3]) / 2 / H
    # 去噪: 只统计连通大块? 简化: 用分位数再定界
    print(f"{os.path.basename(f)} bg={np.round(bg).astype(int).tolist()} cov={cov*100:.1f}% bbox={bb} c=({cx:.2f},{cy:.2f})")
