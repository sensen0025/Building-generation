# -*- coding: utf-8 -*-
import os, glob, time
import numpy as np
from PIL import Image

def post(path, timeout=200):
    import json, urllib.request
    req = urllib.request.Request('http://127.0.0.1:8917' + path, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

OUT = '/home/linux/redstone_ai/render_mc/out'
# 时间冻结验证: 连续渲染同一 idx 两次(间隔 ~15s), 亮度应稳定且为白天
for idx in [400]:
    f = os.path.join(OUT, f'bld_{idx:04d}.png')
    vals = []
    for t in range(2):
        if os.path.exists(f):
            os.remove(f)
        r = post(f'/render?idx={idx}', timeout=180)
        im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
        vals.append(round(float(im.mean()), 0))
        time.sleep(15)
    print('idx', idx, 'means over time:', vals)
