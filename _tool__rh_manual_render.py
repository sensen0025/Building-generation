# -*- coding: utf-8 -*-
"""手动渲染 281/284 并检查像素, 判断为何 QA 失败."""
import json, time, urllib.request, os
import numpy as np
from PIL import Image

def post(path, timeout=200):
    req = urllib.request.Request('http://127.0.0.1:8917' + path, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

OUT = '/home/linux/redstone_ai/render_mc/out'
for idx in [279, 281, 284]:
    f = os.path.join(OUT, f'bld_{idx:04d}.png')
    if os.path.exists(f):
        os.remove(f)
    try:
        r = post(f'/render?idx={idx}', timeout=180)
        print(idx, r)
    except Exception as e:
        print(idx, 'ERR', repr(e)[:120])
    time.sleep(0.5)
    if os.path.exists(f):
        im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
        sky = np.array([177.0, 205.0, 255.0])
        d = np.abs(im - sky).mean(2)
        skyf = (d < 25).mean()
        print('   skyf', round(float(skyf), 3), 'mean', np.round(im.mean()).astype(int).tolist() if im.ndim==3 else '',
              'TL', np.round(im[8:40, 8:40].mean((0, 1))).astype(int).tolist(),
              'BR', np.round(im[-40:-8, -40:-8].mean((0, 1))).astype(int).tolist())
