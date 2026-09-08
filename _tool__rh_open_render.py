# -*- coding: utf-8 -*-
"""/open + 渲染含水样例 idx6/17 并检查底部水层."""
import json, time, urllib.request, os
import numpy as np
from PIL import Image

def post(path, timeout=300):
    req = urllib.request.Request('http://127.0.0.1:8917' + path, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get(path, timeout=30):
    with urllib.request.urlopen('http://127.0.0.1:8917' + path, timeout=timeout) as r:
        return json.loads(r.read().decode())

print('open:', post('/open', timeout=300), flush=True)
OUT = '/home/linux/redstone_ai/render_mc/out'
for idx in [6, 17, 46, 0]:
    f = os.path.join(OUT, f'bld_{idx:04d}.png')
    if os.path.exists(f):
        os.remove(f)
    try:
        r = post(f'/render?idx={idx}', timeout=200)
        print(idx, r, flush=True)
    except Exception as e:
        print(idx, 'ERR', repr(e)[:100], flush=True)
    if os.path.exists(f):
        im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
        H, W, _ = im.shape
        # 底部 1/4 是否大片蓝色水面? 蓝色判定 b>r+30 & b>g+15
        bot = im[int(H*0.72):, :]
        bluish = ((bot[..., 2] - bot[..., 0]) > 30) & ((bot[..., 2] - bot[..., 1]) > 15)
        print('   mean', round(float(im.mean()), 1), 'bottom-bluish%', round(float(bluish.mean())*100, 1), flush=True)
print('status:', get('/status'), flush=True)
