# -*- coding: utf-8 -*-
import json, time, urllib.request, os
import numpy as np
from PIL import Image

def post(path, timeout=200):
    req = urllib.request.Request('http://127.0.0.1:8917' + path, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

OUT = '/home/linux/redstone_ai/render_mc/out'
for idx in [1085, 1090]:
    f = os.path.join(OUT, f'bld_{idx:04d}.png')
    if os.path.exists(f):
        os.remove(f)
    try:
        r = post(f'/render?idx={idx}', timeout=180)
        print(idx, r)
    except Exception as e:
        print(idx, 'ERR', repr(e)[:100])
    time.sleep(0.5)
    if os.path.exists(f):
        im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
        print('   mean', np.round(im.mean()).astype(int).tolist(), 'TL', np.round(im[8:40,8:40].mean((0,1))).astype(int).tolist())
