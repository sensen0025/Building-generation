# -*- coding: utf-8 -*-
"""远端驱动: open + render 测试 + 输出统计."""
import json, time, urllib.request, os, sys, glob
import numpy as np
from PIL import Image

BASE = 'http://127.0.0.1:8917'

def post(path, timeout=240):
    req = urllib.request.Request(BASE + path, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())

print('ping:', get('/ping'))
print('open:', post('/open', timeout=200))
print('status:', get('/status'))

OUT = '/home/linux/redstone_ai/render_mc/out'
for idx in [0, 1, 100, 4]:
    try:
        r = post(f'/render?idx={idx}', timeout=180)
        print('render', idx, r)
    except Exception as e:
        print('render', idx, 'ERR', repr(e))
    time.sleep(0.5)

print('status:', get('/status'))
files = sorted(glob.glob(os.path.join(OUT, '*.png')))
print('png count:', len(files))
for f in files:
    im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32) / 255.0
    print(os.path.basename(f), im.shape[:2], 'mean', np.round(im.mean(axis=(0,1)),3).tolist(),
          'std', round(float(im.std()),3), 'uniq', len(np.unique(im.reshape(-1,3), axis=0)))
