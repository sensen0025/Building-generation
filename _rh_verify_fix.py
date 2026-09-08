# -*- coding: utf-8 -*-
"""远程: open + 渲染验证 idx (light fill 已禁用), 报告 sky%."""
import sys, io, os, json, time, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from PIL import Image

BASE = 'http://127.0.0.1:8917'
OUT = '/home/linux/redstone_ai/render_mc/out'
SKY = np.array([177., 205., 255.])

def post(path, timeout=240):
    req = urllib.request.Request(BASE + path, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def skyf(p):
    im = np.asarray(Image.open(p).convert('RGB')).astype(np.float32)
    d = np.abs(im - SKY).mean(2)
    return (d < 25).mean() * 100

try:
    print('open:', post('/open', timeout=300))
except Exception as e:
    print('open ERR', repr(e)[:150])

IDXS = [250, 600, 2200, 17, 46, 0]
for idx in IDXS:
    f = os.path.join(OUT, f'vfy_bld_{idx:04d}.png')  # 用独立名避免覆盖旧验证
    if os.path.exists(f):
        try: os.remove(f)
        except Exception: pass
    try:
        r = post(f'/render?idx={idx}', timeout=300)
        # 渲染器写 bld_%04d.png, 等它出现后改名
        src = os.path.join(OUT, f'bld_{idx:04d}.png')
        for _ in range(150):
            if os.path.exists(src) and os.path.getsize(src) > 800:
                break
            time.sleep(0.5)
        if os.path.exists(src):
            os.replace(src, f)
            print(f'idx{idx}: ok ms={r.get("ms")} sky={skyf(f):.1f}% size={os.path.getsize(f)}')
        else:
            print(f'idx{idx}: render ok but no file {r}')
    except Exception as e:
        print(f'idx{idx}: ERR {repr(e)[:120]}')
    time.sleep(0.5)
print('DONE', flush=True)
