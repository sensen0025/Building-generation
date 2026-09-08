import numpy as np
from PIL import Image
import glob, sys

def stat(p):
    a = np.asarray(Image.open(p).convert('RGB'))
    uniq = len(np.unique(a.reshape(-1, 3), axis=0))
    return a.shape, uniq, round(float(a.mean()), 1), round(float(a.std()), 1), a.reshape(-1,3)[0].tolist()

print("--- new rot_preview ---")
for f in sorted(glob.glob('/home/linux/redstone_ai/rot_preview/bld_*.png')):
    print(f.split('/')[-1], stat(f))

print("--- existing render_output sample ---")
for f in ['bld_0000.png', 'bld_0001.png', 'bld_0038.png', 'bld_0030.png']:
    p = f'/home/linux/redstone_ai/render_output/{f}'
    try:
        import os
        print(f, os.path.getsize(p), stat(p))
    except Exception as e:
        print(f, 'ERR', e)
