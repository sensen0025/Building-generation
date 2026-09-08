import numpy as np
from PIL import Image
import os, glob

# check the newly-rendered rotated images (latidx not multiple of 4) at the start
checks = ['bld_0001.png','bld_0002.png','bld_0003.png','bld_0005.png','bld_0006.png','bld_0007.png',
          'bld_0009.png','bld_0010.png','bld_0011.png','bld_0030.png','bld_0031.png']
base = '/home/linux/redstone_ai/render_output_rot'
for f in checks:
    p = os.path.join(base, f)
    if not os.path.exists(p):
        print(f, 'MISSING'); continue
    a = np.asarray(Image.open(p).convert('RGB')).astype(float)
    uniq = len(np.unique(a.reshape(-1,3), axis=0))
    print(f, os.path.getsize(p), 'mean', round(a.mean(),1), 'corner', (a[0,0].astype(int)).tolist())
