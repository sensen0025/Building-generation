import numpy as np
from PIL import Image
import glob, os
for f in sorted(glob.glob('/home/linux/redstone_ai/rot_preview/bld_*.png')):
    a = np.asarray(Image.open(f).convert('RGB'))
    uniq = len(np.unique(a.reshape(-1,3), axis=0))
    print(os.path.basename(f), os.path.getsize(f), 'shape', a.shape, 'uniq', uniq, 'mean', round(float(a.mean()),1), 'corner', a[0,0].tolist())
