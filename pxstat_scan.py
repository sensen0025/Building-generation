import numpy as np
from PIL import Image
import os, glob
base = '/home/linux/redstone_ai/render_output_rot'
files = sorted(glob.glob(os.path.join(base, 'bld_*.png')))
print('total files:', len(files))
bad = []
for f in files:
    try:
        a = np.asarray(Image.open(f).convert('RGB')).astype(float)
        std = a.std()
        if std < 25:
            bad.append((os.path.basename(f), round(std,1), os.path.getsize(f)))
    except Exception as e:
        bad.append((os.path.basename(f), 'ERR', str(e)))
print('near-blank/low-std count:', len(bad))
for b in bad[:30]:
    print('  BAD', b)
