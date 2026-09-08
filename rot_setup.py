# -*- coding: utf-8 -*-
"""Setup rotated condition image dir: copy original 0° images to i*4 positions."""
import os, shutil, glob

SRC = '/home/linux/redstone_ai/render_output'
DST = '/home/linux/redstone_ai/render_output_rot'
os.makedirs(DST, exist_ok=True)

orig = sorted(glob.glob(os.path.join(SRC, 'bld_*.png')))
print('original images:', len(orig))
n_copied = 0
for p in orig:
    i = int(os.path.basename(p)[4:-4])
    dst = os.path.join(DST, f'bld_{i*4:04d}.png')
    if not os.path.exists(dst):
        shutil.copy2(p, dst)
        n_copied += 1
print('copied 0° -> i*4 positions:', n_copied)
print('DST total bld_*.png:', len(glob.glob(os.path.join(DST, 'bld_*.png'))))
print('DONE')
