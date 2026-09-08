
import numpy as np, glob, os
BASE = '/home/linux/redstone_ai/dataset'
meta = np.load(f'{BASE}/voxel96_meta.npz')
print('meta keys:', meta.files, 'n=', int(meta['n']), 'total=', int(meta['total']))
bb = np.load(f'{BASE}/bb96.npy')
print('bb96 shape:', bb.shape)
ss = np.load(f'{BASE}/v2/ss64.npy', mmap_mode='r')
print('ss64 shape:', ss.shape)
files = sorted(glob.glob(f'{BASE}/litematics/*.litematic'))
print('litematic files:', len(files))
# 前3个文件的 bb 与 ss64 idx0/1/2 占用对比(数量)
import litemapy
for idx in [0, 1, 2]:
    b = bb[idx]
    if b[0] < 0:
        print(f'idx{idx}: empty bb')
        continue
    x0,y0,z0,x1,y1,z1 = [int(v) for v in b]
    cx=(x0+x1)/2; cy=(y0+y1)/2; cz=(z0+z1)/2
    ox=int(round(96/2-cx)); oy=int(round(96/2-cy)); oz=int(round(96/2-cz))
    print(f'idx{idx}: bb=({x0},{y0},{z0})-({x1},{y1},{z1}) offset=({ox},{oy},{oz})')
    v = np.asarray(ss[idx])
    print(f'  ss64 occ={int((v>0).sum())}')
# 检查 litematic 与顺序文件名的对应(第一有效文件是否 idx0)
s = litemapy.Schematic.load(files[0])
r = list(s.regions.values())[0]
print('files[0] =', os.path.basename(files[0]), 'region size', r.width, r.height, r.length)
