
import numpy as np, glob, os, sys
sys.path.insert(0, '/home/linux/redstone_ai')
BASE = '/home/linux/redstone_ai/dataset'
bb = np.load(f'{BASE}/bb96.npy')
files = sorted(glob.glob(f'{BASE}/litematics/*.litematic'))
print('total litematic:', len(files), flush=True)
# 复现 _voxelize.extract_litematic 的过滤
import litemapy
def valid(path):
    try:
        s = litemapy.Schematic.load(path)
        r = list(s.regions.values())[0]
        W,H,L = abs(r.width), abs(r.height), abs(r.length)
    except Exception:
        return False
    n = 0
    for x in range(W):
        for y in range(H):
            for z in range(L):
                try:
                    b = r[x,y,z]
                except IndexError:
                    continue
                if b.id == 'minecraft:air':
                    continue
                n += 1
                if n >= 5:
                    return True
    return n >= 5
valid_files = [f for f in files if valid(f)]
print('valid files:', len(valid_files), flush=True)
# bb 对比(前5)
def get_bb(path):
    s = litemapy.Schematic.load(path)
    r = list(s.regions.values())[0]
    xs,ys,zs = [],[],[]
    W,H,L = abs(r.width), abs(r.height), abs(r.length)
    for x in range(W):
        for y in range(H):
            for z in range(L):
                try:
                    b = r[x,y,z]
                except IndexError:
                    continue
                if b.id == 'minecraft:air':
                    continue
                xs.append(x); ys.append(y); zs.append(z)
    return min(xs),min(ys),min(zs),max(xs),max(ys),max(zs)
for i in range(5):
    f = valid_files[i]
    b = [int(v) for v in bb[i]]
    print(f'idx{i}: file={os.path.basename(f)[:50]} bb96={b} recompute={get_bb(f)}', flush=True)
