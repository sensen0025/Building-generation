
import numpy as np, glob, os, time, json
import litemapy
BASE = '/home/linux/redstone_ai/dataset'
t0 = time.time()
files = sorted(glob.glob(f'{BASE}/litematics/*.litematic'))
print('total litematic:', len(files), flush=True)
# 复现 _voxelize.extract_litematic: 返回 None 当 <5 cells 或 load 失败
def cells_of(path):
    try:
        s = litemapy.Schematic.load(path)
        r = list(s.regions.values())[0]
        W,H,L = abs(r.width), abs(r.height), abs(r.length)
    except Exception:
        return None
    out = []
    for x in range(W):
        for y in range(H):
            for z in range(L):
                try:
                    b = r[x,y,z]
                except IndexError:
                    continue
                if b.id == 'minecraft:air':
                    continue
                out.append((x,y,z,b.id))
                if len(out) >= 5 and len(out) % 10000 == 0:
                    pass
    return out if len(out) >= 5 else None
valid = []
for i, f in enumerate(files):
    c = cells_of(f)
    if c is not None:
        valid.append(os.path.basename(f))
    if (i+1) % 500 == 0:
        print(f'  scan {i+1}/{len(files)} valid={len(valid)} elapsed={time.time()-t0:.0f}s', flush=True)
print('valid total:', len(valid), 'elapsed:', f'{time.time()-t0:.0f}s', flush=True)
with open(f'{BASE}/valid_files_list.txt','w') as fp:
    fp.write('\n'.join(valid))
print('saved valid_files_list.txt', flush=True)
bb = np.load(f'{BASE}/bb96.npy')
print('bb96 N =', bb.shape[0], flush=True)
