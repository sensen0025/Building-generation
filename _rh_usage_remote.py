
import struct
import numpy as np
P = '/home/linux/redstone_ai/render_mc/builds.rhb'
with open(P,'rb') as f:
    magic = f.read(4)
    npal = struct.unpack('<I', f.read(4))[0]
    pals = []
    for i in range(npal):
        nl = struct.unpack('<H', f.read(2))[0]
        nb = f.read(nl)
        pl = struct.unpack('<H', f.read(2))[0]
        pb = f.read(pl)
        pals.append((nb.decode('utf-8','replace'), pb.decode('utf-8','replace')))
    N = struct.unpack('<I', f.read(4))[0]
    # 统计 palette idx 用量
    use = np.zeros(npal, dtype=np.int64)
    for i in range(N):
        nb = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(nb*8), dtype=np.uint16).reshape(-1,4)
        idxs = data[:,3]
        np.add.at(use, idxs, 1)
    total = int(use.sum())
    print('total cells:', total, flush=True)
    # 名字聚合用量
    name_use = {}
    for i,(nm,ps) in enumerate(pals):
        name_use[nm] = name_use.get(nm,0) + int(use[i])
    print('cells by name (desc):', flush=True)
    for nm,c in sorted(name_use.items(), key=lambda kv:-kv[1]):
        print(f'  {nm}  {c:,}  ({100.0*c/total:.2f}%)', flush=True)
