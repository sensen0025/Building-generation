
import struct
import numpy as np
P = '/home/linux/redstone_ai/render_mc/builds.rhb.full'
with open(P,'rb') as f:
    f.read(4)
    npal = struct.unpack('<I', f.read(4))[0]
    pals = []
    for i in range(npal):
        nl = struct.unpack('<H', f.read(2))[0]
        nb = f.read(nl)
        pl = struct.unpack('<H', f.read(2))[0]
        pb = f.read(pl)
        pals.append((nb.decode('utf-8','replace'), pb.decode('utf-8','replace')))
    N = struct.unpack('<I', f.read(4))[0]
    use = np.zeros(npal, dtype=np.int64)
    for i in range(N):
        nb = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(nb*8), dtype=np.uint16).reshape(-1,4)
        if len(data):
            np.add.at(use, data[:,3], 1)
total = int(use.sum())
print('total:', total, flush=True)
# 按命名空间统计
ns_use = {}
name_use = {}
for i,(nm,ps) in enumerate(pals):
    ns = nm.split(':')[0] if ':' in nm else '(none)'
    ns_use[ns] = ns_use.get(ns,0) + int(use[i])
    name_use[nm] = name_use.get(nm,0) + int(use[i])
for ns,c in sorted(ns_use.items(), key=lambda kv:-kv[1]):
    print(f'  ns {ns}: {c:,} ({100.0*c/total:.2f}%)', flush=True)
# 非 minecraft 命名空间的名字 top 30
print('--- non-minecraft names top 30 ---', flush=True)
for nm,c in sorted(name_use.items(), key=lambda kv:-kv[1]):
    if not nm.startswith('minecraft:'):
        print(f'  {nm}  {c:,}  ({100.0*c/total:.3f}%)', flush=True)
