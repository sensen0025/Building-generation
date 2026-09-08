
import struct, json
import numpy as np
P = '/home/linux/redstone_ai/render_mc/builds.rhb.full'
M = '/home/linux/redstone_ai/render_mc/builds_meta.full.json'
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
    use = np.zeros(npal, dtype=np.int64)
    meta = []
    for i in range(N):
        nb = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(nb*8), dtype=np.uint16).reshape(-1,4)
        if len(data):
            np.add.at(use, data[:,3], 1)
    total = int(use.sum())
print('palette:', npal, 'N:', N, 'total cells:', total, flush=True)
name_use = {}
for i,(nm,ps) in enumerate(pals):
    name_use[nm] = name_use.get(nm,0) + int(use[i])
stone = name_use.get('minecraft:stone',0)
print(f'stone: {stone:,} ({100.0*stone/total:.2f}%)', flush=True)
print('--- top 15 names by cells ---', flush=True)
for nm,c in sorted(name_use.items(), key=lambda kv:-kv[1])[:15]:
    print(f'  {nm}  {c:,}  ({100.0*c/total:.2f}%)', flush=True)
# other 残留? any 'other' or unknown-ish names
others = [nm for nm in name_use if 'other' in nm or nm.count(':')!=1]
print('suspicious names:', others[:10], flush=True)
# 旧 vs 新 stone 对比
print('OLD stone was 28.9% (4.84M); new:', f'{stone:,} ({100.0*stone/total:.2f}%)', flush=True)
