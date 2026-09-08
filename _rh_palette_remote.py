
import struct
P = '/home/linux/redstone_ai/render_mc/builds.rhb'
with open(P,'rb') as f:
    magic = f.read(4)
    npal = struct.unpack('<I', f.read(4))[0]
    pals = []
    for i in range(npal):
        nl, pl = struct.unpack('<HH', f.read(4))
        nb = f.read(nl); pb = f.read(pl)
        pals.append((nb.decode('utf-8','replace'), pb.decode('utf-8','replace')))
    N = struct.unpack('<I', f.read(4))[0]
print('magic', magic, 'palette', npal, 'builds', N, flush=True)
names = {}
for nm, ps in pals:
    names[nm] = names.get(nm, 0) + 1
print('distinct names:', len(names), flush=True)
for nm, c in sorted(names.items(), key=lambda kv:-kv[1]):
    print(f'  {nm}  ({c})', flush=True)
