
import struct
import numpy as np
P = '/home/linux/redstone_ai/render_mc/builds.rhb.v2compressed'
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
    for target in [250]:
        use = {}
        for i in range(target+1):
            nb = struct.unpack('<I', f.read(4))[0]
            data = np.frombuffer(f.read(nb*8), dtype=np.uint16).reshape(-1,4) if nb else np.zeros((0,4),dtype=np.uint16)
            if i == target:
                for (x,y,z,pi) in data:
                    nm = pals[int(pi)][0]
                    ps = pals[int(pi)][1]
                    use[(nm,ps)] = use.get((nm,ps),0)+1
        print(f'=== OLD idx250 total={sum(use.values())} distinct={len(use)} ===', flush=True)
        for (nm,ps),c in sorted(use.items(), key=lambda kv:-kv[1])[:15]:
            print(f'  {nm} [{ps}] : {c}', flush=True)
