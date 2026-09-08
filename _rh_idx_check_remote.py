
import struct, json
import numpy as np
P = '/home/linux/redstone_ai/render_mc/builds.rhb.full'
M = json.load(open('/home/linux/redstone_ai/render_mc/builds_meta.full.json'))
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
    # 找到 idx250 和 idx46
    for target in [250, 46]:
        use = {}
        for i in range(target+1):
            nb = struct.unpack('<I', f.read(4))[0]
            data = np.frombuffer(f.read(nb*8), dtype=np.uint16).reshape(-1,4) if nb else np.zeros((0,4),dtype=np.uint16)
            if i == target:
                for (x,y,z,pi) in data:
                    nm = pals[int(pi)][0]
                    use[nm] = use.get(nm,0)+1
        mm = M['meta'][target]
        print(f'=== idx{target} ===', flush=True)
        print('n=', mm.get('n'), 'bbox=', mm.get('bbox_min'), mm.get('bbox_max'), flush=True)
        print('eye=', [round(v,1) for v in mm.get('eye',[])], 'look=', [round(v,1) for v in mm.get('look',[])], 'dist=', round(mm.get('dist',0),1), flush=True)
        print('top blocks:', flush=True)
        for nm,c in sorted(use.items(), key=lambda kv:-kv[1])[:12]:
            print(f'  {nm}: {c}', flush=True)
        print('total:', sum(use.values()), 'distinct:', len(use), flush=True)
