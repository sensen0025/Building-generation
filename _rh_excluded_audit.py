# -*- coding: utf-8 -*-
"""远程: 对比 valid_files_list 与 litematics 全量, 统计被淘汰 litematic 的原因."""
import sys, io, glob, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = '/home/linux/redstone_ai/dataset'
allf = set(os.path.basename(f) for f in glob.glob(f'{BASE}/litematics/*.litematic'))
valid = set(l.strip() for l in open(f'{BASE}/valid_files_list.txt', encoding='utf-8') if l.strip())
excluded = sorted(allf - valid)
print(f'all litematic: {len(allf)}  valid: {len(valid)}  excluded: {len(excluded)}', flush=True)
# 抽样 excluded 检查原因
import litemapy
fail = 0; small = 0; empty_region = 0; ok = 0
sizes = []
for fn in excluded[:400]:
    p = os.path.join(f'{BASE}/litematics', fn)
    try:
        s = litemapy.Schematic.load(p)
        if not s.regions:
            empty_region += 1
            continue
        r = list(s.regions.values())[0]
        W,H,L = abs(r.width), abs(r.height), abs(r.length)
        n = 0
        for x in range(W):
            for y in range(H):
                for z in range(L):
                    try:
                        b = r[x,y,z]
                    except IndexError:
                        continue
                    if b.id != 'minecraft:air':
                        n += 1
        sizes.append(n)
        if n < 5:
            small += 1
        else:
            ok += 1
    except Exception:
        fail += 1
print(f'sample {400}: ok>=5={ok} small<5={small} loadfail={fail} emptyregion={empty_region}', flush=True)
if sizes:
    import numpy as np
    a = np.array(sizes)
    print('excluded size: p50', int(np.median(a)), 'p90', int(np.percentile(a,90)), 'max', int(a.max()), flush=True)
print('END', flush=True)
