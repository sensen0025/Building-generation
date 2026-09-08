# -*- coding: utf-8 -*-
"""远程: 统计 .schem/.schematic 的可用性: litemapy 能否加载 + cells 数 >= 5.
判断 745 个 schem 能否补充数据集."""
import sys, io, glob, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import litemapy

BASE = '/home/linux/redstone_ai/dataset/litematics'
files = sorted(glob.glob(f'{BASE}/*.schem')) + sorted(glob.glob(f'{BASE}/*.schematic'))
print('schem/schematic total:', len(files), flush=True)

ok = 0; fail = 0; small = 0
sizes = []
t0 = time.time()
for i, f in enumerate(files):
    try:
        s = litemapy.Schematic.load(f)
        r = list(s.regions.values())[0]
        W, H, L = abs(r.width), abs(r.height), abs(r.length)
        n = 0
        for x in range(W):
            for y in range(H):
                for z in range(L):
                    try:
                        b = r[x, y, z]
                    except IndexError:
                        continue
                    if b.id != 'minecraft:air':
                        n += 1
        sizes.append(n)
        if n >= 5:
            ok += 1
        else:
            small += 1
    except Exception as e:
        fail += 1
    if (i+1) % 100 == 0:
        print(f'  scan {i+1}/{len(files)} ok={ok} fail={fail} small={small} el={time.time()-t0:.0f}s', flush=True)
print(f'DONE ok>=5cells: {ok}  small<5: {small}  load_fail: {fail}', flush=True)
import numpy as np
if sizes:
    a = np.array(sizes)
    print('size dist: p50', int(np.median(a)), 'p90', int(np.percentile(a,90)), 'max', int(a.max()), flush=True)
    for th in [200, 1000, 5000]:
        print(f'  >= {th} cells: {(a>=th).sum()}', flush=True)
print('END', flush=True)
