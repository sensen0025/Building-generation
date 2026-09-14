# -*- coding: utf-8 -*-
"""扫描源 litematic, 按真实属性判定每个方块类的 o 字段语义 (facing 还是 axis).

用于 state 旋转重映射: facing 类需 4-循环 (1->3->2->4->1), axis 类只需 1<->3.
输出: /data/redstone_v4/work/block_props_scan.json
"""
import glob
import json
import collections

import litemapy

SRC = '/home/linux/redstone_ai/dataset/litematics/*.litematic'
VOCAB = '/home/linux/redstone_ai/dataset/BLOCK_VOCAB_FULL.txt'
OUT = '/data/redstone_v4/work/block_props_scan.json'

vocab = set(l.strip() for l in open(VOCAB, encoding='utf-8') if l.strip())
import numpy as np
_bm = np.load('/home/linux/redstone_ai/dataset/v2/block_map.npy')
_vl = [l.strip() for l in open(VOCAB, encoding='utf-8') if l.strip()] + ['minecraft:other']
targets = set(_vl[int(i)] for i in _bm)
print(f'vocab={len(vocab)} target_classes={len(targets)}', flush=True)

files = sorted(glob.glob(SRC))
print(f'{len(files)} litematics', flush=True)

seen = collections.defaultdict(set)     # name -> set of property keys
n = 0
MAXF = 250
for f in files[:MAXF]:
    try:
        s = litemapy.Schematic.load(f)
    except Exception:
        continue
    n += 1
    for r in s.regions.values():
        pal = getattr(r, 'palette', None)
        if pal is None:
            for attr in ('_Region__palette', '_palette'):
                pal = getattr(r, attr, None)
                if pal is not None:
                    break
        if pal is None:
            continue
        for bs in pal:
            if bs.id in vocab:
                try:
                    seen[bs.id].update(dict(bs.properties()).keys())
                except Exception:
                    pass
    if n % 5 == 0:
        res_now = sum(1 for t in targets if t in seen)
        print(f'  scanned {n} files, target names seen {res_now}/{len(targets)}', flush=True)
    if all(t in seen for t in targets):
        print(f'  所有目标类已出现, 提前结束于第 {n} 个文件', flush=True)
        break

res = {}
for name, keys in seen.items():
    if 'axis' in keys:
        res[name] = 'axis'
    elif 'facing' in keys:
        res[name] = 'facing'
    else:
        res[name] = 'none'
counts = collections.Counter(res.values())
print(f'scanned {n} files; resolved {len(res)} vocab names: {dict(counts)}', flush=True)
print('axis 类:', sorted(k for k, v in res.items() if v == 'axis'), flush=True)
json.dump({'scanned_files': n, 'result': res}, open(OUT, 'w'), indent=1)
print('WROTE ' + OUT, flush=True)
