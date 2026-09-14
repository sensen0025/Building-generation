from mcfix.codec import RENAMES
# -*- coding: utf-8 -*-
"""Export compacted new samples to .litematic for condition rendering.
Reverses class id -> block name (canonical target) and packed state -> properties.
Output: /data/redstone_v4/litematics_new/<nidx>_<orig-stem>.litematic
"""
import os, sys, json, time, re
sys.path.insert(0, '/data/redstone_v4/code')
import numpy as np
from litemapy import Region, BlockState, Schematic

WORK = '/data/redstone_v4/work'
OUT = '/data/redstone_v4/litematics_new'
os.makedirs(OUT, exist_ok=True)

VOCAB_PATH = '/home/linux/redstone_ai/dataset/BLOCK_VOCAB_FULL.txt'
vocab = [l.strip() for l in open(VOCAB_PATH, encoding='utf-8') if l.strip()] + ['minecraft:other']
bm = np.load('/home/linux/redstone_ai/dataset/v2/block_map.npy')  # new128 -> old202
nid2name = {int(n): vocab[int(o)] for n, o in enumerate(bm)}
nid2name[63] = 'minecraft:stone'
nid2name[127] = 'minecraft:stone'  # other renders as stone

STATE_DIMS = [7, 3, 6, 16, 3]
FIELD_MULT = [1, 7, 21, 126, 2016]

OMAP = {1: 'north', 2: 'south', 3: 'east', 4: 'west', 5: 'up', 6: 'down'}
AXMAP = {3: 'x', 1: 'z', 5: 'y'}
SHAPES = {1: 'straight', 2: 'inner_left', 3: 'inner_right', 4: 'outer_left', 5: 'outer_right'}
NSEW = ['north', 'south', 'east', 'west']


def unpack_state(packed, name):
    from mcfix.legacy import legacy_properties
    return legacy_properties(name, packed)


def to_litematic(vox, stv, name, path):
    D, H, W = vox.shape
    reg = Region(0, 0, 0, D, H, W)
    n = 0
    for x in range(D):
        for y in range(H):
            for z in range(W):
                cid = int(vox[x, y, z])
                if cid == 0:
                    continue
                nm = nid2name.get(cid, 'minecraft:stone')
                base = RENAMES.get(nm.split('[')[0], nm.split('[')[0])
                props = unpack_state(int(stv[x, y, z]), base)
                try:
                    reg[x, y, z] = BlockState(base, **props)
                except Exception:
                    try:
                        reg[x, y, z] = BlockState(base)
                    except Exception:
                        reg[x, y, z] = BlockState('minecraft:stone')
                n += 1
    sch = Schematic(name=name, author='ai', regions={'main': reg}, mc_version=4189)
    sch.save(path)
    return n


def main():
    ss = np.load(f'{WORK}/ss64_new_c.npy', mmap_mode='r')
    st = np.load(f'{WORK}/ss64_state_new_c.npy', mmap_mode='r')
    m = json.load(open(f'{WORK}/ingest_manifest_ok.json'))
    t0 = time.time()
    n_done = 0
    for j, r in enumerate(m):
        outp = os.path.join(OUT, f'{j:05d}.litematic')
        if os.path.exists(outp):
            continue
        stem = re.sub(r'[^\w\-]', '_', r['file'].rsplit('.', 1)[0])[:40]
        vox = np.asarray(ss[j])
        stv = np.asarray(st[j])
        # trim to content bbox for tighter render
        nzc = np.argwhere(vox > 0)
        mn, mx = nzc.min(0), nzc.max(0)
        vox = vox[mn[0]:mx[0] + 1, mn[1]:mx[1] + 1, mn[2]:mx[2] + 1]
        stv = stv[mn[0]:mx[0] + 1, mn[1]:mx[1] + 1, mn[2]:mx[2] + 1]
        try:
            to_litematic(vox, stv, f'new_{j}', outp)
        except Exception as e:
            print('ERR', j, str(e)[:80], flush=True)
        n_done += 1
        if (j + 1) % 250 == 0:
            print(f'{j+1}/{len(m)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'DONE {n_done} exported in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
