import os, sys
import numpy as np
from litemapy import Region, BlockState, Schematic

BASE = '/home/linux/redstone_ai'
SS64 = f'{BASE}/dataset/v2/ss64.npy'
SS64_ST = f'{BASE}/dataset/v2/ss64_state.npy'
BLOCK_MAP = f'{BASE}/dataset/v2/block_map.npy'
VOCAB = f'{BASE}/dataset/BLOCK_VOCAB_FULL.txt'
OUT = f'{BASE}/gen_outputs'

STATE_DIMS = [7, 3, 6, 16, 3]
FIELD_MULT = [1, 7, 21, 126, 2016]

ss = np.load(SS64, mmap_mode='r')
st = np.load(SS64_ST, mmap_mode='r')
bmap = np.load(BLOCK_MAP)
vocab = [l.strip() for l in open(VOCAB, encoding='utf-8') if l.strip()]

def block_name(nid):
    oi = int(bmap[nid])
    return vocab[oi] if 0 <= oi < len(vocab) else 'minecraft:stone'

def state_to_props(name, fields):
    o, v, sh, hz, ac = [int(f) for f in fields]
    props = {}
    facing = {1: 'north', 2: 'south', 3: 'east', 4: 'west'}
    if 'stair' in name:
        if o in facing: props['facing'] = facing[o]
        if v == 1: props['half'] = 'top'
        elif v == 2: props['half'] = 'bottom'
        if sh: props['shape'] = {1:'straight',2:'inner_left',3:'inner_right',4:'outer_left',5:'outer_right'}.get(sh, 'straight')
    elif 'slab' in name:
        if v == 1: props['type'] = 'top'
        elif v == 2: props['type'] = 'bottom'
    elif 'trapdoor' in name:
        if o in facing: props['facing'] = facing[o]
        if v == 1: props['half'] = 'top'
        elif v == 2: props['half'] = 'bottom'
        if ac: props['open'] = 'true' if ac == 2 else 'false'
    elif any(k in name for k in ['log', '_wood', '_stem', 'hay', 'basalt', 'mushroom', 'froglight', 'quartz_pillar', 'purpur_pillar']):
        if o: props['axis'] = {3: 'x', 1: 'z', 5: 'y'}.get(o, 'y')
    elif any(k in name for k in ['fence', 'glass_pane', 'bars']):
        for i, d in enumerate(['north', 'south', 'east', 'west']):
            if hz & (1 << i): props[d] = 'true'
    elif 'door' in name:
        if o in facing: props['facing'] = facing[o]
        if v == 1: props['half'] = 'upper'
        elif v == 2: props['half'] = 'lower'
        if ac: props['open'] = 'true' if ac == 2 else 'false'
    return props

def export_building(idx, filename):
    vox = np.asarray(ss[idx]).astype(np.int16)
    stv = np.asarray(st[idx]).astype(np.int64)
    fields = [(stv // FIELD_MULT[f]) % STATE_DIMS[f] for f in range(5)]

    nz = np.argwhere(vox > 0)
    if len(nz) == 0:
        print(f'idx {idx} has 0 blocks')
        return
    min_x, min_y, min_z = nz.min(0)
    max_x, max_y, max_z = nz.max(0)
    W = int(max_x - min_x + 1)
    H = int(max_y - min_y + 1)
    L = int(max_z - min_z + 1)

    print(f'idx {idx}: bbox=[{W},{H},{L}], non-zero blocks={len(nz)}')
    reg = Region(0, 0, 0, W, H, L)
    block_counts = {}
    for x in range(W):
        for y in range(H):
            for z in range(L):
                gx, gy, gz = min_x + x, min_y + y, min_z + z
                c = int(vox[gx, gy, gz])
                if c > 0:
                    nm = block_name(c)
                    if nm == 'minecraft:air':
                        nm = 'minecraft:stone'
                    props = state_to_props(nm, [f[gx, gy, gz] for f in fields])
                    try:
                        reg.setblock(x, y, z, BlockState(nm, **props))
                    except:
                        reg.setblock(x, y, z, BlockState(nm))
                    block_counts[nm] = block_counts.get(nm, 0) + 1

    sch = Schematic(name=f'Building_idx_{idx}', author='redstone_ai',
                    description=f'Real Minecraft Structure {W}x{H}x{L} with {len(nz)} blocks',
                    regions={'main': reg})
    sch.mc_version = 3465  # Minecraft 1.20.1 标准 DataVersion
    out_path = f'{OUT}/{filename}'
    sch.save(out_path)
    print(f'Saved: {out_path} ({os.path.getsize(out_path)} bytes)')
    print('Top 5 blocks:', sorted(block_counts.items(), key=lambda x: -x[1])[:5])

export_building(4, 'real_building_house_idx4.litematic')
export_building(5, 'real_building_castle_idx5.litematic')
print('DONE_ALL')
