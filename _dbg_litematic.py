"""定位 litemapy DiscriminationError: 找出哪个 block/props 触发."""
import os, sys
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
import numpy as np
from litemapy import Region, BlockState, Schematic

BASE = '/home/linux/redstone_ai'
BLOCK_MAP = f'{BASE}/dataset/v2/block_map.npy'
VOCAB = f'{BASE}/dataset/BLOCK_VOCAB_FULL.txt'
block_map = np.load(BLOCK_MAP)
vocab = [l.strip() for l in open(VOCAB, encoding='utf-8') if l.strip()]
def block_name(nid):
    oi = int(block_map[nid])
    return vocab[oi] if 0 <= oi < len(vocab) else 'minecraft:stone'

# 复现: 用一个简单 props 测试
test_cases = [
    ('minecraft:stone', {}),
    ('minecraft:oak_stairs', {'facing': 'north', 'half': 'bottom'}),
    ('minecraft:oak_log', {'axis': 'x'}),
]
for nm, props in test_cases:
    try:
        bs = BlockState(nm, properties=props)
        print(f'OK  {nm} {props}')
    except Exception as e:
        print(f'ERR {nm} {props}: {type(e).__name__}: {e}')

# 扫描 idx0 GT 全部 cell, 找崩溃 props 组合
ST64 = f'{BASE}/dataset/v2/ss64_state.npy'
SS64 = f'{BASE}/dataset/v2/ss64.npy'
FIELD_MULT = [1, 7, 21, 126, 2016]
STATE_DIMS = [7, 3, 6, 16, 3]

def state_to_props(name, fields):
    o, v, sh, hz, ac = [int(f) for f in fields]
    props = {}
    facing = {1: 'north', 2: 'south', 3: 'east', 4: 'west'}
    if 'stair' in name:
        if o in facing: props['facing'] = facing[o]
        if v == 1: props['half'] = 'top'
        elif v == 2: props['half'] = 'bottom'
        if sh: props['shape'] = {1:'straight',2:'inner_left',3:'inner_right',4:'outer_left',5:'outer_right'}[sh]
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
    elif 'chest' in name:
        if o in facing: props['facing'] = facing[o]
    elif 'torch' in name or 'lever' in name or 'button' in name:
        if o in facing: props['facing'] = facing[o]
    elif 'door' in name:
        if o in facing: props['facing'] = facing[o]
        if v == 1: props['half'] = 'upper'
        elif v == 2: props['half'] = 'lower'
        if ac: props['open'] = 'true' if ac == 2 else 'false'
    elif 'ladder' in name or 'end_rod' in name or 'chain' in name or 'lightning_rod' in name:
        if o in facing: props['facing'] = facing[o]
        if o == 5: props['facing'] = 'up'
        if o == 6: props['facing'] = 'down'
    return {k: str(v) for k, v in props.items()}

ss = np.load(SS64, mmap_mode='r')
st = np.load(ST64, mmap_mode='r')
gt = np.asarray(ss[0]).astype(np.int16)
gt_st = np.asarray(st[0]).astype(np.int64)
gt_fields = [(gt_st // FIELD_MULT[f]) % STATE_DIMS[f] for f in range(5)]

bad = 0
for x in range(64):
    for y in range(64):
        for z in range(64):
            c = int(gt[x, y, z])
            if c <= 0: continue
            nm = block_name(c)
            if nm == 'minecraft:air': nm = 'minecraft:stone'
            fields = [f[x, y, z] for f in gt_fields]
            props = state_to_props(nm, fields)
            for k, val in props.items():
                if not isinstance(k, str) or not isinstance(val, str):
                    print(f'NONSTR cell=({x},{y},{z}) class={c} name={nm} props={props} raw={fields}')
                    bad += 1
                    continue
            try:
                BlockState(nm, properties=props)
            except Exception as e:
                print(f'ERR cell=({x},{y},{z}) class={c} name={nm} props={props} raw_fields={[int(f) for f in fields]}: {type(e).__name__}: {e}')
                bad += 1
print('bad cells:', bad)
print('DONE')
