# -*- coding: utf-8 -*-
"""生成 MC 渲染器放置数据:
  /home/linux/redstone_ai/render_mc/builds.rhb    二进制: 每个建筑的 (x,y,z,palIdx)
  /home/linux/redstone_ai/render_mc/builds_meta.json 每建筑统计(bbox/center/span/dist)
  调色板 = 唯一 (MC方块名, props字符串); other/越界 -> minecraft:stone; air名 -> stone
轴序: ss64 为 (x,y,z), y 向上, 与 MC 世界坐标一致(1体素=1格).
"""
import os, sys, json, struct
import numpy as np

BASE = '/home/linux/redstone_ai/dataset'
OUT = '/home/linux/redstone_ai/render_mc'
os.makedirs(OUT, exist_ok=True)

# ---- 相机几何(与渲染端一致, 显式中心/距离计算) ----
YAW_DEG = 45.0   # 相机绕 +y 的方位角(从 look 点看: 相机位于 NE 上空)
PITCH_DEG = 25.0 # 俯仰(高于水平面)
FOV_DEG = 70.0   # MC fov 0.0 默认值 = 70 (竖直)
TAN_HALF = np.tan(np.deg2rad(FOV_DEG) / 2)
MARGIN = 1.35

def camera_geometry(spans, mass_center):
    """显式中心(占用质量中心 look) + 距离随建筑尺寸:
    dist = max(1.10*max(spans)+2.5, 1.15*diag)  -> 保证相机在包围球外(空腔/异形不贴墙), 同时大轴占画面~55%.
    相机方向固定 yaw=45°, pitch=25°."""
    smax = float(np.max(spans))
    diag = float(np.linalg.norm(spans))
    dist = max(1.10 * smax + 2.5, 1.15 * diag)
    yaw = np.deg2rad(YAW_DEG); pitch = np.deg2rad(PITCH_DEG)
    dx = np.cos(yaw) * np.cos(pitch)
    dz = np.sin(yaw) * np.cos(pitch)
    dy = np.sin(pitch)
    look = np.array(mass_center, dtype=float)
    eye = look + dist * np.array([dx, dy, dz])
    return eye.tolist(), look.tolist(), float(dist)

ss = np.load(f'{BASE}/v2/ss64.npy', mmap_mode='r')
st = np.load(f'{BASE}/v2/ss64_state.npy', mmap_mode='r')
bm = np.load(f'{BASE}/v2/block_map.npy')
vocab = [l.strip() for l in open(f'{BASE}/BLOCK_VOCAB_FULL.txt', encoding='utf-8') if l.strip()]
N = ss.shape[0]
FIELD_MULT = [1, 7, 21, 126, 2016]
STATE_DIMS = [7, 3, 6, 16, 3]

def block_name(nid):
    oi = int(bm[nid])
    if 0 <= oi < len(vocab):
        return vocab[oi]
    return None  # other/越界 -> stone

def state_to_props(name, fields):
    o, v, sh, hz, ac = [int(f) for f in fields]
    props = {}
    facing = {1: 'north', 2: 'south', 3: 'east', 4: 'west'}
    if 'stair' in name:
        if o in facing:
            props['facing'] = facing[o]
        if v == 1:
            props['half'] = 'top'
        elif v == 2:
            props['half'] = 'bottom'
        if sh:
            props['shape'] = {1: 'straight', 2: 'inner_left', 3: 'inner_right', 4: 'outer_left', 5: 'outer_right'}[sh]
    elif 'slab' in name:
        if v == 1:
            props['type'] = 'top'
        elif v == 2:
            props['type'] = 'bottom'
    elif 'trapdoor' in name:
        if o in facing:
            props['facing'] = facing[o]
        if v == 1:
            props['half'] = 'top'
        elif v == 2:
            props['half'] = 'bottom'
        if ac:
            props['open'] = 'true' if ac == 2 else 'false'
    elif any(k in name for k in ['log', '_wood', '_stem', 'hay', 'basalt', 'mushroom', 'froglight', 'quartz_pillar', 'purpur_pillar']):
        if o:
            props['axis'] = {3: 'x', 1: 'z', 5: 'y'}.get(o, 'y')
    elif any(k in name for k in ['fence', 'glass_pane', 'bars']):
        for i, d in enumerate(['north', 'south', 'east', 'west']):
            if hz & (1 << i):
                props[d] = 'true'
    elif 'chest' in name:
        if o in facing:
            props['facing'] = facing[o]
    elif any(k in name for k in ['torch', 'lever', 'button']):
        if o in facing:
            props['facing'] = facing[o]
    elif 'door' in name:
        if o in facing:
            props['facing'] = facing[o]
        if v == 1:
            props['half'] = 'upper'
        elif v == 2:
            props['half'] = 'lower'
        if ac:
            props['open'] = 'true' if ac == 2 else 'false'
    elif any(k in name for k in ['ladder', 'end_rod', 'chain', 'lightning_rod']):
        if o == 5:
            props['facing'] = 'up'
        elif o == 6:
            props['facing'] = 'down'
        elif o in facing:
            props['facing'] = facing[o]
    return props

def props_str(props):
    if not props:
        return ''
    return ','.join(f'{k}={v}' for k, v in sorted(props.items()))

# pass A: 统计唯一 palette
palette = {}          # (name, props_str) -> idx
pal_idx_cache = {}
idx2pal = {}          # (class, state) -> palette idx  状态到属性是确定性的
def pal_for(nid, st_u16):
    key = (int(nid), int(st_u16))
    if key in idx2pal:
        return idx2pal[key]
    name = block_name(nid)
    if name is None or name == 'minecraft:air':
        name = 'minecraft:stone'
    if st_u16:
        fields = [(st_u16 // FIELD_MULT[f]) % STATE_DIMS[f] for f in range(5)]
        props = state_to_props(name, fields)
    else:
        props = {}
    ps = props_str(props)
    pkey = (name, ps)
    if pkey not in palette:
        palette[pkey] = len(palette)
    idx2pal[key] = palette[pkey]
    return palette[pkey]

# pass B: 写出
meta = []
n_zero = 0
total_blocks = 0
with open(f'{OUT}/builds.rhb', 'wb') as f:
    f.write(b'RHB1')
    f.write(struct.pack('<I', len(palette)))  # palette count 占位, 后面回填
    # 先收集再写 -> 由于需要 palette 字符串, 先写所有块到临时, 再组合
    # 简化: 两次扫描 ss(5339x64^3 两次 ~ 700MB*2 读取, memmap 可接受)
    pass

# 由于 palette 未知长度, 先全量构建 palette(需要读全部)
print('pass1: 统计 palette...', flush=True)
for i in range(N):
    v = np.asarray(ss[i])
    sarr = np.asarray(st[i])
    nz = np.argwhere(v > 0)
    for x, y, z in nz:
        cid = int(v[x, y, z])
        stv = int(sarr[x, y, z])
        pal_for(cid, stv)
print(f'palette size: {len(palette)}', flush=True)

# 写入 palette 字符串
pal_list = [None] * len(palette)
for (name, ps), idx in palette.items():
    pal_list[idx] = (name, ps)

with open(f'{OUT}/builds.rhb', 'wb') as f:
    f.write(b'RHB1')
    f.write(struct.pack('<I', len(pal_list)))
    for name, ps in pal_list:
        nb = name.encode('utf-8')
        pb = ps.encode('utf-8')
        f.write(struct.pack('<H', len(nb)))
        f.write(nb)
        f.write(struct.pack('<H', len(pb)))
        f.write(pb)
    f.write(struct.pack('<I', N))
    for i in range(N):
        v = np.asarray(ss[i])
        sarr = np.asarray(st[i])
        nz = np.argwhere(v > 0)
        f.write(struct.pack('<I', len(nz)))
        for x, y, z in nz:
            cid = int(v[x, y, z])
            stv = int(sarr[x, y, z])
            pi = pal_for(cid, stv)
            f.write(struct.pack('<HHHH', int(x), int(y), int(z), pi))
        total_blocks += len(nz)
        # meta
        if len(nz) == 0:
            n_zero += 1
            meta.append({'idx': i, 'n': 0, 'bbox_min': None, 'bbox_max': None, 'center': None, 'dist': 0.0})
        else:
            mn = nz.min(0); mx = nz.max(0)
            spans = (mx - mn + 1).astype(float)
            mass = nz.mean(0)  # 质量中心(占用体素均值)
            eye, look, dist = camera_geometry(spans, mass)
            meta.append({
                'idx': i, 'n': int(len(nz)),
                'bbox_min': [int(mn[0]), int(mn[1]), int(mn[2])],
                'bbox_max': [int(mx[0]), int(mx[1]), int(mx[2])],
                'center': [float(mass[0]), float(mass[1]), float(mass[2])],
                'spans': [float(spans[0]), float(spans[1]), float(spans[2])],
                'eye': eye, 'look': look, 'dist': float(dist),
            })
    print(f'wrote builds.rhb: N={N} blocks={total_blocks} zero_occ={n_zero}', flush=True)

json.dump({'N': N, 'palette_size': len(pal_list), 'meta': meta}, open(f'{OUT}/builds_meta.json', 'w'))
print('meta written', flush=True)

# 抽样打印调色板
print('palette sample:', [(k[0], k[1]) for k in list(palette.keys())[:10]], flush=True)
print('DONE', flush=True)
