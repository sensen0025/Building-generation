"""重建 voxelize 管线: litematic -> voxel96 (96^3 id + state 5字段).

逻辑还原自源服务器 cache_qualified.py (state编码) + voxel96 数据集格式.
- state uint16 = o + 7*(v + 3*(sh + 6*(hz + 16*ac)))
- 输出: voxel96_id.dat(uint8, 202类), voxel96_state.dat(uint16),
        solid96.npy(实心cell索引), solid96_off.npy, bb96.npy, voxel96_meta.npz(top200, n, total)
"""
import os, sys, glob, json, time
import numpy as np

BASE = '/home/linux/redstone_ai'
SRC = f'{BASE}/dataset/litematics'
OUT = f'{BASE}/dataset'
G = 96
MAX_CLASSES = 202  # 0=air, 1..200=top200, 201=other

import litemapy

def parse_state_props(p):
    """BlockState properties dict -> (o, v, sh, hz, ac)"""
    om = {"north": 1, "south": 2, "east": 3, "west": 4, "up": 5, "down": 6}
    o = om.get(p.get("facing", ""), 0)
    if o == 0 and p.get("axis", "") in ("x", "z", "y"):
        o = {"x": 3, "z": 1, "y": 5}[p["axis"]]
    hf = p.get("half", "")
    v = 1 if hf in ("top", "upper") else (2 if hf in ("bottom", "lower") else 0)
    tp = p.get("type", "")
    if v == 0 and tp == "top":
        v = 1
    elif v == 0 and tp == "bottom":
        v = 2
    sm = {"straight": 1, "inner_left": 2, "inner_right": 3, "outer_left": 4, "outer_right": 5}
    sh = sm.get(p.get("shape", ""), 0)
    hz = 0
    for i, d in enumerate(["north", "south", "east", "west"]):
        if str(p.get(d, "")).lower() == "true":
            hz |= (1 << i)
    ac = 0
    for k in ["lit", "powered", "open", "triggered", "extended"]:
        if str(p.get(k, "")).lower() == "true":
            ac = 2
            break
        if str(p.get(k, "")).lower() == "false":
            ac = 1
            break
    return o, v, sh, hz, ac

def pack_state(o, v, sh, hz, ac):
    return o + 7 * (v + 3 * (sh + 6 * (hz + 16 * ac)))

def extract_litematic(path):
    """返回 (cells, w, h, l): cells = [(x,y,z, state_uint16, block_id_str)]"""
    try:
        s = litemapy.Schematic.load(path)
        r = list(s.regions.values())[0]
        W, H, L = abs(r.width), abs(r.height), abs(r.length)
    except Exception as e:
        return None
    cells = []
    for x in range(W):
        for y in range(H):
            for z in range(L):
                try:
                    b = r[x, y, z]
                except IndexError:
                    continue
                if b.id == "minecraft:air":
                    continue
                try:
                    p = dict(b.properties()) if callable(b.properties) else {}
                except Exception:
                    p = {}
                o, v, sh, hz, ac = parse_state_props(p)
                cells.append((x, y, z, pack_state(o, v, sh, hz, ac), b.id))
    if len(cells) < 5:
        return None
    return cells

# ---------- pass 1: 扫描, 统计方块频率 ----------
print('pass 1: scanning litematics...', flush=True)
files = sorted(glob.glob(os.path.join(SRC, '*.litematic')))
print(f'{len(files)} litematic files', flush=True)
freq = {}
valid_files = []
t0 = time.time()
for i, f in enumerate(files):
    cells = extract_litematic(f)
    if cells is None:
        continue
    valid_files.append((f, cells))
    for c in cells:
        freq[c[4]] = freq.get(c[4], 0) + 1
    if (i + 1) % 1000 == 0:
        print(f'  {i+1}/{len(files)} valid={len(valid_files)} ({time.time()-t0:.0f}s)', flush=True)

print(f'valid files: {len(valid_files)}, distinct blocks: {len(freq)}', flush=True)

# ---------- vocab: top200 ----------
top = sorted(freq.items(), key=lambda kv: -kv[1])[:200]
BLOCK_LIST = ['minecraft:air'] + [b for b, _ in top]
TYPE2ID = {n: i for i, n in enumerate(BLOCK_LIST)}  # 1..200
top200 = np.zeros(200, dtype=np.int64)
# 保存 vocab 文件 (映射: 类1..200 -> BLOCK_LIST 索引)
vocab_names = BLOCK_LIST
print('top blocks:', [(b, c) for b, c in top[:10]], flush=True)

with open(f'{OUT}/BLOCK_VOCAB_FULL.txt', 'w', encoding='utf-8') as fp:
    fp.write('\n'.join(BLOCK_LIST) + '\n')

# ---------- pass 2: 构建 96^3 ----------
print('pass 2: voxelizing...', flush=True)
N = len(valid_files)
vox_id = np.lib.format.open_memmap(f'{OUT}/voxel96_id.dat', mode='w+', dtype=np.uint8, shape=(N, G, G, G))
vox_st = np.lib.format.open_memmap(f'{OUT}/voxel96_state.dat', mode='w+', dtype=np.uint16, shape=(N, G, G, G))
solid_cat = []
offsets = np.zeros(N + 1, dtype=np.int64)
bbs = np.zeros((N, 6), dtype=np.int32)

t0 = time.time()
for i, (f, cells) in enumerate(valid_files):
    xs = np.array([c[0] for c in cells]); ys = np.array([c[1] for c in cells]); zs = np.array([c[2] for c in cells])
    x0, y0, z0 = xs.min(), ys.min(), zs.min()
    x1, y1, z1 = xs.max(), ys.max(), zs.max()
    bbs[i] = [x0, y0, z0, x1, y1, z1]
    # 居中 pad 到 96^3
    cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0; cz = (z0 + z1) / 2.0
    ox = int(round(G / 2 - cx)); oy = int(round(G / 2 - cy)); oz = int(round(G / 2 - cz))
    vol = np.zeros((G, G, G), dtype=np.int16)  # -1 air, 0..199 class, 200 other
    st_vol = np.zeros((G, G, G), dtype=np.uint16)
    for (x, y, z, st, bn) in cells:
        nx, ny, nz = x + ox, y + oy, z + oz
        if 0 <= nx < G and 0 <= ny < G and 0 <= nz < G:
            cls = TYPE2ID.get(bn, 201)  # 201 = other (存为 201, 映射到 uint8 ok)
            vol[nx, ny, nz] = cls
            st_vol[nx, ny, nz] = st
    # 保存: id: 0=air, 1..200, 201=other
    vid = np.where(vol < 0, 0, vol).astype(np.uint8)
    vox_id[i] = vid
    vox_st[i] = st_vol
    solid = np.argwhere(vid > 0)
    if len(solid):
        flat = solid[:, 0] * (G * G) + solid[:, 1] * G + solid[:, 2]
        offsets[i + 1] = offsets[i] + len(flat)
        solid_cat.append(flat)
    else:
        offsets[i + 1] = offsets[i]
        bbs[i] = [-1, -1, -1, -1, -1, -1]
    if (i + 1) % 2000 == 0:
        print(f'  {i+1}/{N} ({time.time()-t0:.0f}s)', flush=True)

vox_id.flush(); vox_st.flush()
if solid_cat:
    solid_all = np.concatenate(solid_cat).astype(np.int64)
    np.save(f'{OUT}/solid96.npy', solid_all)
np.save(f'{OUT}/solid96_off.npy', offsets)
np.save(f'{OUT}/bb96.npy', bbs)
np.savez(f'{OUT}/voxel96_meta.npz', top200=top200, n=N, total=N)
print(f'DONE: N={N} total={N}', flush=True)
print(f'files: voxel96_id.dat {os.path.getsize(f"{OUT}/voxel96_id.dat")/1e9:.2f}GB, state {os.path.getsize(f"{OUT}/voxel96_state.dat")/1e9:.2f}GB', flush=True)
