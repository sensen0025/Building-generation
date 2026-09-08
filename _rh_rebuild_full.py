# -*- coding: utf-8 -*-
"""重建无压缩渲染数据: litematic -> 原始方块名+props, 精确复现 ss64 的裁剪几何.
格式与 _rh_prep_builds.py 相同 (RHB1), 但 palette 用真实方块名(含 old 1..200 原本被并进 other 的),
仅真·other(不在 litematic 源内的未知)保留. 用法: 等 rescan 完成生成 valid_files_list.txt 后运行.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os, json, struct, time
import numpy as np
import litemapy

BASE = '/home/linux/redstone_ai/dataset'
OUT = '/home/linux/redstone_ai/render_mc'
LIST = f'{BASE}/valid_files_list.txt'
G = 96   # voxel96 尺寸
C = 64   # ss64 尺寸
OFF = (G - C) // 2  # 16

# 旧版/别名方块名 -> 1.21.11 注册名 (仅当 litematic 用了旧名)
NAME_ALIAS = {
    'minecraft:oak_plank': 'minecraft:oak_planks',
    'minecraft:spruce_plank': 'minecraft:spruce_planks',
    'minecraft:birch_plank': 'minecraft:birch_planks',
    'minecraft:jungle_plank': 'minecraft:jungle_planks',
    'minecraft:acacia_plank': 'minecraft:acacia_planks',
    'minecraft:dark_oak_plank': 'minecraft:dark_oak_planks',
    'minecraft:cherry_plank': 'minecraft:cherry_planks',
    'minecraft:grass': 'minecraft:grass_block',
    'minecraft:white_stained_clay': 'minecraft:white_terracotta',
    'minecraft:orange_stained_clay': 'minecraft:orange_terracotta',
    'minecraft:magenta_stained_clay': 'minecraft:magenta_terracotta',
    'minecraft:light_blue_stained_clay': 'minecraft:light_blue_terracotta',
    'minecraft:yellow_stained_clay': 'minecraft:yellow_terracotta',
    'minecraft:lime_stained_clay': 'minecraft:lime_terracotta',
    'minecraft:pink_stained_clay': 'minecraft:pink_terracotta',
    'minecraft:gray_stained_clay': 'minecraft:gray_terracotta',
    'minecraft:light_gray_stained_clay': 'minecraft:light_gray_terracotta',
    'minecraft:cyan_stained_clay': 'minecraft:cyan_terracotta',
    'minecraft:purple_stained_clay': 'minecraft:purple_terracotta',
    'minecraft:blue_stained_clay': 'minecraft:blue_terracotta',
    'minecraft:brown_stained_clay': 'minecraft:brown_terracotta',
    'minecraft:green_stained_clay': 'minecraft:green_terracotta',
    'minecraft:red_stained_clay': 'minecraft:red_terracotta',
    'minecraft:black_stained_clay': 'minecraft:black_terracotta',
    'minecraft:double_stone_slab': 'minecraft:stone_slab',  # 双叠 -> 单叠近似(渲染视觉)
}

def norm_name(nm):
    if nm in NAME_ALIAS:
        return NAME_ALIAS[nm]
    return nm

def read_cells(path):
    """返回 [(x,y,z, block_id, props_dict)] 全部非air."""
    s = litemapy.Schematic.load(path)
    r = list(s.regions.values())[0]
    W, H, L = abs(r.width), abs(r.height), abs(r.length)
    cells = []
    for x in range(W):
        for y in range(H):
            for z in range(L):
                try:
                    b = r[x, y, z]
                except IndexError:
                    continue
                if b.id == 'minecraft:air':
                    continue
                try:
                    p = dict(b.properties()) if callable(b.properties) else {}
                except Exception:
                    p = {}
                cells.append((x, y, z, b.id, p))
    return cells

def main():
    names = [l.strip() for l in open(LIST, encoding='utf-8') if l.strip()]
    N = len(names)
    print(f'valid files: {N}', flush=True)
    meta = []
    palette = {}       # (name, props_str) -> idx
    idx2pal = {}
    pal_list = []

    # 与 ss64 对齐检查: bb96
    bb96 = np.load(f'{BASE}/bb96.npy')
    print('bb96 N =', bb96.shape[0], flush=True)

    t0 = time.time()
    blocks_total = 0
    # pass1 统计 palette
    for i, fn in enumerate(names):
        path = os.path.join(f'{BASE}/litematics', fn)
        try:
            cells = read_cells(path)
        except Exception as e:
            print(f'  idx{i} load fail {fn}: {e}', flush=True)
            meta.append({'idx': i, 'n': 0})
            continue
        # 复现 voxelize 平移: 基于 bb (cells min/max)
        if not cells:
            meta.append({'idx': i, 'n': 0})
            continue
        xs = [c[0] for c in cells]; ys = [c[1] for c in cells]; zs = [c[2] for c in cells]
        x0,y0,z0 = min(xs),min(ys),min(zs); x1,y1,z1 = max(xs),max(ys),max(zs)
        cx=(x0+x1)/2; cy=(y0+y1)/2; cz=(z0+z1)/2
        ox=int(round(G/2-cx)); oy=int(round(G/2-cy)); oz=int(round(G/2-cz))
        # 只统计落在 64^3 (voxel96 OFF..OFF+64) 内的 cell
        for (x,y,z,nm,p) in cells:
            nx, ny, nz = x+ox-OFF, y+oy-OFF, z+oz-OFF
            if not (0 <= nx < C and 0 <= ny < C and 0 <= nz < C):
                continue
            nm2 = norm_name(nm)
            ps = ','.join(f'{k}={v}' for k,v in sorted(p.items()))
            pk = (nm2, ps)
            if pk not in palette:
                palette[pk] = len(palette)
        if (i+1) % 500 == 0:
            print(f'  pass1 {i+1}/{N} palette={len(palette)} elapsed={time.time()-t0:.0f}s', flush=True)
    print(f'pass1 done palette={len(palette)} elapsed={time.time()-t0:.0f}s', flush=True)

    pal_list = [None]*len(palette)
    for (nm, ps), idx in palette.items():
        pal_list[idx] = (nm, ps)

    # pass2 写文件
    with open(f'{OUT}/builds.rhb.full', 'wb') as f:
        f.write(b'RHB1')
        f.write(struct.pack('<I', len(pal_list)))
        for nm, ps in pal_list:
            nb = nm.encode('utf-8'); pb = ps.encode('utf-8')
            f.write(struct.pack('<H', len(nb))); f.write(nb)
            f.write(struct.pack('<H', len(pb))); f.write(pb)
        f.write(struct.pack('<I', N))
        for i, fn in enumerate(names):
            path = os.path.join(f'{BASE}/litematics', fn)
            try:
                cells = read_cells(path)
            except Exception:
                f.write(struct.pack('<I', 0))
                meta.append({'idx': i, 'n': 0})
                continue
            out_cells = []
            if cells:
                xs=[c[0] for c in cells]; ys=[c[1] for c in cells]; zs=[c[2] for c in cells]
                x0,y0,z0=min(xs),min(ys),min(zs); x1,y1,z1=max(xs),max(ys),max(zs)
                cx=(x0+x1)/2; cy=(y0+y1)/2; cz=(z0+z1)/2
                ox=int(round(G/2-cx)); oy=int(round(G/2-cy)); oz=int(round(G/2-cz))
                for (x,y,z,nm,p) in cells:
                    nx,ny,nz = x+ox-OFF, y+oy-OFF, z+oz-OFF
                    if not (0<=nx<C and 0<=ny<C and 0<=nz<C):
                        continue
                    nm2 = norm_name(nm)
                    ps = ','.join(f'{k}={v}' for k,v in sorted(p.items()))
                    pk = (nm2, ps)
                    out_cells.append((nx,ny,nz, palette[pk]))
            f.write(struct.pack('<I', len(out_cells)))
            for (x,y,z,pi) in out_cells:
                f.write(struct.pack('<HHHH', int(x),int(y),int(z),int(pi)))
            blocks_total += len(out_cells)
            if len(out_cells) == 0:
                meta.append({'idx': i, 'n': 0})
            else:
                mn = np.array([min(c[0] for c in out_cells), min(c[1] for c in out_cells), min(c[2] for c in out_cells)])
                mx = np.array([max(c[0] for c in out_cells), max(c[1] for c in out_cells), max(c[2] for c in out_cells)])
                spans = (mx-mn+1).astype(float)
                # 质量中心(与 ss64 版本一致: 用 voxel96 坐标的 cell?这里直接用渲染坐标算)
                mass = np.mean([[c[0],c[1],c[2]] for c in out_cells], axis=0)
                from math import atan2, sin, cos, tan, pi as _pi
                YAW=45.0; PITCH=25.0; FOV=70.0
                smax=float(np.max(spans)); diag=float(np.linalg.norm(spans))
                dist=max(1.10*smax+2.5, 1.15*diag)
                yaw=_pi/180*YAW; pitch=_pi/180*PITCH
                dx=cos(yaw)*cos(pitch); dz=sin(yaw)*cos(pitch); dy=sin(pitch)
                eye=(mass+dist*np.array([dx,dy,dz])).tolist()
                meta.append({'idx': i, 'n': int(len(out_cells)),
                             'bbox_min': mn.tolist(), 'bbox_max': mx.tolist(),
                             'center': mass.tolist(), 'spans': spans.tolist(),
                             'eye': eye, 'look': mass.tolist(), 'dist': float(dist)})
            if (i+1) % 500 == 0:
                print(f'  pass2 {i+1}/{N} blocks={blocks_total} elapsed={time.time()-t0:.0f}s', flush=True)
    print(f'wrote builds.rhb.full N={N} blocks={blocks_total} palette={len(pal_list)} elapsed={time.time()-t0:.0f}s', flush=True)
    json.dump({'N': N, 'palette_size': len(pal_list), 'meta': meta}, open(f'{OUT}/builds_meta.full.json','w'))
    # 报告
    nmc = {}
    for (nm, ps) in pal_list:
        nmc[nm] = nmc.get(nm,0)+1
    print('palette names:', len(nmc), flush=True)
    for nm,c in sorted(nmc.items(), key=lambda kv:-kv[1])[:20]:
        print(f'  {nm} x{c}', flush=True)
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
