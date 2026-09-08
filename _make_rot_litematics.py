# -*- coding: utf-8 -*-
"""从旋转后的 ss64 voxel 生成 litematic(供渲染), 与 latent 编码严格同源.
输入: ss64_rot.npy(由 _encode_rot_latents.py 生成, [4*N,64,64,64]), valid_list 文件名,
      block_map+nid->name 映射(用 BLOCK_VOCAB_FULL + block_map).
输出: /home/linux/redstone_ai/render_mc/rot_litematics/<idx>_r{k}.litematic  (k=1,2,3; 0°复用原图)
注意: 只导出 voxel 里的非零 class(需 id->MC name), state 无法从 class 还原 -> 渲染用默认 state(仅影响朝向类方块外观).
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import litemapy
from litemapy import Region, BlockState, Schematic

BASE = '/home/linux/redstone_ai/dataset'
LIST = f'{BASE}/valid_files_list.txt'
OUT = '/home/linux/redstone_ai/render_mc/rot_litematics'
BLOCK_MAP = f'{BASE}/v2/block_map.npy'
VOCAB = f'{BASE}/BLOCK_VOCAB_FULL.txt'
N_ROT = 4
os.makedirs(OUT, exist_ok=True)

# id -> MC name
block_map = np.load(BLOCK_MAP)
vocab = [l.strip() for l in open(VOCAB, encoding='utf-8') if l.strip()]
def id_to_name(nid):
    oi = int(block_map[nid])
    return vocab[oi] if 0 <= oi < len(vocab) else 'minecraft:stone'

ss = np.load(f'{BASE}/v2/ss64_rot.npy', mmap_mode='r')
N = ss.shape[0] // N_ROT
G = 64
print('ss64_rot N*4 =', ss.shape[0], 'N =', N, flush=True)

cnt_out = 0
for i in range(N):
    for k in range(1, N_ROT):  # 只 90/180/270
        idx4 = i * N_ROT + k
        v = np.asarray(ss[idx4]).astype(np.int16)
        nz = np.argwhere(v > 0)
        if len(nz) == 0:
            continue
        # voxel 坐标已在 0..63 (64^3 体素), 直接用; schematic-renderer 自动聚焦
        reg = Region(0, 0, 0, G, G, G)
        for (x, y, z) in nz:
            nid = int(v[x, y, z])
            nm = id_to_name(nid)
            if nm == 'minecraft:air':
                nm = 'minecraft:stone'
            try:
                reg[int(x), int(y), int(z)] = BlockState(nm)
            except Exception:
                try:
                    reg[int(x), int(y), int(z)] = BlockState('minecraft:stone')
                except Exception:
                    pass
        sch = Schematic(name=f'idx{i}_r{k*90}', author='ai', regions={'main': reg})
        sch.save(f'{OUT}/{i:04d}_r{k*90:03d}.litematic')
        cnt_out += 1
    if (i + 1) % 200 == 0:
        print(f'  {i+1}/{N} wrote {cnt_out} litematics', flush=True)
print(f'DONE wrote {cnt_out} rot litematics', flush=True)
