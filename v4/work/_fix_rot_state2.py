# -*- coding: utf-8 -*-
"""修复 state 语义: 旋转槽位 (k>0) 的属性值随几何一起重映射.

权威 schema (来自 v1 源码 _voxelize.py):
  state = o + 7*(v + 3*(sh + 6*(hz + 16*ac)))
  o : facing {north:1,south:2,east:3,west:4,up:5,down:6};
      无 facing 时用 axis {x:3,z:1,y:5}   <-- 与 facing 共用码域, 必须按方块类别区分
  v : half/type {top:1,bottom:2}          绕 y 旋转不变
  sh: 楼梯 shape {straight:1,inner_left:2,...}  相对 facing 定义, 刚体旋转不变
  hz: 4 位连接掩码, 位序 [north,south,east,west]
  ac: {lit,powered,open,triggered,extended} 状态, 旋转不变

方向映射实测 (合成方块探针): d_new = (-dz, dy, dx)
  north(-Z)->east(+X), east(+X)->south(+Z), south(+Z)->west(-X), west(-X)->north(-Z)

用法:
  python _fix_rot_state2.py --validate      # 校验(不写文件)
  python _fix_rot_state2.py                 # 写 train/ss64_state_all_rotfix.npy
"""
import os
import sys
import json
import time

import numpy as np

SRC = '/data/redstone_v4/train/ss64_state_all.npy'
BLK = '/data/redstone_v4/train/ss64_all.npy'
DST = '/data/redstone_v4/train/ss64_state_all_rotfix.npy'
REPORT = '/data/redstone_v4/work/state_rotfix_report.json'
N0 = 21356
O_DIM, V_DIM, SH_DIM, HZ_DIM, AC_DIM = 7, 3, 6, 16, 3
MULT = [1, 7, 21, 126, 2016]

# ---- facing 用 4-循环; axis 只涉及 1(z)<->3(x), 5(y) 不变 ----
R_FACING_1 = np.array([0, 3, 4, 2, 1, 5, 6], dtype=np.int64)
R_AXIS_1 = np.array([0, 3, 2, 1, 4, 5, 6], dtype=np.int64)

# ---- hz 位重映射: 位方向按 d_new 搬移 ----
DIRS = {0: (0, 0, -1), 1: (0, 0, 1), 2: (1, 0, 0), 3: (-1, 0, 0)}   # bit -> 方向
DIR2BIT = {v: k for k, v in DIRS.items()}
HZ_1 = np.zeros(HZ_DIM, dtype=np.int64)
for old in range(HZ_DIM):
    n = 0
    for b, d in DIRS.items():
        if (old >> b) & 1:
            nd = (-d[2], d[1], d[0])
            n |= 1 << DIR2BIT[nd]
    HZ_1[old] = n


def compose(table, k):
    out = np.arange(len(table), dtype=np.int64)
    for _ in range(k):
        out = np.asarray(table, dtype=np.int64)[out]
    return out


R_TAB = {'facing': {k: compose(R_FACING_1, k) for k in (0, 1, 2, 3)},
         'axis': {k: compose(R_AXIS_1, k) for k in (0, 1, 2, 3)}}
HZ_TAB = {k: compose(HZ_1, k) for k in (0, 1, 2, 3)}

# ---- 哪些方块类的 o 来自 axis: 依据扫描 248 个源 litematic 得到的真实属性 ----
SCAN = '/data/redstone_v4/work/block_props_scan.json'
# 扫描中未出现但同族明确为 axis 的类
AXIS_EXTRA = {'minecraft:stripped_acacia_wood', 'minecraft:stripped_pale_oak_wood'}


def build_axis_flag():
    vocab = [l.strip() for l in open('/home/linux/redstone_ai/dataset/BLOCK_VOCAB_FULL.txt',
                                     encoding='utf-8') if l.strip()] + ['minecraft:other']
    bm = np.load('/home/linux/redstone_ai/dataset/v2/block_map.npy')
    scan = json.load(open(SCAN))['result'] if os.path.exists(SCAN) else {}
    axis_names = set(k for k, v in scan.items() if v == 'axis') | AXIS_EXTRA
    flag = np.zeros(128, dtype=np.int64)
    names = {}
    for cid in range(128):
        nm = vocab[int(bm[cid])] if int(bm[cid]) < len(vocab) else 'minecraft:other'
        names[cid] = nm
        if nm in axis_names:
            flag[cid] = 1
    # 其余一律按 facing 处理: 无朝向属性的类 o 恒为 0, 走 facing 表等价于不变
    return flag, names


def k_of(slot):
    return slot % 4 if slot < N0 else (slot - N0) % 4


def main():
    do_val = '--validate' in sys.argv
    axis_flag, names = build_axis_flag()
    n_axis = int((axis_flag[1:] > 0).sum())
    print(f'axis-like 方块类: {n_axis} / 127')
    print('  ' + ', '.join(names[c] for c in range(128) if axis_flag[c] and c))
    print(f'R facing k=1: {R_TAB["facing"][1].tolist()}')
    print(f'R axis   k=1: {R_TAB["axis"][1].tolist()}')
    print(f'HZ k=1: {HZ_TAB[1].tolist()}', flush=True)

    st = np.load(SRC, mmap_mode='r')
    bl = np.load(BLK, mmap_mode='r')
    M = st.shape[0]
    CH = 128
    t0 = time.time()

    def remap_chunk(state_chunk, blk_chunk, slots):
        out = state_chunk.copy()
        for i, slot in enumerate(slots):
            k = k_of(int(slot))
            if k == 0:
                continue
            p = out[i].astype(np.int64)
            o = p % O_DIM
            hz = (p // MULT[3]) % HZ_DIM
            af = axis_flag[blk_chunk[i]]                 # 1=axis 语义, 0=facing 语义
            o_new = np.where(af > 0, R_TAB['axis'][k][o], R_TAB['facing'][k][o])
            hz_new = HZ_TAB[k][hz]
            out[i] = (p - o + o_new + MULT[3] * (hz_new - hz)).astype(np.uint16)
        return out

    if do_val:
        slots = list(range(0, 1024))
        bad0 = 0
        for i in range(0, len(slots), CH):
            sl = slots[i:i + CH]
            a = np.asarray(st[sl]); b = np.asarray(bl[sl])
            new = remap_chunk(a, b, sl)
            for j, s in enumerate(sl):
                if k_of(int(s)) == 0 and not (new[j] == a[j]).all():
                    bad0 += 1
        print(f'k=0 槽位被改动数: {bad0} (应为 0)')
        # 值域合法性
        mx = 0
        for i in range(0, len(slots), CH):
            sl = slots[i:i + CH]
            new = remap_chunk(np.asarray(st[sl]), np.asarray(bl[sl]), sl)
            mx = max(mx, int(new.max()))
        print(f'重映射后 packed 最大值 {mx} (schema 上限 {7*3*6*16*3 - 1})')
        print('R 与 HZ 均为双射:',
              len(set(R_TAB['facing'][1].tolist())) == O_DIM,
              len(set(R_TAB['axis'][1].tolist())) == O_DIM,
              len(set(HZ_TAB[1].tolist())) == HZ_DIM)
        # 统计改动量
        ch_cells = 0
        for i in range(0, len(slots), CH):
            sl = slots[i:i + CH]
            a = np.asarray(st[sl]); b = np.asarray(bl[sl])
            new = remap_chunk(a, b, sl)
            ch_cells += int((new != a).sum())
        print(f'前 {len(slots)} 槽位改动格点: {ch_cells:,}')
        return

    out = np.lib.format.open_memmap(DST, mode='w+', dtype=np.uint16, shape=st.shape)
    changed = 0
    for i in range(0, M, CH):
        sl = list(range(i, min(i + CH, M)))
        a = np.asarray(st[i:i + CH]); b = np.asarray(bl[i:i + CH])
        new = remap_chunk(a, b, sl)
        out[i:i + CH] = new
        changed += int((new != a).sum())
        if (i // CH) % 25 == 0:
            out.flush()
            print(f'  {i+len(sl)}/{M} ({time.time()-t0:.0f}s) changed={changed:,}', flush=True)
    out.flush()
    del out
    print(f'DONE {DST} changed_cells={changed:,} in {time.time()-t0:.0f}s', flush=True)
    json.dump({'src': SRC, 'dst': DST, 'slots': int(M), 'changed_cells': changed,
               'schema': 'o + 7*(v + 3*(sh + 6*(hz + 16*ac)))',
               'direction_map': 'd_new=(-dz,dy,dx): north->east->south->west',
               'R_facing_k1': R_TAB['facing'][1].tolist(),
               'R_axis_k1': R_TAB['axis'][1].tolist(),
               'HZ_k1': HZ_TAB[1].tolist(),
               'axis_like_classes': [names[c] for c in range(128) if axis_flag[c] and c]},
              open(REPORT, 'w'), indent=1)


if __name__ == '__main__':
    main()
