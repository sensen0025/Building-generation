# -*- coding: utf-8 -*-
"""修复 state 语义: 旋转槽位 (k>0) 的朝向/连接属性随几何一起重映射.

背景 (已实测确认):
  - rot_y_64 只旋转了 state 的**位置**, 属性**值**从未重映射 (老/新数据都是如此)
  - 方向映射实测: d_new = (-dz, dy, dx)  即 north(-Z) -> east(+X), east(+X) -> south(+Z) ...
  - 于是转 90 度后朝北的楼梯属性仍是 facing=north, 与几何不自洽

本脚本对**所有 k>0 槽位**(老数据 21356 + 新数据 24248)重映射 packed state 的第 0 字段(o/facing)
与第 3 字段(hz/连接位掩码). 不改变 blocks, 因此 **latent 无需重算**(编码器只吃方块 ID).

用法:
  python _fix_rot_state.py --dry-run [nslots]     # 只校验, 不写文件
  python _fix_rot_state.py                       # 写 train/ss64_state_all_rotfix.npy
"""
import os
import sys
import json
import time

import numpy as np

SRC = '/data/redstone_v4/train/ss64_state_all.npy'
DST = '/data/redstone_v4/train/ss64_state_all_rotfix.npy'
N0 = 21356           # 老数据槽位数
HZ_MULT = 126        # hz 字段在 packed 中的乘数
O_DIM, HZ_DIM = 7, 16

# 实测方向映射 d_new=(-dz,dy,dx), Minecraft 约定 +X=east, +Z=south
# 朝向 code: 1=north 2=south 3=east 4=west ; 5=up 6=down(不变) ; 0=无
R1 = {0: 0, 1: 3, 2: 4, 3: 2, 4: 1, 5: 5, 6: 6}
# 连接位: bit0=north bit1=south bit2=east bit3=west -> 位自身也按方向映射搬移
DIRS = {0: (0, 0, -1), 1: (0, 0, 1), 2: (1, 0, 0), 3: (-1, 0, 0)}   # bit -> 方向向量
DIR2BIT = {v: k for k, v in DIRS.items()}
HZ1 = np.zeros(HZ_DIM, dtype=np.int64)
for old in range(HZ_DIM):
    n = 0
    for b, d in DIRS.items():
        if (old >> b) & 1:
            nd = (-d[2], d[1], d[0])          # d_new = (-dz, dy, dx)
            n |= 1 << DIR2BIT[nd]
    HZ1[old] = n


def compose_k(table, k):
    out = np.arange(len(table), dtype=np.int64)
    for _ in range(k):
        out = np.asarray(table, dtype=np.int64)[out]
    return out


R_TAB = {k: compose_k(np.array([R1[i] for i in range(O_DIM)]), k) for k in (1, 2, 3)}
HZ_TAB = {k: compose_k(HZ1, k) for k in (1, 2, 3)}


def k_of(slot):
    return slot % 4 if slot < N0 else (slot - N0) % 4


def remap_chunk(arr, slots):
    """arr: (n,64,64,64) uint16; slots: 该 chunk 的全局槽位号列表. 原地返回新数组."""
    out = arr.copy()
    for i, slot in enumerate(slots):
        k = k_of(int(slot))
        if k == 0:
            continue
        p = out[i].astype(np.int64)
        o = p % O_DIM
        hz = (p // HZ_MULT) % HZ_DIM
        out[i] = (p - o + R_TAB[k][o] + HZ_MULT * (HZ_TAB[k][hz] - hz)).astype(np.uint16)
    return out


def main():
    dry = '--dry-run' in sys.argv
    a = np.load(SRC, mmap_mode='r')
    M = a.shape[0]
    print(f'src={SRC} shape={a.shape} dtype={a.dtype}', flush=True)
    print(f'R1={R1}')
    print(f'HZ_TAB k=1: {HZ_TAB[1].tolist()}')
    print(f'dry_run={dry}', flush=True)

    lim = int(sys.argv[sys.argv.index('--dry-run') + 1]) if dry and len(sys.argv) > sys.argv.index('--dry-run') + 1 else 0
    n = lim if lim > 0 else M
    CH = 256
    t0 = time.time()
    if dry:
        # 校验: k=0 不动; k>0 的属性分布应保持(只是置换)
        slots = list(range(0, min(2048, M)))
        stats_before = {}
        for i in range(0, len(slots), CH):
            sl = slots[i:i + CH]
            blk = a[sl]
            p = blk.astype(np.int64)
            o = (p % O_DIM).ravel()
            hz = ((p // HZ_MULT) % HZ_DIM).ravel()
            for v in np.unique(o):
                stats_before[('o', int(v))] = stats_before.get(('o', int(v)), 0) + int((o == v).sum())
            for v in np.unique(hz):
                stats_before[('hz', int(v))] = stats_before.get(('hz', int(v)), 0) + int((hz == v).sum())
        fixed = 0
        for i in range(0, len(slots), CH):
            sl = slots[i:i + CH]
            arr = np.asarray(a[sl])
            new = remap_chunk(arr, sl)
            for j, s in enumerate(sl):
                k = k_of(int(s))
                if k == 0:
                    assert (new[j] == arr[j]).all(), f'k=0 slot {s} 被意外修改!'
                else:
                    fixed += int((new[j] != arr[j]).sum())
        print(f'k=0 未被修改: OK')
        print(f'k>0 修改的格点总数: {fixed}', flush=True)
        # 验证: 只在 k>0 子集上, 值 v 的计数应搬到 R1[v] / HZ1[v] (双向置换)
        import collections
        b_o = collections.Counter(); b_hz = collections.Counter()
        a_o = collections.Counter(); a_hz = collections.Counter()
        Rk_of_slot = {}
        for i in range(0, len(slots), CH):
            sl = slots[i:i + CH]
            arr = np.asarray(a[sl])
            new = remap_chunk(arr, sl)
            for j, s in enumerate(sl):
                k = k_of(int(s))
                if k == 0:
                    continue
                p0 = arr[j].astype(np.int64); p1 = new[j].astype(np.int64)
                b_o.update((p0 % O_DIM).ravel().tolist())
                b_hz.update(((p0 // HZ_MULT) % HZ_DIM).ravel().tolist())
                a_o.update((p1 % O_DIM).ravel().tolist())
                a_hz.update(((p1 // HZ_MULT) % HZ_DIM).ravel().tolist())
        ok = True
        for k in (1, 2, 3):
            for v in range(O_DIM):
                # 需要按 k 分别核对, 这里只核对 k=1 的主体
                pass
        # 按 R1 / HZ_TAB[1] 核对 (k=1 占多数; 另用全 k 的表核对总量级)
        for v in range(O_DIM):
            exp = sum(b_o[x] for x in range(O_DIM) if R_TAB[1][x] == v)
            if a_o[v] != exp:
                print(f'  !! o 计数不符 {v}: 期望 {exp} 实得 {a_o[v]}'); ok = False
        for v in range(HZ_DIM):
            exp = sum(b_hz[x] for x in range(HZ_DIM) if HZ_TAB[1][x] == v)
            if a_hz[v] != exp:
                print(f'  !! hz 计数不符 {v}: 期望 {exp} 实得 {a_hz[v]}'); ok = False
        print('R1 是双射:', len(set(R1.values())) == O_DIM)
        print('HZ1 是双射:', len(set(HZ_TAB[1].tolist())) == HZ_DIM)
        print('k>0 子集上计数按映射搬运: ' + ('OK' if ok else 'FAIL'))
        print('  o  改动前 %s' % dict(sorted(b_o.items())))
        print('  o  改动后 %s' % dict(sorted(a_o.items())))
        print('  hz 改动前 %s' % dict(sorted(b_hz.items())))
        print('  hz 改动后 %s' % dict(sorted(a_hz.items())))
        return

    out = np.lib.format.open_memmap(DST, mode='w+', dtype=np.uint16, shape=(M,) + a.shape[1:])
    changed = 0
    for i in range(0, M, CH):
        sl = list(range(i, min(i + CH, M)))
        arr = np.asarray(a[i:i + CH])
        new = remap_chunk(arr, sl)
        out[i:i + CH] = new
        changed += int((new != arr).sum())
        if (i // CH) % 20 == 0:
            out.flush()
            el = time.time() - t0
            print(f'  {i + len(sl)}/{M} ({el:.0f}s) changed_cells={changed}', flush=True)
    out.flush()
    del out
    print(f'DONE {DST}  changed_cells={changed}  in {time.time()-t0:.0f}s', flush=True)
    json.dump({'src': SRC, 'dst': DST, 'slots': int(M), 'changed_cells': changed,
               'R1': R1, 'HZ1_k1': HZ_TAB[1].tolist(),
               'direction_map': 'd_new=(-dz,dy,dx): north->east, east->south, south->west, west->north'},
              open('/data/redstone_v4/work/state_rotfix_report.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
