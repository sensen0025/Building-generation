# -*- coding: utf-8 -*-
"""从 train/ss64_all.npy 的指定 slot 直接导出 litematic (用于验证旋转渲染).
用法: python _mk_slot_litematic.py <slot_index> <out.litematic>
"""
import sys
import numpy as np

sys.path.insert(0, '/data/redstone_v4/work')
import _gen_v4_test as G  # noqa: E402  (仅模块级常量, 不初始化 CUDA)

slot = int(sys.argv[1])
out = sys.argv[2]

ss = np.load('/data/redstone_v4/train/ss64_all.npy', mmap_mode='r')
st = np.load('/data/redstone_v4/train/ss64_state_all.npy', mmap_mode='r')
v = np.asarray(ss[slot]).astype(np.int64)
s = np.asarray(st[slot]).astype(np.int64)
n = G.to_litematic(v, G.fields_from_packed(s), f'slot{slot}', out)
print(f'slot {slot}: {n} blocks -> {out}', flush=True)
