# -*- coding: utf-8 -*-
"""找含 minecraft:water 的样例(验证清除效果用)."""
import numpy as np
ss = np.load('/home/linux/redstone_ai/dataset/v2/ss64.npy', mmap_mode='r')
bm = np.load('/home/linux/redstone_ai/dataset/v2/block_map.npy')
vocab = [l.strip() for l in open('/home/linux/redstone_ai/dataset/BLOCK_VOCAB_FULL.txt', encoding='utf-8') if l.strip()]
water_cls = [c for c in range(1, 128) if 0 <= int(bm[c]) < len(vocab) and 'water' in vocab[int(bm[c])]]
print('water classes:', water_cls)
found = []
for i in range(ss.shape[0]):
    v = np.asarray(ss[i])
    if any((v == c).any() for c in water_cls):
        found.append(i)
        if len(found) >= 12:
            break
print('indices with water:', found)
