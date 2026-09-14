# -*- coding: utf-8 -*-
"""端到端校验: 修复后, 训练管线读到的条件图是否已随朝向分开."""
import os
import sys
import hashlib
import json

os.environ.setdefault('ATTN_BACKEND', 'sdpa')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')

CFG = '/data/redstone_v4/work/ss_flow_mc_v4_comb.json'
NEWSTATE = '/data/redstone_v4/train/ss64_state_all_rotfix.npy'
N0 = 21356

cfg = json.load(open(CFG))
d = cfg['dataset']['args']
IMG = d['image_dir']


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()[:10]


def p(slot):
    # 与数据集 mc_flow.py 一致的命名规则
    return os.path.join(IMG, 'bld_%04d.png' % slot)


print('=== 用数据集同一套命名规则检查 (image_dir=%s) ===' % IMG)
same = 0
tot = 0
for j in [0, 100, 1000, 3000, 5076, 6000]:
    k0 = N0 + j * 4
    hs = [md5(p(k0 + k)) for k in range(4)]
    distinct = len(set(hs))
    tot += 1
    same += (distinct == 4)
    print('  j=%-5d 4 槽位 md5 distinct=%d  %s' % (j, distinct, hs))
print('  四槽位图像互不相同: %d/%d' % (same, tot))

print()
print('=== 数据集实际加载 (states 指向修复后文件) ===')
from trellis.datasets.mc_flow import MCFlowImageDataset
ds = MCFlowImageDataset(roots='', latent_npy=d['latent_npy'], image_dir=IMG,
                        image_size=d.get('image_size', 518), normalize=d.get('normalize'),
                        blocks_npy=d['blocks_npy'], states_npy=NEWSTATE,
                        inpaint_probability=d.get('inpaint_probability', 0.0),
                        up_axis=d.get('up_axis', 1))
print('  可用样本数:', len(ds.ids), '(修复前 45544)')
for slot in (N0 + 5076 * 4, N0 + 5076 * 4 + 1):
    if slot in ds.ids:
        s = ds[ds.ids.index(slot)]
        print('  slot %d: cond%s  known_states.max=%d  known_blocks>0=%d' % (
            slot, tuple(s['cond'].shape), int(s['known_states'].max()),
            int((s['known_blocks'] > 0).sum())))
