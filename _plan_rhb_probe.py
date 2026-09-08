# -*- coding: utf-8 -*-
"""plan-mode 只读: 检查 builds.rhb 结构 + 现有 rhb 是否可直接旋转复用.
同时确认 render_output 现状(0°图已存在, 旋转只需90/180/270)."""
import sys, io, struct, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RHB = '/home/linux/redstone_ai/render_mc/builds.rhb'
META = '/home/linux/redstone_ai/render_mc/builds_meta.json'
REN_OUT = '/home/linux/redstone_ai/render_output'

# 检查 render_output 0° 图数量
import glob
n_png = len(glob.glob(f'{REN_OUT}/bld_*.png'))
print('render_output 0deg png:', n_png)

# rhb 头部解析
with open(RHB, 'rb') as f:
    f.read(4)
    npal = struct.unpack('<I', f.read(4))[0]
    for _ in range(npal):
        nl = struct.unpack('<H', f.read(2))[0]; f.read(nl)
        pl = struct.unpack('<H', f.read(2))[0]; f.read(pl)
    N = struct.unpack('<I', f.read(4))[0]
    print('rhb: npal', npal, 'N', N)

meta = json.load(open(META))
print('meta N:', meta['N'], 'entries:', len(meta['meta']))
# 检查 0° 图是否与 render_output 对应(应该有 5339)
print('DONE')
