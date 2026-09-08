# -*- coding: utf-8 -*-
"""远程: 正确解析新 rhb, 对比 idx250/600 实际方块坐标 vs meta bbox (判定放置是否错位)."""
import sys, io, struct, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RHB = '/home/linux/redstone_ai/render_mc/builds.rhb'
META = '/home/linux/redstone_ai/render_mc/builds_meta.json'

with open(RHB, 'rb') as f:
    magic = f.read(4)
    npal = struct.unpack('<I', f.read(4))[0]
    print('magic', magic, 'npal', npal)
    entries = []
    for _ in range(npal):
        nl = struct.unpack('<H', f.read(2))[0]
        name = f.read(nl).decode('utf-8', 'replace')
        pl = struct.unpack('<H', f.read(2))[0]
        props = f.read(pl).decode('utf-8', 'replace')
        entries.append((name, props))
    N = struct.unpack('<I', f.read(4))[0]
    print('N', N, 'first palette entries:', entries[:5])
    # 顺序读取每建筑块数构建 offsets
    counts = []
    for i in range(N):
        nb = struct.unpack('<I', f.read(4))[0]
        counts.append(nb)
    print('counts sample idx0-5:', counts[:6])
    print('counts idx250/600:', counts[250], counts[600])

meta = json.load(open(META))['meta']
for i in [0, 250, 600, 2200]:
    print(f'idx{i}: meta n={meta[i]["n"]} counts={counts[i]}')
