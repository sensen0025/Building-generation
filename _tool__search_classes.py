# -*- coding: utf-8 -*-
import sys, io, zipfile, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

J = '/home/linux/.gradle/caches/fabric-loom/minecraftMaven/net/minecraft/minecraft-merged/1.21.11-loom.mappings.1_21_11.layered+hash.2198-v2/minecraft-merged-1.21.11-loom.mappings.1_21_11.layered+hash.2198-v2.jar'
z = zipfile.ZipFile(J)
names = z.namelist()
for kw in ['RenderTarget', 'ParticleStatus', 'ResourceLocation', 'EnumProperty', 'IntegerProperty', 'DirectionProperty', 'Screenshot', 'WorldOpenFlows', 'createWorldOpenFlows']:
    if kw.startswith('create'):
        continue
    hits = [n for n in names if kw in n and n.endswith('.class')]
    print(f'== {kw}: {len(hits)}')
    for h in hits[:6]:
        print('   ', h)
