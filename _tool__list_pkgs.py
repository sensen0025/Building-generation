# -*- coding: utf-8 -*-
import sys, io, zipfile, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

J = '/home/linux/.gradle/caches/fabric-loom/minecraftMaven/net/minecraft/minecraft-merged/1.21.11-loom.mappings.1_21_11.layered+hash.2198-v2/minecraft-merged-1.21.11-loom.mappings.1_21_11.layered+hash.2198-v2.jar'
z = zipfile.ZipFile(J)
names = z.namelist()
def show(pkg, kw=None, limit=40):
    hits = [n for n in names if n.startswith(pkg) and n.endswith('.class')]
    if kw:
        hits = [h for h in hits if kw in h]
    print(f'== {pkg} (kw={kw}): {len(hits)}')
    for h in hits[:limit]:
        print('   ', h)
show('net/minecraft/resources/')
show('net/minecraft/world/level/block/state/properties/', limit=80)
show('net/minecraft/core/')
