# -*- coding: utf-8 -*-
import zipfile
J = '/home/linux/.gradle/caches/fabric-loom/minecraftMaven/net/minecraft/minecraft-merged/1.21.11-loom.mappings.1_21_11.layered+hash.2198-v2/minecraft-merged-1.21.11-loom.mappings.1_21_11.layered+hash.2198-v2.jar'
z = zipfile.ZipFile(J)
hits = [n for n in z.namelist() if 'GameRules' in n or n.endswith('GameRules.class')]
for h in hits[:10]:
    print(h)
# 找 RULE_DAYLIGHT 常量所在
data = z.read('net/minecraft/world/level/GameRules.class') if 'net/minecraft/world/level/GameRules.class' in z.namelist() else b''
print('level/GameRules present:', bool(data))
