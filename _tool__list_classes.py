# -*- coding: utf-8 -*-
import sys, io, zipfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

J = '/home/linux/.gradle/caches/fabric-loom/minecraftMaven/net/minecraft/minecraft-merged/1.21.11-loom.mappings.1_21_11.layered+hash.2198-v2/minecraft-merged-1.21.11-loom.mappings.1_21_11.layered+hash.2198-v2.jar'
z = zipfile.ZipFile(J)
names = z.namelist()
print('total entries:', len(names))
import re
pats = [
    r'^net/minecraft/client/Minecraft\.class$',
    r'^net/minecraft/client/Options\.class$',
    r'^net/minecraft/client/Screenshot\.class$',
    r'^net/minecraft/client/CloudStatus\.class$',
    r'^net/minecraft/client/ParticleStatus\.class$',
    r'^net/minecraft/client/renderer/RenderTarget\.class$',
    r'^net/minecraft/client/renderer/GameRenderer\.class$',
    r'^net/minecraft/client/gui/screens/worldselection/WorldOpenFlows\.class$',
    r'^net/minecraft/client/multiplayer/ClientLevel\.class$',
    r'^net/minecraft/client/player/LocalPlayer\.class$',
    r'^net/minecraft/client/player/AbstractClientPlayer\.class$',
    r'^net/minecraft/server/MinecraftServer\.class$',
    r'^net/minecraft/server/level/ServerLevel\.class$',
    r'^net/minecraft/world/level/block/state/BlockState\.class$',
    r'^net/minecraft/world/level/block/state/properties/Property\.class$',
    r'^net/minecraft/world/level/block/state/properties/(BooleanProperty|IntegerProperty|EnumProperty|DirectionProperty)\.class$',
    r'^net/minecraft/core/registries/BuiltInRegistries\.class$',
    r'^net/minecraft/resources/ResourceLocation\.class$',
    r'^net/minecraft/core/BlockPos\.class$',
    r'^net/minecraft/world/entity/Entity\.class$',
    r'^net/minecraft/world/entity/player/Player\.class$',
    r'^net/minecraft/client/gui/screens/Screen\.class$',
    r'^net/minecraft/client/gui/screens/TitleScreen\.class$',
]
for p in pats:
    hits = [n for n in names if re.match(p, n)]
    print(('OK ' if hits else 'MISS ') + p + ' -> ' + (hits[0] if hits else ''))
