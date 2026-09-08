# -*- coding: utf-8 -*-
import sys, io, zipfile, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

J = '/home/linux/.gradle/caches/fabric-loom/minecraftMaven/net/minecraft/minecraft-merged/1.21.11-loom.mappings.1_21_11.layered+hash.2198-v2/minecraft-merged-1.21.11-loom.mappings.1_21_11.layered+hash.2198-v2.jar'

classes = [
    'net/minecraft/client/Minecraft',
    'net/minecraft/client/Options',
    'net/minecraft/client/Screenshot',
    'net/minecraft/client/renderer/GameRenderer',
    'net/minecraft/client/gui/screens/worldselection/WorldOpenFlows',
    'net/minecraft/client/multiplayer/ClientLevel',
    'net/minecraft/client/player/LocalPlayer',
    'net/minecraft/server/MinecraftServer',
    'net/minecraft/server/level/ServerLevel',
    'net/minecraft/server/level/ParticleStatus',
    'net/minecraft/client/CloudStatus',
    'net/minecraft/world/level/block/state/BlockState',
    'net/minecraft/world/level/block/state/properties/Property',
    'net/minecraft/world/level/block/state/properties/BooleanProperty',
    'net/minecraft/world/level/block/state/properties/EnumProperty',
    'net/minecraft/core/registries/BuiltInRegistries',
    'net/minecraft/resources/Identifier',
    'net/minecraft/core/BlockPos',
    'net/minecraft/world/entity/Entity',
    'net/minecraft/world/entity/player/Player',
    'net/minecraft/world/level/block/Block',
    'com/mojang/blaze3d/pipeline/RenderTarget',
    'net/minecraft/client/player/AbstractClientPlayer',
]
flt = '|'.join(re.escape(c) for c in classes)
print('writing filter file with', len(classes), 'classes')
open('/home/linux/rhrender/cfr_classes.txt', 'w').write('\n'.join(classes))
