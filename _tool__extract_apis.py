# -*- coding: utf-8 -*-
"""从 apis_all.java 抽取关键类的方法/字段签名."""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SRC = '/home/linux/rhrender/apis_all.java'
txt = open(SRC, encoding='utf-8', errors='replace').read()

def section(class_re, kw_lines, label, maxlines=4000):
    m = re.search(r'class ' + class_re + r'(?:<[^>]*>)?[^{]*\{', txt)
    if not m:
        print(f'### {label}: CLASS NOT FOUND')
        return
    start = m.start()
    # 简单括号配平找类体结束
    depth = 0
    i = start
    while i < len(txt):
        if txt[i] == '{':
            depth += 1
        elif txt[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = txt[start:i + 1]
    print(f'### {label}: {len(body)} chars')
    for kw in kw_lines:
        hits = [ln.strip() for ln in body.splitlines() if kw in ln]
        for h in hits[:14]:
            print('   ', h[:220])
        print(f'   --- ({len(hits)} hits for "{kw}")')
    print()

# Minecraft
section(r'Minecraft\b', ['public static Minecraft getInstance', 'getSingleplayerServer', 'public final Options', 'public ClientLevel level', 'LocalPlayer player', 'getMainRenderTarget', 'createWorldOpenFlows', 'public void execute', 'runTick', 'public Window', 'getWindow', 'getGameRenderer', 'getResourceManager'], 'Minecraft')

# Options 云/粒子/暂停
section(r'Options\b', ['cloudStatus', 'CloudStatus', 'particles', 'ParticleStatus', 'pauseOnLostFocus', 'fov', 'setOption', 'fullscreen', 'guiScale'], 'Options')

# Screenshot
section(r'Screenshot\b', ['public static', 'static', 'grab', 'saveScreenshot', 'TaskResult'], 'Screenshot')

# WorldOpenFlows
section(r'WorldOpenFlows\b', ['openWorld', 'createFreshLevel', 'WorldOpenFlows', 'public'], 'WorldOpenFlows')

# ServerLevel.setBlock / MinecraftServer
section(r'ServerLevel\b', ['setBlock', 'public void setDayTime', 'getLevel'], 'ServerLevel')
section(r'MinecraftServer\b', ['public ServerLevel getLevel', 'public void execute', 'createCommandSourceStack', 'getCommands', 'public boolean isSameThread', 'getPlayerList'], 'MinecraftServer')

# BlockState/Block/props
section(r'BlockState\b', ['getProperties', 'setValue', 'public', 'getBlock'], 'BlockState')
section(r'Block\b', ['defaultBlockState'], 'Block')
section(r'class BuiltInRegistries\b', ['BLOCK', 'public static'], 'BuiltInRegistries')

# Identifier
section(r'class Identifier\b', ['parse', 'of(', 'withDefaultNamespace', 'public static Identifier'], 'Identifier')

# Entity/Player 相机
section(r'abstract class Entity\b', ['setPos', 'setYRot', 'setXRot', 'getEyeHeight', 'getYRot', 'getXRot', 'moveTo', 'teleportTo'], 'Entity')
section(r'class LocalPlayer\b', ['public', 'tick'], 'LocalPlayer')

# ParticleStatus/CloudStatus enums
section(r'enum ParticleStatus\b', ['OFF', 'MINIMAL', 'DECREASED', 'ALL'], 'ParticleStatus')
section(r'enum CloudStatus\b', ['OFF', 'FAST', 'FANCY'], 'CloudStatus')
