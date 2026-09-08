# -*- coding: utf-8 -*-
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SRC = '/home/linux/rhrender/apis_all.java'
txt = open(SRC, encoding='utf-8', errors='replace').read()

def body_of(pat):
    m = re.search(pat, txt)
    if not m:
        return ''
    start = m.start()
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
    return txt[start:i + 1]

b = body_of(r'class Minecraft[^{]*\{')
print('== Minecraft level-ish lines ==')
for ln in b.splitlines():
    s = ln.strip()
    if 'level' in s and ('ClientLevel' in s or s.startswith('public') or '= null;' in s) and 'public static' not in s:
        if len(s) < 130 and ('level' in s):
            print('  ', s[:120])
print('== getInstance usage of level ==')
for m2 in re.findall(r'.{0,60}\bminecraft\.level\b.{0,40}', b)[:3]:
    print('   ...', m2[:120])
for m2 in re.findall(r'.{0,60}\bthis\.level\b.{0,60}', b)[:5]:
    print('   this.level...', m2[:120])

# Identifier parse decl anywhere
for m2 in re.findall(r'public static Identifier (parse|of|withDefaultNamespace)[^;{]*', txt)[:8]:
    print('Identifier decl:', m2[:120])

# Commands.performPrefixedCommand
b = body_of(r'class Commands[^{]*\{')
if b:
    for ln in b.splitlines():
        s = ln.strip()
        if 'performPrefixedCommand' in s and 'public' in s:
            print('Commands:', s[:150])
else:
    print('Commands class not found')

# MinecraftServer execute / 常用方法确认
b = body_of(r'class MinecraftServer[^{]*\{')
for ln in b.splitlines():
    s = ln.strip()
    if s.startswith('public') and ('executeIfPossible' in s or 'isSameThread' in s or 'getWorldData' in s or 'getPlayerList' in s or 'getCommands' in s or 'getLevel(' in s):
        print('Server:', s[:150])

# hideGui field in Options
b = body_of(r'class Options[^{]*\{')
for ln in b.splitlines():
    s = ln.strip()
    if 'hideGui' in s or 'renderDistance' in s and 'OptionInstance' in s:
        print('Opt:', s[:130])

# mc.player 类型 LocalPlayer 方法 moveTo? 及 Camera?
b = body_of(r'class LocalPlayer[^{]*\{')
print('LocalPlayer ctor-ish lines with super(')
# GameRenderer.render 方法签名
b = body_of(r'class GameRenderer[^{]*\{')
for ln in b.splitlines():
    s = ln.strip()
    if 'public void render(' in s or 'void render(' in s:
        print('GameRenderer.render:', s[:140])
