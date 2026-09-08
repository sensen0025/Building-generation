# -*- coding: utf-8 -*-
"""第二轮抽取: StateHolder/Registry/Identifier/Entity/Player/BlockState 真实签名."""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SRC = '/home/linux/rhrender/apis_all.java'
txt = open(SRC, encoding='utf-8', errors='replace').read()

def find_section(pat, label):
    m = re.search(pat, txt)
    if not m:
        print(f'### {label}: NOT FOUND')
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
    body = txt[start:i + 1]
    print(f'### {label}: {len(body)} chars')
    return body

def dump(body, kws, limit=12):
    for kw in kws:
        hits = [ln.strip() for ln in body.splitlines() if kw in ln]
        for h in hits[:limit]:
            print('   ', h[:230])
        print(f'   --- ({len(hits)} hits "{kw}")')
    print()

b = find_section(r'(?:public |abstract |final )*class StateHolder[^{]*\{', 'StateHolder')
if b:
    dump(b, ['public Set<Property<?>> getProperties', 'public S setValue', 'getPossibleValues', 'public abstract class Property'])

b = find_section(r'(?:public |abstract |final )*class BlockState[^{]*\{', 'BlockState2')
if b:
    dump(b, ['getBlock', 'getValue', 'public', 'canOcclude'])

b = find_section(r'(?:public |abstract )*interface Registry[^{]*\{', 'Registry')
if b:
    dump(b, ['get(', 'getValue', 'ResourceKey', 'byNameCodec', 'public interface'])

b = find_section(r'(?:public |abstract |final )*class Identifier[^{]*\{', 'Identifier')
if b:
    dump(b, ['public static Identifier', 'parse', 'of(', 'withDefaultNamespace'])

b = find_section(r'(?:public |abstract )*class BuiltInRegistries[^{]*\{', 'BuiltInReg')
if b:
    dump(b, ['BLOCK', 'public static'])

b = find_section(r'abstract class Entity[^{]*\{', 'Entity')
if b:
    dump(b, ['public void setPos', 'setPos(', 'public void setYRot', 'public void setXRot', 'setYRot', 'getEyeHeight', 'teleportTo', 'public void moveTo'], 10)

b = find_section(r'class Player extends[^{]*\{', 'Player')
if b:
    dump(b, ['getEyeHeight', 'public'], 6)

# Minecraft level 字段与 execute
b = find_section(r'class Minecraft[^{]*\{', 'Minecraft2')
if b:
    dump(b, ['ClientLevel level', 'level;', 'public ClientLevel', 'getClientLevel', 'public void tell', 'setScreen', 'public void disconnect'], 8)

# Blocks 常用
b = find_section(r'class Blocks[^{]*\{', 'Blocks')
if b:
    dump(b, ['public static final Block STONE', 'AIR', 'GLASS'], 4)

# 全局方法存在性(全文件)
print('### global existence checks')
for kw in ['public void setPos(double', 'public void setYRot(', 'public void setXRot(', 'public float getEyeHeight(', 'teleportTo(double', 'public static BlockState parseForBlock', 'Blocks.STONE']:
    print(kw, '->', len(re.findall(re.escape(kw), txt)))
