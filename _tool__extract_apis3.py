# -*- coding: utf-8 -*-
"""第三轮: Property/StateHolder/BlockState 声明; Registry getValue; Identifier 构造; 用法示例."""
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

b = body_of(r'(?:public |abstract |final )*class Property[^{]*\{')
print('Property section', len(b))
if b:
    for ln in b.splitlines():
        s = ln.strip()
        if ('getValue(' in s or 'getPossibleValues' in s or 'public abstract' in s or 'class Property' in s) and len(s) < 200:
            print('  ', s[:190])

b = body_of(r'(?:public |abstract |final )*class StateHolder[^{]*\{')
print('StateHolder section', len(b))
if b:
    for ln in b.splitlines():
        s = ln.strip()
        if ('public Set<Property' in s or 'setValue(Property' in s or 'getProperties(' in s or 'getValues(' in s or 'create(' in s and 'property' in s.lower()) and len(s) < 200:
            print('  ', s[:190])

b = body_of(r'(?:public |abstract |final )*class BlockState[^{]*\{')
print('BlockState section', len(b))
if b:
    for ln in b.splitlines():
        s = ln.strip()
        if ('getBlock(' in s or 'public Block' in s or 'getValue(' in s or 'public static BlockState' in s) and len(s) < 200:
            print('  ', s[:190])

# Registry 与 DefaultedRegistry 中 by-id 查询
for cls in ['interface Registry', 'class DefaultedMappedRegistry', 'class MappedRegistry']:
    b = body_of(re.escape(cls).replace(r'\ ', ' ') + r'[^{]*\{')
    if b:
        hits = [ln.strip() for ln in b.splitlines() if ('getValue(' in ln or 'public T get(' in ln or 'Identifier identifier' in ln or 'Identifier id' in ln) and len(ln) < 200]
        print(f'== {cls}:')
        for h in hits[:10]:
            print('   ', h[:190])

# 用法示例: 按 id 查方块
import re as _re
for pat in [r'.{0,80}BuiltInRegistries\.BLOCK\.(getValue|get)\([^)]*\).{0,60}', r'.{0,60}Identifier\.(parse|of|withDefaultNamespace)\([^)]*\).{0,80}']:
    ms = _re.findall(pat, txt)
    print('example count', len(ms))
    for m2 in ms[:8]:
        print('   ...', m2[:190])

# Minecraft level 字段名
b = body_of(r'class Minecraft[^{]*\{')
print('Minecraft level field:')
for ln in b.splitlines():
    s = ln.strip()
    if ('ClientLevel' in s and ('level' in s) and ('=' in s or ';' in s)) or s.startswith('public ClientLevel'):
        print('  ', s[:160])
    if re.search(r'\blevel\b;', s) or s.startswith('ClientLevel level') or 'ClientLevel level =' in s:
        print('  F:', s[:160])
