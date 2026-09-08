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

b = body_of(r'enum ParticleStatus[^{]*\{')
print('ParticleStatus enum:')
if b:
    for ln in b.splitlines():
        s = ln.strip()
        if len(s) < 120 and s:
            print('  ', s[:110])

b = body_of(r'class NativeImage[^{]*\{')
print('NativeImage write-ish:')
if b:
    for ln in b.splitlines():
        s = ln.strip()
        if ('write' in s or 'save' in s or 'public void' in s or 'public boolean' in s) and len(s) < 160 and ('File' in s or 'Path' in s or 'OutputStream' in s or '(' not in s):
            print('  ', s[:150])
    for ln in b.splitlines():
        s = ln.strip()
        if 'writeFile' in s or 'public void write' in s or 'public void save' in s:
            print('  W:', s[:150])
