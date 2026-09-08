# -*- coding: utf-8 -*-
import re
txt = open('/home/linux/rhrender/apis_all.java', encoding='utf-8', errors='replace').read()

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

b = body_of(r'class LevelChunk[^{]*\{')
print('LevelChunk len', len(b))
for ln in b.splitlines():
    s = ln.strip()
    if 'setBlockState' in s or 'removeBlockEntity' in s or 'public void setUnsaved' in s:
        print('  ', s[:170])

b2 = body_of(r'class ServerChunkCache[^{]*\{')
print('ServerChunkCache len', len(b2))
for ln in b2.splitlines():
    s = ln.strip()
    if 'getChunk(' in s and 'public' in s:
        print('  ', s[:170])
