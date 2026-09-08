# -*- coding: utf-8 -*-
import re
txt = open('/home/linux/rhrender/apis_all.java', encoding='utf-8', errors='replace').read()
m = re.search(r'class GameRules[^{]*\{', txt)
if m:
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
    body = txt[start:i]
    for ln in body.splitlines():
        s = ln.strip()
        if ('DAYLIGHT' in s or 'WEATHER' in s or 'public static final GameRule' in s) and len(s) < 180:
            print(s[:170])
else:
    print('GameRules class not found')
