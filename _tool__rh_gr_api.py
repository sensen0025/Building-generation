# -*- coding: utf-8 -*-
import re
txt = open('/home/linux/rhrender/apis_all.java', encoding='utf-8', errors='replace').read()
for m2 in re.findall(r'.{0,50}getGameRules\(\)\s*\.\s*(\w+)\([^)]*\).{0,80}', txt)[:12]:
    print('usage:', m2[:160])
print('--- methods of GameRules ---')
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
        if ('getRule' in s or 'setRule' in s or 'GameRuleKey' in s or 'public GameRules' in s) and len(s) < 200:
            print('  ', s[:190])
