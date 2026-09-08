# -*- coding: utf-8 -*-
import re
txt = open('/home/linux/rhrender/apis_all.java', encoding='utf-8', errors='replace').read()
pat = r'.{0,90}(ADVANCE_TIME|ADVANCE_WEATHER|GameRules\.[A-Z_]+).{0,110}'
ms = re.findall(pat, txt)
seen = 0
for m2 in re.findall(r'.{0,80}(?:ADVANCE_TIME|ADVANCE_WEATHER).{0,120}', txt):
    print('...', m2[:200])
    seen += 1
    if seen > 6:
        break
