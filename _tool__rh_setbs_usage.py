# -*- coding: utf-8 -*-
import re
txt = open('/home/linux/rhrender/apis_all.java', encoding='utf-8', errors='replace').read()
for m in re.finditer(r'\.setBlockState\(', txt):
    i = m.start()
    seg = txt[max(0, i - 120):i + 140].replace('\n', ' ')
    if 'chunk' in seg.lower() or 'ChunkAccess' in seg:
        print('...', seg[:240])
        print()
