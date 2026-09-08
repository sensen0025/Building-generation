# -*- coding: utf-8 -*-
import json
meta = json.load(open('/home/linux/redstone_ai/render_mc/builds_meta.json'))['meta']
for idx in [1790, 1795, 1798, 1800, 1805]:
    m = meta[idx]
    print(idx, 'n', m['n'], 'min', m.get('bbox_min'), 'max', m.get('bbox_max'), 'spans', m.get('spans'),
          'center', [round(x,1) for x in m.get('center', [])] if m.get('center') else None,
          'eye', [round(x,1) for x in m.get('eye', [])] if m.get('eye') else None,
          'dist', round(m.get('dist', 0), 1))
