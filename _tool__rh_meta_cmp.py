# -*- coding: utf-8 -*-
import json
meta = json.load(open('/home/linux/redstone_ai/render_mc/builds_meta.json'))['meta']
for idx in [100, 279, 281, 284, 260]:
    m = meta[idx]
    print(idx, 'n', m['n'], 'bbox_min', m.get('bbox_min'), 'bbox_max', m.get('bbox_max'),
          'spans', m.get('spans'), 'center', m.get('center'), 'dist', round(m.get('dist', 0), 1))
