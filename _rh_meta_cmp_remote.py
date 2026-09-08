
import json
old = json.load(open('/home/linux/redstone_ai/render_mc/builds_meta.v2compressed.json'))
new = json.load(open('/home/linux/redstone_ai/render_mc/builds_meta.full.json'))
for target in [250, 46, 6, 17]:
    o = old['meta'][target]
    n = new['meta'][target]
    print(f'=== idx{target} ===')
    for k in ['n','bbox_min','bbox_max','center','spans','eye','look','dist']:
        ov = o.get(k); nv = n.get(k)
        same = ov == nv
        print(f'  {k}: old={ov} new={nv} {"SAME" if same else "DIFF"}')
