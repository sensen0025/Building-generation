# -*- coding: utf-8 -*-
import json, urllib.request

def get(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

m = json.loads(get('https://piston-meta.mojang.com/mc/game/version_manifest_v2.json'))
ids = [x['id'] for x in m['versions']]
print('recent ids:', ids[-25:])
for vid in ['26.2', '26.1', '1.21.11', '1.21.8', '1.21.9']:
    try:
        v = next(x for x in m['versions'] if x['id'] == vid)
        vj = json.loads(get(v['url']))
        print(vid, 'downloads keys:', list(vj.get('downloads', {}).keys()))
    except StopIteration:
        print(vid, 'NOT IN MANIFEST')
    except Exception as e:
        print(vid, 'err', e)
