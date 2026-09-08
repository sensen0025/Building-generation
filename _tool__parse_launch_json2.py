# -*- coding: utf-8 -*-
import json, os
V = '/home/linux/Downloads/.minecraft/versions/26.2-Fabric/26.2-Fabric.json'
d = json.load(open(V))
print('mainClass:', d.get('mainClass'))
print('assetIndex:', json.dumps(d.get('assetIndex', {})))
print('--- libraries (%d) ---' % len(d['libraries']))
for lib in d['libraries']:
    name = lib['name']
    if 'natives' in json.dumps(lib.get('downloads', {})):
        name += ' [NATIVES]'
    print(' ', name)
print('--- arguments.game ---')
for a in d.get('arguments', {}).get('game', []):
    if isinstance(a, str):
        print('  ARG', a)
    else:
        print('  RULE', json.dumps(a)[:250])
print('--- arguments.jvm ---')
for a in d.get('arguments', {}).get('jvm', []):
    if isinstance(a, str):
        print('  JVM', a)
    else:
        print('  RULE', json.dumps(a)[:250])
print('--- downloads.client ---')
print(json.dumps(d.get('downloads', {}).get('client', {}), indent=1)[:300])
