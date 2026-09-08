# -*- coding: utf-8 -*-
"""验证 1.21.11 工具链要素: piston-data 可达性 / loader / yarn."""
import sys, io, json, urllib.request, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def get(url, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

# piston-data 可达性(用 1.21.11 client jar url 探)
m = json.loads(get('https://piston-meta.mojang.com/mc/game/version_manifest_v2.json'))
v = next(x for x in m['versions'] if x['id'] == '1.21.11')
vj = json.loads(get(v['url']))
print('1.21.11 downloads keys:', list(vj['downloads'].keys()))
cu = vj['downloads']['client']['url']
try:
    req = urllib.request.Request(cu, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=25) as r:
        print('piston-data client HEAD', r.status, r.headers.get('Content-Length'))
except Exception as e:
    print('piston-data client FAIL', e)
mu = vj['downloads']['client_mappings']['url']
try:
    req = urllib.request.Request(mu, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=25) as r:
        print('piston-data mappings HEAD', r.status, r.headers.get('Content-Length'))
except Exception as e:
    print('piston-data mappings FAIL', e)

# loader for 1.21.11
lm = json.loads(get('https://meta.fabricmc.net/v2/versions/loader/1.21.11'))
print('loader 1.21.11 latest:', lm[0]['loader']['version'], 'stable', lm[0]['loader']['stable'])
# yarn 1.21.11
ym = json.loads(get('https://meta.fabricmc.net/v2/versions/yarn/1.21.11'))
print('yarn 1.21.11 latest:', ym[0]['version'])
# bmclapi client jar 也可达?
try:
    req = urllib.request.Request('https://bmclapi2.bangbang93.com/version/1.21.11/client', method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=25) as r:
        print('bmclapi client HEAD', r.status, r.headers.get('Content-Length'))
except Exception as e:
    print('bmclapi client FAIL', e)
