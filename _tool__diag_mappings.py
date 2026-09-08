# -*- coding: utf-8 -*-
"""诊断 mojang mappings 下载路径."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import urllib.request

def get(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

try:
    m = json.loads(get('https://piston-meta.mojang.com/mc/game/version_manifest_v2.json'))
    v = next(x for x in m['versions'] if x['id'] == '26.2')
    vj = json.loads(get(v['url']))
    cm = vj['downloads']['client_mappings']
    print('mojang client_mappings:', cm['url'], cm['size'])
    try:
        head = get(cm['url'], timeout=25)
        print('piston-data mappings OK size', len(head))
    except Exception as e:
        print('piston-data mappings FAIL:', e)
except Exception as e:
    print('manifest fail:', e)

# BMCLAPI 镜像
try:
    b = json.loads(get('https://bmclapi2.bangbang93.com/version/26.2/json'))
    cmb = b['downloads']['client_mappings']
    print('bmclapi client_mappings:', cmb['url'], cmb['size'])
    try:
        headb = get(cmb['url'], timeout=25)
        print('bmclapi mappings OK size', len(headb))
    except Exception as e:
        print('bmclapi mappings FAIL:', e)
except Exception as e:
    print('bmclapi fail:', e)
