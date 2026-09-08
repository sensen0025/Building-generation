# -*- coding: utf-8 -*-
import urllib.request, re

def get(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8', errors='replace')

for a in ['filament', 'intermediary', 'intermediary-forwarder']:
    try:
        h = get(f'https://maven.fabricmc.net/net/fabricmc/{a}/maven-metadata.xml')
        vs = re.findall(r'<version>([^<]+)</version>', h)
        print(f'== {a}: last 6: {vs[-6:]} total={len(vs)}')
    except Exception as e:
        print(a, 'err', e)

# loom jar 内找 filament/mappings DSL 提示
try:
    import zipfile, io
    ur = urllib.request.urlopen('https://maven.fabricmc.net/net/fabricmc/fabric-loom/1.17.20/fabric-loom-1.17.20.jar', timeout=60)
    data = ur.read()
    z = zipfile.ZipFile(io.BytesIO(data))
    hits = set()
    for n in z.namelist():
        if n.endswith('.class') and ('Mappings' in n or 'Dependency' in n or 'Extension' in n):
            hits.add(n)
    print('loom classes w/ Mappings/Dependency/Extension:', sorted(hits)[:20])
    # 全 jar 搜字符串 filament
    blob = data
    for kw in [b'filament', b'FilamentMappings', b'officialMojangMappings', b'loom.yarn', b'loom.official']:
        print(kw, kw in blob)
except Exception as e:
    print('loom jar fetch err', e)
