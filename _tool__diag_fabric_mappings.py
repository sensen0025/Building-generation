# -*- coding: utf-8 -*-
import json, urllib.request, re

def get(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8', errors='replace')

# 1) fabric maven net/fabricmc 目录
html = get('https://maven.fabricmc.net/net/fabricmc/')
arts = sorted(set(re.findall(r'href="([a-zA-Z0-9_.-]+)/"', html)))
print('artifacts under net/fabricmc:', arts)

# 2) 各映射相关 artifact 的版本列表(尾部)
for a in ['yarn', 'intermediary', 'mappings', 'minecraft-mappings', 'mojang-mappings']:
    try:
        h = get(f'https://maven.fabricmc.net/net/fabricmc/{a}/maven-metadata.xml')
        vs = re.findall(r'<version>([^<]+)</version>', h)
        print(f'{a}: latest={vs[-3:] if vs else "none"} total={len(vs)}')
    except Exception as e:
        print(a, 'err', e)

# 3) meta yarn 支持的游戏版本(最后几个)
y = json.loads(get('https://meta.fabricmc.net/v2/versions/yarn'))
games = sorted(set(x['gameVersion'] for x in y))
print('yarn games tail:', games[-8:])
