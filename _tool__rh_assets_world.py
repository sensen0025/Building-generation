# -*- coding: utf-8 -*-
"""补拉缺失资产(piston resources) + 生成 1.21.11 void 世界 + 更新存档."""
import os, sys, json
import urllib.request
import subprocess, time

MC = '/home/linux/Downloads/.minecraft'

def get(url, timeout=60):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

# 1) 缺失资产
idx = json.load(open(f'{MC}/assets/indexes/29.json'))
missing = []
for k, o in idx.get('objects', {}).items():
    h = o['hash']
    p = f"{MC}/assets/objects/{h[:2]}/{h}"
    if not os.path.exists(p):
        missing.append((h, k))
print('missing assets:', len(missing))
for h, k in missing:
    p = f"{MC}/assets/objects/{h[:2]}/{h}"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        data = get(f'https://resources.download.minecraft.net/{h[:2]}/{h}')
        open(p, 'wb').write(data)
    except Exception as e:
        print('fail', h, k, e)
print('asset backfill done')

# 2) 1.21.11 void 世界生成
os.chdir('/home/linux/rhrender/mcserver12111')
if os.path.exists('rhvoid'):
    print('world exists, skip gen')
else:
    p = subprocess.Popen(['/snap/pycharm-community/662/jbr/bin/java', '-Xmx3G', '-jar', 'server.jar', 'nogui'],
                         stdin=subprocess.PIPE, stdout=open('gen.log', 'w'), stderr=subprocess.STDOUT)
    time.sleep(80)
    try:
        p.communicate(input=b'stop\n', timeout=30)
    except Exception:
        p.kill()
    print('server stopped; log tail:')
    print(open('gen.log').read()[-1500:])

# 3) 复制到渲染实例 saves
import shutil
dst = '/home/linux/rhrender/mc-render/saves/rhvoid'
if os.path.exists(dst):
    shutil.rmtree(dst)
if os.path.exists('/home/linux/rhrender/mcserver12111/rhvoid'):
    shutil.copytree('/home/linux/rhrender/mcserver12111/rhvoid', dst)
    print('world copied to render saves')
    print(os.listdir(dst))
