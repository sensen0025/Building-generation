# -*- coding: utf-8 -*-
"""准备 1.21.11 渲染运行时: 客户端jar / 资产 / void世界 / launch.sh"""
import os, sys, json, glob
import urllib.request
import subprocess

MC = '/home/linux/Downloads/.minecraft'
VER = '1.21.11'
VERDIR = f'{MC}/versions/{VER}'
os.makedirs(VERDIR, exist_ok=True)

def get(url, timeout=60, binary=False):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def dl(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000000:
        print('exists', dest, os.path.getsize(dest))
        return
    tmp = dest + '.part'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    print('downloaded', dest, os.path.getsize(dest))

# 1) version json + 客户端 jar (bmclapi 优先)
try:
    vj = json.loads(get('https://bmclapi2.bangbang93.com/version/%s/json' % VER))
    print('bmclapi version json ok, assetIndex:', vj.get('assetIndex', {}).get('id'))
except Exception as e:
    print('bmclapi json fail', e)
    m = json.loads(get('https://piston-meta.mojang.com/mc/game/version_manifest_v2.json'))
    v = next(x for x in m['versions'] if x['id'] == VER)
    vj = json.loads(get(v['url']))

aid = vj['assetIndex']['id']
dl(f'https://bmclapi2.bangbang93.com/version/{VER}/client', f'{VERDIR}/{VER}.jar')

# 2) 资产索引 + 缺失 objects
idxp = f'{MC}/assets/indexes/{aid}.json'
if not os.path.exists(idxp):
    dl(vj['assetIndex']['url'], idxp)
idx = json.load(open(idxp))
objs = idx.get('objects', {})
have = 0
need = []
for k, o in objs.items():
    h = o['hash']
    p = f"{MC}/assets/objects/{h[:2]}/{h}"
    if os.path.exists(p):
        have += 1
    else:
        need.append((h, o['size']))
print(f'assets: have={have} need={len(need)} total={len(objs)}')
missing_bytes = sum(s for _, s in need)
print('missing MB:', round(missing_bytes / 1e6, 1))
if need:
    os.makedirs(f'{MC}/assets/objects', exist_ok=True)
    for i, (h, s) in enumerate(need[:5000]):
        p = f"{MC}/assets/objects/{h[:2]}/{h}"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            dl(f'https://bmclapi2.bangbang93.com/assets/{h}', p)
        except Exception as e:
            print('asset fail', h, e)
    print('asset fetch done')

# 3) 1.21.11 服务端 -> 生成 void 世界
os.makedirs('/home/linux/rhrender/mcserver12111', exist_ok=True)
os.chdir('/home/linux/rhrender/mcserver12111')
if not os.path.exists('server.jar'):
    dl(f'https://bmclapi2.bangbang93.com/version/{VER}/server', 'server.jar')
open('eula.txt', 'w').write('eula=true\n')
open('server.properties', 'w').write('''online-mode=false
level-type=minecraft:flat
generator-settings={"layers":[],"biome":"minecraft:the_void","structure_overrides":[]}
level-name=rhvoid
spawn-protection=0
view-distance=6
simulation-distance=4
max-players=1
motd=rhvoid
sync-chunk-writes=false
''')
print('server files ready')

# 4) 写 launch.sh(1.21.11 版, 复用 libraries)
libs_root = f'{MC}/libraries'
def lib_path(name):
    parts = name.split(':')
    if len(parts) < 3:
        return None
    group, art, ver = parts[0], parts[1], parts[2]
    cls = parts[3] if len(parts) > 3 else None
    fname = f'{art}-{ver}' + (f'-{cls}' if cls else '') + '.jar'
    p = f"{libs_root}/{group.replace('.', '/')}/{art}/{ver}/{fname}"
    return p if os.path.exists(p) else None

# 用 26.2 实例 json 的库列表(版本相同), 替换占位 intermediary
ref = json.load(open('/home/linux/Downloads/.minecraft/versions/26.2-Fabric/26.2-Fabric.json'))
cp = []
for lib in ref['libraries']:
    name = lib['name']
    if ':natives' in name and ':natives' in name.split(':')[-1]:
        continue
    if name == 'net.fabricmc:intermediary:0.0.0':
        continue
    p = lib_path(name)
    if p:
        cp.append(p)
cp.append(f'{VERDIR}/{VER}.jar')
cp = list(dict.fromkeys(cp))
nat = '/home/linux/Downloads/.minecraft/versions/26.2-Fabric/natives-linux-x86_64'
game_dir = '/home/linux/rhrender/mc-render'
jvm = [
    '-Xmx4G', '-Xms1G',
    '--sun-misc-unsafe-memory-access=allow',
    '--enable-native-access=ALL-UNNAMED',
    f'-Djava.library.path={nat}/java',
    f'-Djna.tmpdir={nat}/jna',
    f'-Dorg.lwjgl.system.SharedLibraryExtractPath={nat}/lwjgl',
    f'-Dio.netty.native.workdir={nat}/netty',
    '-Dminecraft.launcher.brand=HMCL', '-Dminecraft.launcher.version=3.16.3',
]
game = ['--username', 'Renderer', '--version', VER, '--gameDir', game_dir,
        '--assetsDir', f'{MC}/assets', '--assetIndex', aid,
        '--uuid', '00000000-0000-0000-0000-000000000000', '--accessToken', '0',
        '--clientId', '0', '--xuid', '0', '--versionType', 'release']
lines = ['#!/bin/bash',
         'export DISPLAY=:0',
         'export XAUTHORITY=/run/user/1000/Xauthority',
         'cd ' + game_dir,
         'export JAVA_HOME=/snap/pycharm-community/662/jbr',
         f'exec $JAVA_HOME/bin/java {" ".join(jvm)} -cp "{":".join(cp)}" net.fabricmc.loader.impl.launch.knot.KnotClient {" ".join(game)}']
open('/home/linux/rhrender/launch.sh', 'w').write('\n'.join(lines) + '\n')
os.chmod('/home/linux/rhrender/launch.sh', 0o755)
print('launch.sh written (1.21.11), assetIndex', aid, 'cp entries', len(cp))
