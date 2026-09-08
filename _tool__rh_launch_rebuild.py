# -*- coding: utf-8 -*-
"""按 1.21.11 真实 libraries 重建 launch.sh; 下载缺失 jar; 重启客户端."""
import os, sys, json, urllib.request, glob, shutil

MC = '/home/linux/Downloads/.minecraft'
VER = '1.21.11'
VERDIR = f'{MC}/versions/{VER}'

def get(url, timeout=120):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def dl(url, dest, timeout=600):
    if os.path.exists(dest) and os.path.getsize(dest) > 100:
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.part'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, 'wb') as f:
            while True:
                c = r.read(1 << 20)
                if not c:
                    break
                f.write(c)
        os.replace(tmp, dest)
        return True
    except Exception as e:
        print('DL FAIL', url, e)
        if os.path.exists(tmp):
            os.remove(tmp)
        return False

# 1) version json
try:
    vj = json.loads(get('https://bmclapi2.bangbang93.com/version/%s/json' % VER))
except Exception:
    vj = None
if vj is None:
    m = json.loads(get('https://piston-meta.mojang.com/mc/game/version_manifest_v2.json'))
    v = next(x for x in m['versions'] if x['id'] == VER)
    vj = json.loads(get(v['url']))
json.dump(vj, open(f'{VERDIR}/{VER}-libs.json', 'w'), indent=1)

# 2) libraries 列表 + 下载缺失
libs = []
missing = []
for lib in vj['libraries']:
    name = lib['name']
    parts = name.split(':')
    if len(parts) < 3:
        continue
    group, art, ver = parts[0], parts[1], parts[2]
    cls = parts[3] if len(parts) > 3 else None
    # 排除非 linux 平台 natives, 保留 linux natives(类路径自动解包)
    if cls and any(k in cls for k in ['natives-macos', 'natives-windows', 'osx-', 'natives-linux-x86']):
        if cls and ('natives-linux-x86' in cls):
            pass
        else:
            continue
    if cls and 'natives-linux' in cls:
        pass  # 保留
    if cls and any(k in cls for k in ['natives-macos', 'natives-windows']):
        continue
    rel = f"{group.replace('.', '/')}/{art}/{ver}/{art}-{ver}" + (f'-{cls}' if cls else '') + '.jar'
    dest = f'{MC}/libraries/{rel}'
    if os.path.exists(dest):
        libs.append(dest)
        continue
    # 找下载 url(官源或 bmclapi)
    url = None
    dl_ent = lib.get('downloads', {})
    if 'artifact' in dl_ent:
        url = dl_ent['artifact'].get('url')
    if not url:
        url = f'https://libraries.minecraft.net/{rel}'
    if not dl(url, dest):
        if not dl(f'https://bmclapi2.bangbang93.com/maven/{rel}', dest):
            missing.append(rel)
    if os.path.exists(dest):
        libs.append(dest)
    else:
        missing.append(rel)
print('libs resolved:', len(libs), 'missing:', missing)

client_jar = f'{VERDIR}/{VER}.jar'
# fabric loader 及其依赖 + intermediary(运行时装在 intermediary 空间)
extra = []
for pat in ['org/ow2/asm/asm', 'org/ow2/asm/asm-analysis', 'org/ow2/asm/asm-commons', 'org/ow2/asm/asm-tree',
            'org/ow2/asm/asm-util', 'net/fabricmc/sponge-mixin', 'net/fabricmc/fabric-loader',
            'net/fabricmc/intermediary']:
    g = glob.glob(f'{MC}/libraries/{pat}/*/*.jar')
    if g:
        extra.append(g[-1])
# 确保 intermediary 1.21.11(上面 glob 可能命中 26.2/1.21.11; 显式指定)
im = f'{MC}/libraries/net/fabricmc/intermediary/1.21.11/intermediary-1.21.11.jar'
if os.path.exists(im) and im not in extra:
    extra.append(im)
extra = list(dict.fromkeys(extra))
cp = libs + extra + [client_jar]
cp = list(dict.fromkeys(cp))
print('extra:', [os.path.basename(e) for e in extra])
print('cp jars:', len(cp))

# 3) launch.sh
nat = '/home/linux/Downloads/.minecraft/versions/26.2-Fabric/natives-linux-x86_64'
game_dir = '/home/linux/rhrender/mc-render'
aid = vj['assetIndex']['id']
jvm = ['-Xmx6G', '-Xms2G', '--sun-misc-unsafe-memory-access=allow', '--enable-native-access=ALL-UNNAMED',
       f'-Djava.library.path={nat}/java', f'-Djna.tmpdir={nat}/jna',
       f'-Dorg.lwjgl.system.SharedLibraryExtractPath={nat}/lwjgl', f'-Dio.netty.native.workdir={nat}/netty',
       '-Dminecraft.launcher.brand=HMCL', '-Dminecraft.launcher.version=3.16.3']
game = ['--username', 'Renderer', '--version', VER, '--gameDir', game_dir, '--assetsDir', f'{MC}/assets',
        '--assetIndex', aid, '--uuid', '00000000-0000-0000-0000-000000000000', '--accessToken', '0',
        '--clientId', '0', '--xuid', '0', '--versionType', 'release',
        '--width', '768', '--height', '768']
lines = ['#!/bin/bash', 'export DISPLAY=:0', 'export XAUTHORITY=/run/user/1000/Xauthority', 'cd ' + game_dir,
         'export JAVA_HOME=/snap/pycharm-community/662/jbr',
         f'exec $JAVA_HOME/bin/java {" ".join(jvm)} -cp "{":".join(cp)}" net.fabricmc.loader.impl.launch.knot.KnotClient {" ".join(game)}']
open('/home/linux/rhrender/launch.sh', 'w').write('\n'.join(lines) + '\n')
os.chmod('/home/linux/rhrender/launch.sh', 0o755)
print('launch.sh rewritten; cp jars:', len(cp), 'assetIndex', aid)
