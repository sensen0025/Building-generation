# -*- coding: utf-8 -*-
"""组装渲染实例的 MC 客户端启动命令 -> /home/linux/rhrender/launch.sh"""
import json, os

MC = '/home/linux/Downloads/.minecraft'
VER = f'{MC}/versions/26.2-Fabric'
jf = f'{VER}/26.2-Fabric.json'
d = json.load(open(jf))

def lib_path(name):
    parts = name.split(':')
    if len(parts) >= 3:
        group, art, ver = parts[0], parts[1], parts[2]
    else:
        return None
    # classifier?
    cls = parts[3] if len(parts) > 3 else None
    fname = f'{art}-{ver}' + (f'-{cls}' if cls else '') + '.jar'
    p = f"{MC}/libraries/{group.replace('.', '/')}/{art}/{ver}/{fname}"
    return p if os.path.exists(p) else None

cp = []
missing = []
for lib in d['libraries']:
    name = lib['name']
    if any(c in name for c in [':natives-', ':linux-aarch_64', ':osx-', ':windows-']) and ':natives' in name:
        continue
    if name == 'net.fabricmc:intermediary:0.0.0':
        # HMCL placeholder: 尝试实际 intermediary
        import glob
        g = glob.glob(f'{MC}/libraries/net/fabricmc/intermediary/*/intermediary-*.jar')
        if g:
            cp.append(g[-1])
        continue
    p = lib_path(name)
    if p:
        cp.append(p)
    else:
        missing.append(name)

client_jar = f'{VER}/26.2-Fabric.jar'
cp.append(client_jar)
cp = list(dict.fromkeys(cp))  # 去重保序

nat = f'{VER}/natives-linux-x86_64'
game_dir = '/home/linux/rhrender/mc-render'

lines = []
lines.append('#!/bin/bash')
lines.append(f'export DISPLAY=:0')
lines.append(f'export XAUTHORITY=/run/user/1000/Xauthority')
lines.append('cd ' + game_dir)
lines.append('export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64')
jvm = [
    '-Xmx4G',
    '-Xms1G',
    '--sun-misc-unsafe-memory-access=allow',
    '--enable-native-access=ALL-UNNAMED',
    f'-Djava.library.path={nat}/java',
    f'-Djna.tmpdir={nat}/jna',
    f'-Dorg.lwjgl.system.SharedLibraryExtractPath={nat}/lwjgl',
    f'-Dio.netty.native.workdir={nat}/netty',
    '-Dminecraft.launcher.brand=HMCL',
    '-Dminecraft.launcher.version=3.16.3',
    '-Dlog4j.configurationFile=' + game_dir + '/log4j2.xml',
]
game = [
    '--username', 'Renderer',
    '--version', '26.2-Fabric',
    '--gameDir', game_dir,
    '--assetsDir', f'{MC}/assets',
    '--assetIndex', '32',
    '--uuid', '00000000-0000-0000-0000-000000000000',
    '--accessToken', '0',
    '--clientId', '0',
    '--xuid', '0',
    '--versionType', 'release',
]
cl = ':'.join(cp)
lines.append(f'exec $JAVA_HOME/bin/java {" ".join(jvm)} -cp "{cl}" net.fabricmc.loader.impl.launch.knot.KnotClient {" ".join(game)}')
open('/home/linux/rhrender/launch.sh', 'w').write('\n'.join(lines) + '\n')
print('launch.sh written')
print('classpath entries:', len(cp))
print('missing libs:', missing)
import subprocess
os.chmod('/home/linux/rhrender/launch.sh', 0o755)
