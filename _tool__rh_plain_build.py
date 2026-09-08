# -*- coding: utf-8 -*-
"""纯 javac 构建(不 remap): 直接引用 named 类."""
import os, sys, glob

JBR = '/snap/pycharm-community/662/jbr'
MC = '/home/linux/Downloads/.minecraft'
NAMED = glob.glob('/home/linux/.gradle/caches/fabric-loom/minecraftMaven/net/minecraft/minecraft-merged/1.21.11-*/minecraft-merged-1.21.11-*.jar')[0]
LOADER = f'{MC}/libraries/net/fabricmc/fabric-loader/0.19.5/fabric-loader-0.19.5.jar'
GSON = glob.glob(f'{MC}/libraries/com/google/code/gson/gson/*/gson-*.jar')[-1]

print('NAMED:', NAMED)
print('LOADER:', LOADER)
print('GSON:', GSON)

SRC = '/home/linux/rhrender/src/main/java'
OUT = '/home/linux/rhrender/plainbuild'
os.makedirs(f'{OUT}/classes', exist_ok=True)

srcs = []
for root, dirs, files in os.walk(SRC):
    for f in files:
        if f.endswith('.java'):
            srcs.append(os.path.join(root, f))
print('sources:', srcs)

# 完整库类路径(与 launch.sh 一致) + named jar + loader + gson/jspec
import json
ref = json.load(open('/home/linux/Downloads/.minecraft/versions/26.2-Fabric/26.2-Fabric.json'))
libs = []
for lib in ref['libraries']:
    name = lib['name']
    parts = name.split(':')
    if len(parts) < 3:
        continue
    if ':natives' in parts[-1] or 'natives-' in parts[-1]:
        continue
    group, art, ver = parts[0], parts[1], parts[2]
    cls = parts[3] if len(parts) > 3 else None
    fname = f'{art}-{ver}' + (f'-{cls}' if cls else '') + '.jar'
    p = f"{MC}/libraries/{group.replace('.', '/')}/{art}/{ver}/{fname}"
    if os.path.exists(p):
        libs.append(p)
print('lib jars:', len(libs))
JSPEC = glob.glob(f'{MC}/libraries/org/jspecify/jspecify/*/jspecify-*.jar')
cp = os.pathsep.join([NAMED, LOADER, GSON] + libs + JSPEC)
rc = os.system(f'{JBR}/bin/javac -encoding UTF-8 -cp "{cp}" -d {OUT}/classes {" ".join(srcs)} 2>&1')
print('javac rc=', rc)
if rc != 0:
    sys.exit(rc)

# 打包: fabric.mod.json 在 jar 根
import shutil, zipfile
res = '/home/linux/rhrender/src/main/resources'
shutil.copy(f'{res}/fabric.mod.json', f'{OUT}/fabric.mod.json')
jar = f'{OUT}/rhrender-plain.jar'
if os.path.exists(jar):
    os.remove(jar)
with zipfile.ZipFile(jar, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(f'{OUT}/fabric.mod.json', 'fabric.mod.json')
    for root, dirs, files in os.walk(f'{OUT}/classes'):
        for f in files:
            fp = os.path.join(root, f)
            arc = os.path.relpath(fp, f'{OUT}/classes')
            z.write(fp, arc)
print('jar written:', jar, os.path.getsize(jar))
