# -*- coding: utf-8 -*-
import zipfile, glob, os, time
jar = '/home/linux/rhrender/mc-render/mods/rhrender.jar'
print('jar mtime', time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(jar))), 'size', os.path.getsize(jar))
src = '/home/linux/rhrender/build/libs/rhrender-0.1.0.jar'
print('src mtime', time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(src))))
z = zipfile.ZipFile(jar)
cls = [n for n in z.namelist() if n.endswith('RhRenderClient.class')]
print('has class:', cls)
data = z.read(cls[0]) if cls else b''
print('dayTime= in class:', b'dayTime=' in data)
print('ADVANCE_TIME in class:', b'ADVANCE_TIME' in data)
