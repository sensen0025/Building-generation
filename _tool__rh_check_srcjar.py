# -*- coding: utf-8 -*-
import zipfile, os, time
src = '/home/linux/rhrender/build/libs/rhrender-0.1.0.jar'
print('src mtime', time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(src))), os.path.getsize(src))
z = zipfile.ZipFile(src)
cls = [n for n in z.namelist() if n.endswith('RhRenderClient.class')]
data = z.read(cls[0]) if cls else b''
print('dayTime= in class:', b'dayTime=' in data)
print('ADVANCE_TIME in class:', b'ADVANCE_TIME' in data)
