# -*- coding: utf-8 -*-
import sys, io, struct
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
p = '/home/linux/redstone_ai/render_mc/builds.rhb'
f = open(p, 'rb')
head = f.read(16)
print('size:', __import__('os').path.getsize(p))
print('head bytes:', head)
magic, np = struct.unpack('<II', head[:8])
print('magic', hex(magic), 'np', np)
f.close()
