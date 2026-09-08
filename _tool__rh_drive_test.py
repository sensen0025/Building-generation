# -*- coding: utf-8 -*-
"""远端驱动: 启动客户端并测试渲染几个样本."""
import json, time, urllib.request, os, sys

BASE = 'http://127.0.0.1:8917'

def post(path, timeout=180):
    req = urllib.request.Request(BASE + path, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())

# 等 ping
for _ in range(120):
    try:
        print('ping:', get('/ping'))
        break
    except Exception as e:
        time.sleep(2)
else:
    print('client never up')
    sys.exit(1)

print('open:', post('/open', timeout=200))
print('status:', get('/status'))

for idx in [0, 1, 100, 4]:
    try:
        r = post(f'/render?idx={idx}', timeout=120)
        print('render', idx, r)
    except Exception as e:
        print('render', idx, 'ERR', e)
    time.sleep(1)
print('status:', get('/status'))
