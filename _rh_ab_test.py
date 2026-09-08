# -*- coding: utf-8 -*-
"""A/B 远程脚本(无 paramiko): 换 v2compressed 数据 -> 重启 -> 渲染 idx -> 本地下载测 sky."""
import sys, io, time, os, subprocess, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RM = '/home/linux/redstone_ai/render_mc'

def sh(cmd, timeout=120):
    r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout, r.stderr

def skyf_local(p):
    import numpy as np
    from PIL import Image
    SKY = np.array([177., 205., 255.])
    im = np.asarray(Image.open(p).convert('RGB')).astype(np.float32)
    d = np.abs(im - SKY).mean(2)
    return (d < 25).mean() * 100

# 备份新数据 -> 换 v2compressed (保留新数据在 .newdata)
sh("pgrep -f 'KnotCli[ e]nt' | xargs -r kill; sleep 3")
o, e = sh(f"cd {RM} && cp builds.rhb builds.rhb.newdata && cp builds.rhb.v2compressed builds.rhb && ls -la builds.rhb")
print('swapped:', o.strip())
o, e = sh("cd /home/linux/rhrender/mc-render && setsid nohup bash /home/linux/rhrender/launch.sh > logs/launch.out 2>&1 & echo launched", timeout=60)
print(o.strip())

# 等客户端
up = False
for i in range(150):
    time.sleep(2)
    o, _ = sh("grep -c 'HTTP on 8917' /home/linux/rhrender/mc-render/logs/latest.log")
    if o.strip() and int(o.strip()) > 0:
        up = True
        break
print('client up:', up)
time.sleep(8)

def post(path, timeout=300):
    o, e = sh(f"curl -s -X POST 'http://127.0.0.1:8917{path}' --max-time {timeout}", timeout=timeout+30)
    return o

print('open:', post('/open'))
for idx in [250, 600, 2200]:
    print('render', idx, ':', post(f'/render?idx={idx}')[:150])
    time.sleep(1.5)
    o, _ = sh(f'ls -la {RM}/out/bld_{idx:04d}.png 2>/dev/null')
    print(' ', o.strip())
sh("pgrep -f 'KnotCli[ e]nt' | xargs -r kill; sleep 1")
o, e = sh(f"cd {RM} && cp builds.rhb.newdata builds.rhb && ls -la builds.rhb")
print('restored:', o.strip())
print('DONE')
