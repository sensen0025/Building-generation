# -*- coding: utf-8 -*-
"""续训体检: 进度/速率/ETA/loss 趋势/报错."""
import json
import bisect
import datetime
import subprocess

LOG = '/data/redstone_v4/out_v4_rotfix/log.txt'
SUP = '/data/redstone_v4/work/rotfix_supervisor.log'
TOTAL = 50000

v, e = [], []
for l in open(LOG):
    try:
        d = json.loads(l.split(': ', 1)[1])
        v.append(d['loss']['loss'])
        e.append(d['time']['elapsed'])
    except Exception:
        pass
n = len(v)
print('now:', datetime.datetime.now().strftime('%m-%d %H:%M:%S'))
print('=== 进度 / 速率 ===')
print('  step=%d/%d (%.1f%%)  wall=%.2fh' % (n, TOTAL, 100.0 * n / TOTAL, e[-1] / 3600))
rate = (e[-1] - e[0]) / max(n - 1, 1)
print('  平均速率 %.3f s/step (%.2f step/s)' % (rate, 1 / rate))
i = bisect.bisect_left(e, e[-1] - 600)
print('  最近 10 分钟: %d 步 -> %.2f step/s' % (n - i, (n - i) / 600.0))
eta = (TOTAL - n) * rate
print('  ETA: %.1f h -> %s' % (eta / 3600, (datetime.datetime.now() + datetime.timedelta(seconds=eta)).strftime('%m-%d %H:%M')))

print()
print('=== loss 分段均值 ===')
seg = max(1, n // 10)
for a in range(0, n, seg):
    b = v[a:a + seg]
    print('  step %6d-%6d: %.4f' % (a + 1, a + len(b), sum(b) / len(b)))
first, last = sum(v[:seg]) / seg, sum(v[-seg:]) / seg
print('  首段 %.4f -> 末段 %.4f  (%+.1f%%)' % (first, last, 100 * (last / first - 1)))
if n >= 2000:
    print('  最近1000步 %.4f | 之前1000步 %.4f  (%+.1f%%)' % (
        sum(v[-1000:]) / 1000, sum(v[-2000:-1000]) / 1000,
        100 * ((sum(v[-1000:]) / 1000) / (sum(v[-2000:-1000]) / 1000) - 1)))

print()
print('=== 报错检查 ===')
raw = open(LOG, errors='ignore').read()
for k in ('error', 'Error', 'traceback', 'Traceback', 'nan', 'NaN'):
    c = raw.count(k)
    if c:
        print('  命中 %-10s x%d' % (k, c))
print('  error/traceback/nan 合计命中:', sum(raw.count(k) for k in ('error', 'Error', 'Traceback', 'nan', 'NaN')))
print('  守护重启(relaunching) 次数:', open(SUP).read().count('relaunching'))
print('  守护日志尾:', open(SUP).read().strip().splitlines()[-1])
print('  训练进程数:', subprocess.run(['bash', '-c', 'pgrep -fc train.py'], capture_output=True, text=True).stdout.strip())
print('  ckpt:', subprocess.run(['bash', '-c', 'ls /data/redstone_v4/out_v4_rotfix/ckpts/ | wc -l'],
                              capture_output=True, text=True).stdout.strip(), '个')
