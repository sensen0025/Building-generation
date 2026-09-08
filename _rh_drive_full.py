
import json, time, urllib.request, os
def post(path, timeout=300):
    req = urllib.request.Request('http://127.0.0.1:8917' + path, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())
print('open:', post('/open', timeout=300), flush=True)
IDXS = [6, 17, 46, 250, 500]
OUT = '/home/linux/redstone_ai/render_mc/out'
for idx in IDXS:
    f = os.path.join(OUT, 'bld_%04d.png' % idx)
    if os.path.exists(f): os.remove(f)
    try:
        print(idx, post('/render?idx=%d' % idx, timeout=300), flush=True)
    except Exception as e:
        print(idx, 'ERR', repr(e)[:150], flush=True)
    time.sleep(0.3)
print('DRIVE DONE', flush=True)
