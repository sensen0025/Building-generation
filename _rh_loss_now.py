import re, statistics
LOG = '/home/linux/redstone_ai/TRELLIS/outputs/ss_flow_mc_img_v3/log.txt'
loss, lr = {}, {}
for line in open(LOG):
    try:
        s = int(line.split(':')[0])
        m = re.search(r'"loss": ([0-9.e-]+)', line)
        if m: loss[s] = float(m.group(1))
        m2 = re.search(r'"lr": ([0-9.e-]+)', line)
        if m2: lr[s] = float(m2.group(1))
    except Exception: pass
print('logged steps:', len(loss))
# 最近 2000 步平均 + 全程分桶
last = max(loss)
w = [v for k, v in loss.items() if k > last - 2000]
print(f'last {len(w)} steps mean={statistics.mean(w):.4f}')
for b0 in [0, 5000, 10000, 20000, 30000]:
    vals = [v for k, v in loss.items() if b0 <= k < b0 + 5000]
    if vals: print(f'{b0}-{b0+5000}: mean={statistics.mean(vals):.4f}')
if lr:
    print('lr now:', f'{max(lr):.2e}')
