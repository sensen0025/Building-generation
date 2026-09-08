import re, statistics
losses = {}
for line in open('/home/linux/redstone_ai/TRELLIS/outputs/ss_flow_mc_img_v3/log.txt'):
    try:
        s = int(line.split(':')[0])
        m = re.search(r'"loss": ([0-9.e-]+)', line)
        if m:
            losses[s] = float(m.group(1))
    except Exception:
        pass
buckets = [(0, 10000), (10000, 50000), (50000, 100000), (100000, 150000), (150000, 190000), (190000, 200000)]
print('total logged steps:', len(losses))
for b0, b1 in buckets:
    vals = [v for k, v in losses.items() if b0 <= k < b1]
    if vals:
        print(f'{b0}-{b1}: n={len(vals)} mean={statistics.mean(vals):.4f} min={min(vals):.4f} last5={[round(v,4) for v in vals[-3:]]}')
