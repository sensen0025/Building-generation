import numpy as np

OUT = '/home/linux/redstone_ai/dataset/v2'
ss = np.load(f'{OUT}/ss64.npy', mmap_mode='r')
st = np.load(f'{OUT}/ss64_state.npy', mmap_mode='r')
N = ss.shape[0]
NC = 128

cls_stated = np.zeros(NC, dtype=np.int64)
cls_total = np.zeros(NC, dtype=np.int64)
for i in range(0, N, 400):
    v = np.asarray(ss[i:i+400]).reshape(-1).astype(np.int64)
    s = np.asarray(st[i:i+400]).reshape(-1)
    nz = v > 0
    cls_total += np.bincount(v[nz], minlength=NC)[:NC]
    cls_stated += np.bincount(v[(nz) & (s != 0)], minlength=NC)[:NC]

state_cls = [int(c) for c in range(1, NC) if cls_stated[c] > 0]
print(f'有 state 的类 ({len(state_cls)}): {state_cls}')
with open(f'{OUT}/state_class_list.txt', 'w') as f:
    f.write(','.join(str(c) for c in state_cls))

# 汇总: stated cells 覆盖
print('--- 各类 stated 占比 (前 20) ---')
order = sorted(range(1, NC), key=lambda c: -cls_stated[c])[:20]
for c in order:
    if cls_total[c] > 0:
        print(f'  cls {c}: cells={cls_total[c]:,} stated={cls_stated[c]:,} ({100.0*cls_stated[c]/cls_total[c]:.1f}%)')
tot = int(cls_total.sum()); stt = int(cls_stated.sum())
print(f'TOTAL: cells={tot:,} stated={stt:,} ({100.0*stt/tot:.2f}%)')
