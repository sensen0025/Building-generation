import numpy as np
import time

BASE = '/home/linux/redstone_ai/dataset'
G = 96
meta = np.load(f'{BASE}/voxel96_meta.npz')
N = int(meta['n'])
print(f'N={N} total={int(meta["total"])}')

flat_id = np.memmap(f'{BASE}/voxel96_id.dat', dtype=np.uint8, mode='r', shape=(N * G * G * G,))
flat_st = np.memmap(f'{BASE}/voxel96_state.dat', dtype=np.uint16, mode='r', shape=(N * G * G * G,))
solid = np.load(f'{BASE}/solid96.npy', mmap_mode='r')
off = np.load(f'{BASE}/solid96_off.npy')
print(f'offsets len={len(off)} solid total={len(solid)}')

vocab = [l.strip() for l in open(f'{BASE}/BLOCK_VOCAB_FULL.txt', encoding='utf-8') if l.strip()]
top200 = meta['top200'].astype(int)
id2name = ['minecraft:air'] * 203
for cls in range(1, 201):
    orig = int(top200[cls - 1])
    id2name[cls] = vocab[orig] if 0 <= orig < len(vocab) else '?'
id2name[201] = 'minecraft:other(201)'
id2name[202] = 'minecraft:other(202)'

total = 0
stated = 0
id_total = np.zeros(203, dtype=np.int64)
id_stated = np.zeros(203, dtype=np.int64)
id_uniq_states = {i: set() for i in range(203)}

t0 = time.time()
for i in range(N):
    s = solid[off[i]:off[i + 1]]
    if len(s) == 0:
        continue
    ids = flat_id[s]
    sts = flat_st[s]
    total += len(ids)
    nz = sts != 0
    stated += int(nz.sum())
    ui, inv = np.unique(ids, return_inverse=True)
    id_total[ui] += np.bincount(inv, minlength=len(ui))
    if nz.any():
        id_stated[ui] += np.bincount(inv[nz], minlength=len(ui))
        pairs = np.unique(np.stack([ids[nz], sts[nz]], axis=1), axis=0)
        for uid, ust in pairs:
            id_uniq_states[int(uid)].add(int(ust))
    if (i + 1) % 2000 == 0:
        print(f'  {i+1}/{N} elapsed={time.time()-t0:.0f}s', flush=True)

print(f'\n=== TOTAL: solid cells={total:,} with-state={stated:,} ({100.0*stated/total:.2f}%) ===')

print('\n=== 按方块类别: total, stated, stated%, unique-state-count ===')
rows = []
for c in range(203):
    t = int(id_total[c])
    if t == 0:
        continue
    s = int(id_stated[c])
    u = len(id_uniq_states[c])
    rows.append((c, id2name[c], t, s, 100.0 * s / t, u))
rows.sort(key=lambda r: -r[4])  # by stated% desc

print(f'{"id":>4} {"block":<35} {"total":>10} {"stated":>10} {"stated%":>8} {"uniqSt":>7}')
for c, nm, t, s, pct, u in rows:
    print(f'{c:>4} {nm:<35} {t:>10,} {s:>10,} {pct:>7.2f}% {u:>7}')

# aggregate: how many distinct blocks have ANY state, and their share of cells
any_state_ids = [r for r in rows if r[4] > 0]
cell_any_state = sum(r[2] for r in any_state_ids)
print(f'\n=== 汇总 ===')
print(f'有非默认state的方块类别数: {len(any_state_ids)} / {len(rows)}')
print(f'这些类别的cell总数: {cell_any_state:,} / {total:,} = {100.0*cell_any_state/total:.2f}%')
print(f'其中 state!=0 的cell: {stated:,} = {100.0*stated/total:.2f}% of all solid')

# top state-heavy classes by cell share
print('\n=== state 复杂方块按 cell 占比排序 (top 15) ===')
for c, nm, t, s, pct, u in sorted(any_state_ids, key=lambda r: -r[2])[:15]:
    print(f'  {nm:<35} cells={t:>10,} ({100.0*t/total:5.2f}%) stated={100.0*s/t:5.1f}% uniqSt={u}')
