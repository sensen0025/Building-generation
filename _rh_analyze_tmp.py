
import numpy as np, json, os
np.set_printoptions(precision=6, suppress=True)
BASE='/home/linux/redstone_ai'
V2=f'{BASE}/dataset/v2'

z = np.load(f'{V2}/ss_latent_mc64_v2.npy', mmap_mode='r')
print('latent shape', z.shape)
n = z.shape[0]
m_bad = z.reshape(-1,8).mean(0); s_bad = z.reshape(-1,8).std(0)
sums = np.zeros(8, dtype=np.float64); sumsq = np.zeros(8, dtype=np.float64)
cnt = 0
for i in range(0, n, 50):
    blk = np.asarray(z[i:i+50], dtype=np.float64)
    sums += blk.sum(axis=(0,2,3,4)); sumsq += (blk**2).sum(axis=(0,2,3,4)); cnt += blk.shape[0]*16**3
m_good = sums/cnt; s_good = np.sqrt(np.clip(sumsq/cnt - m_good**2, 0, None))
print('BAD  mean', m_bad.tolist()); print('BAD  std ', s_bad.tolist())
print('GOOD mean', m_good.tolist()); print('GOOD std ', s_good.tolist())

if os.path.exists(f'{V2}/latent_norm.npz'):
    ln = np.load(f'{V2}/latent_norm.npz')
    print('latent_norm.npz mean', ln['mean'].tolist())
    print('latent_norm.npz std ', ln['std'].tolist())
cfg = json.load(open(f'{BASE}/TRELLIS/configs/flow/ss_flow_mc_v3.json'))
nm = cfg['dataset']['args']['normalize']
print('config normalize mean', nm['mean']); print('config normalize std ', nm['std'])
print('config == latent_norm?', bool(np.allclose(ln['mean'], nm['mean']) and np.allclose(ln['std'], nm['std'])) if os.path.exists(f'{V2}/latent_norm.npz') else 'n/a')
gm=[-0.172096,-0.253491,-0.464961,-0.182004,0.084891,0.07631,0.095924,0.17365]
gs=[0.845492,0.816747,1.224406,0.872764,0.942653,0.81222,0.820962,0.76632]
print('gen_v3 NORM mean == config?', bool(np.allclose(gm, nm['mean'])))
print('gen_v3 NORM std  == config?', bool(np.allclose(gs, nm['std'])))
print('gen_v3 NORM mean == latent_norm?', bool(np.allclose(gm, ln['mean'])) if os.path.exists(f'{V2}/latent_norm.npz') else 'n/a')

ss = np.load(f'{V2}/ss64.npy', mmap_mode='r')
st = np.load(f'{V2}/ss64_state.npy', mmap_mode='r')
NC=128
cls_tot=np.zeros(NC,dtype=np.int64); cls_st=np.zeros(NC,dtype=np.int64)
for i in range(0, n, 100):
    v=np.asarray(ss[i:i+100]).reshape(-1).astype(np.int64)
    s=np.asarray(st[i:i+100]).reshape(-1)
    nz=v>0
    cls_tot += np.bincount(v[nz], minlength=NC)[:NC]
    cls_st  += np.bincount(v[nz & (s!=0)], minlength=NC)[:NC]
stated = [int(c) for c in range(1,NC) if cls_st[c]>0]
print('classes with state!=0:', stated)
print('len:', len(stated))
mask_cls = list(range(1,72))+[127]
mask_tot = int(cls_tot[mask_cls].sum()); mask_st = int(cls_st[mask_cls].sum())
print('mask(1..71,127) total cells:', mask_tot, ' state!=0:', mask_st, f'({100.0*mask_st/mask_tot:.1f}%)')
print('64..71 total:', int(cls_tot[64:72].sum()), f'= {100.0*cls_tot[64:72].sum()/mask_tot:.1f}% of mask')
tot=int(cls_tot.sum()); print('ALL solid cells:', tot, 'state!=0:', int(cls_st.sum()), f'({100.0*cls_st.sum()/tot:.2f}%)')
print('state!=0 covered by stated classes:', int(cls_st[stated].sum()), f'({100.0*cls_st[stated].sum()/max(int(cls_st.sum()),1):.1f}% of all state!=0)')
bm = np.load(f'{V2}/block_map.npy')
print('block_map[62,63,64,71,72,126,127]=', bm[[62,63,64,71,72,126,127]].tolist())
print('cls63 total:', int(cls_tot[63]), ' cls127 total:', int(cls_tot[127]))
vocab=[l.strip() for l in open(f'{BASE}/dataset/BLOCK_VOCAB_FULL.txt', encoding='utf-8') if l.strip()]
print('VOCAB lines:', len(vocab))
for c in [62,63,64,71,72,126,127]:
    oi=int(bm[c])
    nm2 = vocab[oi] if 0<=oi<len(vocab) else '?'
    print(f'  cls {c}: old={oi} name={nm2} cells={int(cls_tot[c]):,} stated={int(cls_st[c]):,}')
