# -*- coding: utf-8 -*-
"""Stage D: encode new-sample latents, 4x rotate aug, merge with old arrays, recompute norm.

Inputs:
  /data/redstone_v4/work/ss64_new_c.npy          (N,64,64,64) uint8
  /data/redstone_v4/work/ss64_state_new_c.npy    (N,64,64,64) uint16
  old: /home/linux/redstone_ai/dataset/v2/ss64_rot.npy / ss64_state_rot.npy / ss_latent_mc64_v2_rot.npy

Outputs (all in /data/redstone_v4/train/):
  ss64_all.npy           (21356 + N*4, 64,64,64) uint8
  ss64_state_all.npy     uint16
  ss_latent_all.npy      (M, 8,16,16,16) float32
  latent_norm_all.npz    mean/std
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
os.environ.setdefault('ATTN_BACKEND', 'sdpa')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from trellis.models import SparseStructureEncoder

NUM_CLASSES = 128
BASE = '/data/redstone_v4'
OLD = '/home/linux/redstone_ai/dataset/v2'
CKPT = '/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/ckpts/encoder_step0050000.pt'
OUTD = f'{BASE}/train'
os.makedirs(OUTD, exist_ok=True)
N_ROT = 4
BATCH = 6


def rot_y_64(a):
    a = np.asarray(a)
    return np.transpose(a[:, :, ::-1], (2, 1, 0))


def main():
    t0 = time.time()
    enc = SparseStructureEncoder(in_channels=NUM_CLASSES, latent_channels=8, num_res_blocks=2,
                                 num_res_blocks_middle=2, channels=[32, 128, 512], use_fp16=False).to('cuda')
    enc.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    enc.eval()
    print('encoder loaded', flush=True)

    ss_new = np.load(f'{BASE}/work/ss64_new_c.npy', mmap_mode='r')
    st_new = np.load(f'{BASE}/work/ss64_state_new_c.npy', mmap_mode='r')
    N = ss_new.shape[0]
    print('new samples:', N, flush=True)

    ss_old = np.load(f'{OLD}/ss64_rot.npy', mmap_mode='r')
    st_old = np.load(f'{OLD}/ss64_state_rot.npy', mmap_mode='r')
    lat_old = np.load(f'{OLD}/ss_latent_mc64_v2_rot.npy', mmap_mode='r')
    print('old rot:', ss_old.shape, lat_old.shape, flush=True)

    M = ss_old.shape[0] + N * N_ROT
    out_s = np.lib.format.open_memmap(f'{OUTD}/ss64_all.npy', mode='w+', dtype=np.uint8, shape=(M, 64, 64, 64))
    out_t = np.lib.format.open_memmap(f'{OUTD}/ss64_state_all.npy', mode='w+', dtype=np.uint16, shape=(M, 64, 64, 64))
    out_l = np.lib.format.open_memmap(f'{OUTD}/ss_latent_all.npy', mode='w+', dtype=np.float32, shape=(M, 8, 16, 16, 16))

    # copy old
    out_s[:ss_old.shape[0]] = np.asarray(ss_old)
    out_t[:ss_old.shape[0]] = np.asarray(st_old)
    out_l[:ss_old.shape[0]] = np.asarray(lat_old)
    out_s.flush(); out_t.flush(); out_l.flush()
    print('old copied', time.time() - t0, flush=True)

    # new: idx mapping old N0.. ; sample j, rot k -> M index N0 + j*4 + k
    N0 = ss_old.shape[0]
    for k in range(N_ROT):
        print(f'--- rot {k*90}deg ---', flush=True)
        # build rotated voxel+state into memmap first
        for j in range(N):
            v = np.asarray(ss_new[j])
            s = np.asarray(st_new[j])
            for _ in range(k):
                v = rot_y_64(v)
                s = rot_y_64(s)
            out_s[N0 + j * N_ROT + k] = v
            out_t[N0 + j * N_ROT + k] = s
        out_s.flush(); out_t.flush()
        # encode batch
        for b in range(0, N, BATCH):
            e = min(b + BATCH, N)
            labels = np.stack([out_s[N0 + jj * N_ROT + k] for jj in range(b, e)]).astype(np.int64)
            t = torch.from_numpy(labels).to('cuda')
            x = F.one_hot(t, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            with torch.no_grad():
                z = enc(x, sample_posterior=False)
            for jj, zz in zip(range(b, e), z):
                out_l[N0 + jj * N_ROT + k] = zz.cpu().numpy()
            if (b // BATCH) % 100 == 0:
                out_l.flush()
                print(f'  enc {b}/{N} rot{k} ({time.time()-t0:.0f}s)', flush=True)
    print('encoding done', time.time() - t0, flush=True)

    # recompute norm over ALL latents (channel axis)
    tot = np.zeros(8, dtype=np.float64)
    tot2 = np.zeros(8, dtype=np.float64)
    cnt = 0
    for b in range(0, M, 512):
        e = min(b + 512, M)
        a = np.asarray(out_l[b:e]).astype(np.float64)
        tot += a.sum(axis=(0, 2, 3, 4))
        tot2 += (a * a).sum(axis=(0, 2, 3, 4))
        cnt += a.shape[0] * 16 * 16 * 16
    mean = tot / cnt
    std = np.sqrt(np.maximum(tot2 / cnt - mean * mean, 1e-12))
    print('NORM mean:', mean.tolist(), flush=True)
    print('NORM std:', std.tolist(), flush=True)
    np.savez(f'{OUTD}/latent_norm_all.npz', mean=mean, std=std)
    print('DONE', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
