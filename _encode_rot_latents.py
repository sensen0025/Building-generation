# -*- coding: utf-8 -*-
"""旋转增广: 每个建筑 4 朝向(0/90/180/270 绕y) -> voxel 旋转 + 重编码 latent + state 同步.
输出:
  ss64_rot.npy        [4*N,64,64,64]  voxel (0°复用原, 90/180/270旋转)
  ss64_state_rot.npy  [4*N,64,64,64]  state (同步旋转)
  ss_latent_mc64_v2_rot.npy [4*N,8,16,16,16] latent (0°用原, 其余重编码)
  重算 mean/std -> latent_norm_rot.npz
注意: VAE 非旋转等变, 必须旋转 voxel 后再 encoder, 不能直接旋转 latent.
"""
import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
from trellis.models import SparseStructureEncoder

BASE = '/home/linux/redstone_ai/dataset/v2'
CKPT_DIR = '/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/ckpts'
NUM_CLASSES = 128
N_ROT = 4
BATCH = 4

def latest(prefix):
    fs = [f for f in glob.glob(os.path.join(CKPT_DIR, f'{prefix}*.pt')) if 'ema' not in f]
    return max(fs, key=lambda f: int(os.path.basename(f).split('step')[-1].split('.')[0]))

def rot_y_64(a):
    """voxel [64,64,64] 绕 y(维1) 转90: (x,y,z)->(63-z,y,x); 空间x=维0, y=维1, z=维2."""
    a = np.asarray(a)
    return np.transpose(a[:, :, ::-1], (2, 1, 0))

def rot_state_64(a):
    """state 逐体素, 同 voxel 旋转."""
    return rot_y_64(a)

def main():
    dev = 'cuda'
    ckpt = latest('encoder_step')
    print('encoder:', ckpt, flush=True)
    enc = SparseStructureEncoder(in_channels=NUM_CLASSES, latent_channels=8, num_res_blocks=2,
                                 num_res_blocks_middle=2, channels=[32, 128, 512], use_fp16=False).to(dev)
    enc.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    enc.eval()

    ss = np.load(f'{BASE}/ss64.npy', mmap_mode='r')
    st = np.load(f'{BASE}/ss64_state.npy', mmap_mode='r')
    N = ss.shape[0]
    print(f'N={N}, building {N_ROT}x dataset...', flush=True)

    # 先加载原 latent(0°复用)
    old_lat = np.load(f'{BASE}/ss_latent_mc64_v2.npy', mmap_mode='r')

    out_v = np.zeros((N * N_ROT, 64, 64, 64), dtype=np.uint8)
    out_s = np.zeros((N * N_ROT, 64, 64, 64), dtype=np.uint8)
    out_l = np.zeros((N * N_ROT, 8, 16, 16, 16), dtype=np.float32)

    for k in range(N_ROT):
        ang = k * 90
        print(f'--- rot {ang}deg ---', flush=True)
        if k == 0:
            # 0°: 直接拷贝原数据
            out_v[0::N_ROT] = np.asarray(ss)
            out_s[0::N_ROT] = np.asarray(st)
            out_l[0::N_ROT] = np.asarray(old_lat)
            print(f'  0deg copied from originals', flush=True)
            continue
        # 旋转体素 + state
        for i in range(N):
            v = np.asarray(ss[i])
            sv = np.asarray(st[i])
            # 应用 k*90 旋转(重复旋转)
            for _ in range(k):
                v = rot_y_64(v)
                sv = rot_state_64(sv)
            out_v[i * N_ROT + k] = v
            out_s[i * N_ROT + k] = sv
        # 批量编码这部分 latent
        idxs = [i * N_ROT + k for i in range(N)]
        for b in range(0, N, BATCH):
            j = min(b + BATCH, N)
            labels = out_v[idxs[b:j]].astype(np.int64)
            t = torch.from_numpy(labels).to(dev)
            x = F.one_hot(t, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            with torch.no_grad():
                z = enc(x, sample_posterior=False)
            out_l[idxs[b:j]] = z.cpu().numpy()
            if (b // BATCH) % 50 == 0:
                print(f'  enc {b}/{N} for rot{ang}', flush=True)
        print(f'  rot{ang} encoded', flush=True)

    np.save(f'{BASE}/ss64_rot.npy', out_v)
    np.save(f'{BASE}/ss64_state_rot.npy', out_s)
    np.save(f'{BASE}/ss_latent_mc64_v2_rot.npy', out_l)
    print('saved ss64_rot/state_rot/latent_rot shape', out_l.shape, flush=True)

    # normalize 按 (N*4, C, ...) channel 轴聚合: axis=(0,2,3,4)
    mean = out_l.mean(axis=(0, 2, 3, 4))
    std = out_l.std(axis=(0, 2, 3, 4))
    print('NORM mean:', mean.tolist(), flush=True)
    print('NORM std :', std.tolist(), flush=True)
    np.savez(f'{BASE}/latent_norm_rot.npz', mean=mean, std=std)
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
