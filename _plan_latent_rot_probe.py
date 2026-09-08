# -*- coding: utf-8 -*-
"""plan-mode 只读探索: 验证 latent 绕 y 轴旋转 是否 ≈ encoder(旋转后 voxel).
若近似成立 -> 可以直接旋转 latent(省重新编码); 否则需旋转 voxel 再编码.
不写文件, 只打印对比误差."""
import sys, os, glob
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
from trellis.models import SparseStructureEncoder

CKPT_DIR = '/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/ckpts'
DATA = '/home/linux/redstone_ai/dataset/v2/ss64.npy'
NUM_CLASSES = 128

def latest(prefix):
    fs = [f for f in glob.glob(os.path.join(CKPT_DIR, f'{prefix}*.pt')) if 'ema' not in f]
    return max(fs, key=lambda f: int(os.path.basename(f).split('step')[-1].split('.')[0]))

dev = 'cuda'
enc = SparseStructureEncoder(in_channels=NUM_CLASSES, latent_channels=8, num_res_blocks=2,
                             num_res_blocks_middle=2, channels=[32, 128, 512], use_fp16=False).to(dev)
enc.load_state_dict(torch.load(latest('encoder_step'), map_location='cpu', weights_only=True))
enc.eval()

def rot_y_64(a):
    """voxel 64^3 绕 y 转90: (x,y,z)->(63-z,y,x)."""
    return np.transpose(a[:, :, ::-1].copy(), (2, 1, 0))

def rot_y_16(a):
    """latent [C,16,16,16] 绕空间 y(维2) 转90: (x,y,z)->(15-z,y,x); 保留 channel(C,维0)."""
    arr = a.copy()
    # 沿空间z(维3)反转, 再交换 空间x(维1) 和 空间z(维3)
    arr = arr[:, :, :, ::-1]
    return np.transpose(arr, (0, 1, 3, 2))

ss = np.load(DATA, mmap_mode='r')
for idx in [250, 17, 46]:
    v = np.asarray(ss[idx]).astype(np.int64)
    vr = rot_y_64(v)
    # 重新编码旋转后的 voxel
    with torch.no_grad():
        t = torch.from_numpy(vr[None]).to(dev)
        x = F.one_hot(t, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        z_re = enc(x, sample_posterior=False).cpu().numpy()[0]  # [8,16,16,16]
    # 原 voxel 编码后直接旋转 latent
    with torch.no_grad():
        t0 = torch.from_numpy(v[None]).to(dev)
        x0 = F.one_hot(t0, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        z0 = enc(x0, sample_posterior=False).cpu().numpy()[0]
    z_rot = rot_y_16(z0)  # 直接旋转 latent
    # 对比: z_re(重编码) vs z_rot(直接转)
    diff = np.abs(z_re - z_rot)
    rel = diff.mean() / (np.abs(z_re).mean() + 1e-6)
    print(f'idx{idx}: z_re mean|.|={np.abs(z_re).mean():.4f}  '
          f'直接转latent vs 重编码 meanAbsDiff={diff.mean():.5f} rel={rel*100:.2f}%', flush=True)
print('DONE', flush=True)
