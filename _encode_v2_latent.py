"""用 VAE v2 encoder 编码全部 GT 样本 -> latent npy + 统计 mean/std."""
import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
from trellis.models import SparseStructureEncoder

CKPT_DIR = '/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/ckpts'
DATA = '/home/linux/redstone_ai/dataset/v2/ss64.npy'
OUT = '/home/linux/redstone_ai/dataset/v2/ss_latent_mc64_v2.npy'
NUM_CLASSES = 128
BATCH = 8

def latest(prefix):
    files = [f for f in glob.glob(os.path.join(CKPT_DIR, f'{prefix}*.pt')) if 'ema' not in f]
    return max(files, key=lambda f: int(os.path.basename(f).split('step')[-1].split('.')[0]))

def main():
    dev = 'cuda'
    ckpt = latest('encoder_step')
    print('encoder:', ckpt, flush=True)
    enc = SparseStructureEncoder(in_channels=NUM_CLASSES, latent_channels=8, num_res_blocks=2,
                                 num_res_blocks_middle=2, channels=[32, 128, 512], use_fp16=False).to(dev)
    enc.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    enc.eval()

    ss = np.load(DATA, mmap_mode='r')
    N = ss.shape[0]
    print(f'encoding {N} samples...', flush=True)
    latents = np.zeros((N, 8, 16, 16, 16), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, BATCH):
            j = min(i + BATCH, N)
            label = np.asarray(ss[i:j]).astype(np.int64)
            t = torch.from_numpy(label).to(dev)
            x = F.one_hot(t, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            z = enc(x, sample_posterior=False)
            latents[i:j] = z.cpu().numpy()
            if (i // BATCH) % 25 == 0:
                print(f'  {j}/{N}', flush=True)
    np.save(OUT, latents)
    print('saved:', OUT, latents.shape, f'{os.path.getsize(OUT)/1e9:.2f}GB', flush=True)

    # mean/std per channel (用于 flow 归一化): 按 (N, C, 16,16,16) 的 channel 轴聚合,
    # 即 axis=(0,2,3,4)。注意: 不能 reshape(-1,8) —— 那会把 8 个 channel 混在一起按 z 相位平均。
    mean = latents.mean(axis=(0, 2, 3, 4))
    std = latents.std(axis=(0, 2, 3, 4))
    print('NORM mean:', mean.tolist(), flush=True)
    print('NORM std :', std.tolist(), flush=True)
    np.savez('/home/linux/redstone_ai/dataset/v2/latent_norm.npz', mean=mean, std=std)
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
