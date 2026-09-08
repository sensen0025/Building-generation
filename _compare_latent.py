import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
import numpy as np, torch
from trellis.models import SparseStructureDecoder, SparseStructureEncoder

VAE_CKPT = '/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/ckpts'
NORM = {'mean': [-0.172096, -0.253491, -0.464961, -0.182004, 0.084891, 0.07631, 0.095924, 0.17365],
        'std': [0.845492, 0.816747, 1.224406, 0.872764, 0.942653, 0.81222, 0.820962, 0.76632]}
dev = 'cuda'

decoder = SparseStructureDecoder(out_channels=128 + 35, latent_channels=8,
    num_res_blocks=2, num_res_blocks_middle=2, channels=[512, 128, 32], use_fp16=False).to(dev)
decoder.load_state_dict(torch.load(f'{VAE_CKPT}/decoder_ema0.9999_step0050000.pt', map_location='cpu', weights_only=True))
decoder.eval()

# 从预先缓存的真实 latent 库中读取 idx=4 的真实 latent
latent_file = '/home/linux/redstone_ai/dataset/v2/ss_latent_mc64_v2.npy'
if os.path.exists(latent_file):
    latents = np.load(latent_file, mmap_mode='r')
    gt_z = torch.from_numpy(np.array(latents[4])).unsqueeze(0).to(dev).float()
    print(f'GT Latent stats: min={gt_z.min().item():.3f}, max={gt_z.max().item():.3f}, mean={gt_z.mean().item():.3f}, std={gt_z.std().item():.3f}')

    m = torch.tensor(NORM['mean'], device=dev).view(1, 8, 1, 1, 1)
    sd = torch.tensor(NORM['std'], device=dev).view(1, 8, 1, 1, 1)
    # 注意: ss_latent_mc64_v2 是没有归一化还是已经归一化过的？
    # 让我们直接输入 decoder 测试
    with torch.no_grad():
        logits_raw = decoder(gt_z)
        id_raw = logits_raw[0, :128]
        occ_raw = (id_raw.argmax(0) > 0).sum().item()
        print(f'GT latent (direct to decoder) -> blocks: {occ_raw}')

        logits_scaled = decoder(gt_z * sd + m)
        id_scaled = logits_scaled[0, :128]
        occ_scaled = (id_scaled.argmax(0) > 0).sum().item()
        print(f'GT latent (scaled by NORM) -> blocks: {occ_scaled}')

        logits_unscaled = decoder((gt_z - m) / sd)
        id_un = logits_unscaled[0, :128]
        occ_un = (id_un.argmax(0) > 0).sum().item()
        print(f'GT latent (unscaled by NORM) -> blocks: {occ_un}')
else:
    print('latent file not found')
