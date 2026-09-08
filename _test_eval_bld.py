import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
import numpy as np
import torch
from PIL import Image
from litemapy import Region, BlockState, Schematic
from trellis.models import SparseStructureFlowModel, SparseStructureDecoder
from trellis.pipelines.samplers import FlowEulerCfgSampler
from transformers import AutoModel

TRELLIS = '/home/linux/redstone_ai/TRELLIS'
FLOW_CKPT = f'{TRELLIS}/outputs/ss_flow_mc_img_v3/ckpts/denoiser_ema0.9999_step0004000.pt'
VAE_CKPT = f'{TRELLIS}/outputs/ss_vae_mc_v2/ckpts/decoder_ema0.9999_step0050000.pt'
IMG_DIR = '/home/linux/redstone_ai/render_output'
SS64 = '/home/linux/redstone_ai/dataset/v2/ss64.npy'
BLOCK_MAP = '/home/linux/redstone_ai/dataset/v2/block_map.npy'
VOCAB = '/home/linux/redstone_ai/dataset/BLOCK_VOCAB_FULL.txt'
OUT = '/home/linux/redstone_ai/gen_outputs'
NUM_CLASSES = 128
STATE_DIMS = [7, 3, 6, 16, 3]
NORM = {'mean': [-0.172096, -0.253491, -0.464961, -0.182004, 0.084891, 0.07631, 0.095924, 0.17365],
        'std': [0.845492, 0.816747, 1.224406, 0.872764, 0.942653, 0.81222, 0.820962, 0.76632]}

dev = 'cuda'
print('Loading models...', flush=True)
denoiser = SparseStructureFlowModel(resolution=16, in_channels=8, out_channels=8,
    model_channels=512, cond_channels=1024, num_blocks=8, num_heads=8, num_head_channels=64,
    mlp_ratio=4, patch_size=2, pe_mode='ape', use_fp16=False).to(dev)
denoiser.load_state_dict(torch.load(FLOW_CKPT, map_location='cpu', weights_only=True))
denoiser.eval()

decoder = SparseStructureDecoder(out_channels=NUM_CLASSES + sum(STATE_DIMS), latent_channels=8,
    num_res_blocks=2, num_res_blocks_middle=2, channels=[512, 128, 32], use_fp16=False).to(dev)
decoder.load_state_dict(torch.load(VAE_CKPT, map_location='cpu', weights_only=True))
decoder.eval()

dino = AutoModel.from_pretrained('facebook/dinov2-large', local_files_only=True).to(dev).eval()
im_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
im_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def gen_sample(idx, seed=0, cfg_strength=3.0, steps=50):
    torch.manual_seed(seed)
    img_path = os.path.join(IMG_DIR, f'bld_{idx:04d}.png')
    img = Image.open(img_path).convert('RGB')
    w, h = img.size; s = min(w, h); l = (w - s) // 2; t = (h - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((518, 518), Image.LANCZOS)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
    with torch.no_grad():
        feats = dino((x - im_mean) / im_std).last_hidden_state
    sampler = FlowEulerCfgSampler(1e-5)
    noise = torch.randn(1, 8, 16, 16, 16, device=dev)
    with torch.no_grad():
        res = sampler.sample(denoiser, noise=noise, cond=feats, neg_cond=torch.zeros_like(feats),
                             steps=steps, cfg_strength=cfg_strength, verbose=False)
    z = res.samples
    m = torch.tensor(NORM['mean'], device=dev).view(1, 8, 1, 1, 1)
    sd = torch.tensor(NORM['std'], device=dev).view(1, 8, 1, 1, 1)
    z = z * sd + m
    with torch.no_grad():
        logits = decoder(z)
        id_logits = logits[:, :NUM_CLASSES]
        vox = id_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int16)
    return vox

idx = 4
ss = np.load(SS64, mmap_mode='r')
gt = np.asarray(ss[idx]).astype(np.int16)
gt_blocks = int((gt > 0).sum())
print(f'=== Testing Building Sample idx={idx} ===', flush=True)
print(f'GT non-zero blocks: {gt_blocks}', flush=True)

best_res = None
for cfg in [1.5, 3.0, 5.0]:
    for seed in [0, 1]:
        vox = gen_sample(idx, seed=seed, cfg_strength=cfg)
        gen_blocks = int((vox > 0).sum())
        inter = int(((gt > 0) & (vox > 0)).sum())
        union = int(((gt > 0) | (vox > 0)).sum())
        iou = inter / max(union, 1)
        print(f'CFG={cfg} Seed={seed} -> Gen Blocks={gen_blocks}, Intersect={inter}, IoU={iou:.4f}', flush=True)
        if best_res is None or iou > best_res[0]:
            best_res = (iou, vox)

np.save('/home/linux/redstone_ai/gen_outputs/gen_idx4.npy', best_res[1])
np.save('/home/linux/redstone_ai/gen_outputs/gt_idx4.npy', gt)
print('ALL DONE', flush=True)
