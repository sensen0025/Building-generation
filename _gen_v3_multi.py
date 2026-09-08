"""v3 多 idx 生成测试: 用最新 flow EMA ckpt, 对几个代表性建筑生成, 报 IoU/occ/bbox."""
import os, sys, glob
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
import numpy as np
import torch
from PIL import Image

from trellis.models import SparseStructureFlowModel, SparseStructureDecoder
from trellis.pipelines.samplers import FlowEulerCfgSampler
from transformers import AutoModel

TRELLIS = '/home/linux/redstone_ai/TRELLIS'
FLOW_CKPT = f'{TRELLIS}/outputs/ss_flow_mc_img_v3/ckpts'
VAE_CKPT = f'{TRELLIS}/outputs/ss_vae_mc_v2/ckpts'
IMG_DIR = '/home/linux/redstone_ai/render_output'
SS64 = '/home/linux/redstone_ai/dataset/v2/ss64.npy'
NUM_CLASSES = 128

def load_norm_from_config():
    import json as _json
    with open(os.path.join(TRELLIS, 'configs/flow/ss_flow_mc_v3.json'), encoding='utf-8') as _f:
        cfg = _json.load(_f)
    nm = cfg['dataset']['args']['normalize']
    return nm['mean'], nm['std']

MEAN, STD = load_norm_from_config()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def latest_ema(prefix, d):
    files = glob.glob(os.path.join(d, f'{prefix}*ema*.pt'))
    if not files:
        files = glob.glob(os.path.join(d, f'{prefix}*.pt'))
    return max(files, key=lambda f: int(os.path.basename(f).split('step')[-1].split('.')[0]))

flow_ckpt = latest_ema('denoiser', FLOW_CKPT)
dec_ckpt = latest_ema('decoder', VAE_CKPT)
print('flow ckpt:', os.path.basename(flow_ckpt), flush=True)
print('dec  ckpt:', os.path.basename(dec_ckpt), flush=True)

denoiser = SparseStructureFlowModel(resolution=16, in_channels=8, out_channels=8,
    model_channels=512, cond_channels=1024, num_blocks=8, num_heads=8, num_head_channels=64,
    mlp_ratio=4, patch_size=2, pe_mode='ape', use_fp16=False).to(dev)
denoiser.load_state_dict(torch.load(flow_ckpt, map_location='cpu', weights_only=True))
denoiser.eval()
decoder = SparseStructureDecoder(out_channels=NUM_CLASSES + 7 + 3 + 6 + 16 + 3, latent_channels=8,
    num_res_blocks=2, num_res_blocks_middle=2, channels=[512, 128, 32], use_fp16=False).to(dev)
decoder.load_state_dict(torch.load(dec_ckpt, map_location='cpu', weights_only=True))
decoder.eval()
dino = AutoModel.from_pretrained('facebook/dinov2-large', local_files_only=True).to(dev).eval()
im_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
im_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def gen_from_image(idx, cfg=1.5, steps=50, seed=0):
    torch.manual_seed(seed)
    img = Image.open(os.path.join(IMG_DIR, f'bld_{idx:04d}.png')).convert('RGB')
    w, h = img.size; s = min(w, h); l = (w - s) // 2; t = (h - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((518, 518), Image.LANCZOS)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
    with torch.no_grad():
        feats = dino((x - im_mean) / im_std).last_hidden_state
    sampler = FlowEulerCfgSampler(1e-5)
    noise = torch.randn(1, 8, 16, 16, 16, device=dev)
    with torch.no_grad():
        res = sampler.sample(denoiser, noise=noise, cond=feats, neg_cond=torch.zeros_like(feats),
                             steps=steps, cfg_strength=cfg, verbose=False)
    z = res.samples
    m = torch.tensor(MEAN, device=dev).view(1, 8, 1, 1, 1)
    sd = torch.tensor(STD, device=dev).view(1, 8, 1, 1, 1)
    z = z * sd + m
    with torch.no_grad():
        logits = decoder(z)
        id_logits = logits[:, :NUM_CLASSES]
        vox = id_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int16)
    return vox

def bbox(v):
    nz = np.argwhere(v > 0)
    return (nz.max(0) - nz.min(0) + 1).tolist() if len(nz) else None

ss = np.load(SS64, mmap_mode='r')

# 测试 idx: 不同规模 (GT occ 需 > 0.5% 才有意义)
IDXS = [250, 6, 17, 46]
for idx in IDXS:
    gt = np.asarray(ss[idx]).astype(np.int16)
    gocc = (gt > 0).mean() * 100
    print(f'\n=== idx {idx}: GT occ={gocc:.2f}% bbox={bbox(gt)} ===', flush=True)
    best = None
    for seed in range(3):
        vox = gen_from_image(idx, seed=seed)
        inter = ((gt > 0) & (vox > 0)).sum(); union = ((gt > 0) | (vox > 0)).sum()
        iou = inter / max(union, 1)
        vocc = (vox > 0).mean() * 100
        print(f'  seed {seed}: IoU={iou:.3f} occ={vocc:.2f}% bbox={bbox(vox)}', flush=True)
        if best is None or iou > best[0]:
            best = (iou, vox)
    print(f'  BEST IoU={best[0]:.3f}', flush=True)
    # 保存最佳生成
    np.save(f'/home/linux/redstone_ai/gen_outputs/gen_idx{idx}.npy', best[1])
print('DONE', flush=True)
