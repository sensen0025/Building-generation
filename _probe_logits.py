import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
import numpy as np, torch
from PIL import Image
from trellis.models import SparseStructureFlowModel, SparseStructureDecoder
from trellis.pipelines.samplers import FlowEulerCfgSampler
from transformers import AutoModel

TRELLIS = '/home/linux/redstone_ai/TRELLIS'
FLOW_CKPT = f'{TRELLIS}/outputs/ss_flow_mc_img_v3/ckpts/denoiser_ema0.9999_step0004000.pt'
VAE_CKPT = f'{TRELLIS}/outputs/ss_vae_mc_v2/ckpts/decoder_ema0.9999_step0050000.pt'
NORM = {'mean': [-0.172096, -0.253491, -0.464961, -0.182004, 0.084891, 0.07631, 0.095924, 0.17365],
        'std': [0.845492, 0.816747, 1.224406, 0.872764, 0.942653, 0.81222, 0.820962, 0.76632]}
dev = 'cuda'

denoiser = SparseStructureFlowModel(resolution=16, in_channels=8, out_channels=8,
    model_channels=512, cond_channels=1024, num_blocks=8, num_heads=8, num_head_channels=64,
    mlp_ratio=4, patch_size=2, pe_mode='ape', use_fp16=False).to(dev)
denoiser.load_state_dict(torch.load(FLOW_CKPT, map_location='cpu', weights_only=True))
denoiser.eval()

decoder = SparseStructureDecoder(out_channels=128 + 35, latent_channels=8,
    num_res_blocks=2, num_res_blocks_middle=2, channels=[512, 128, 32], use_fp16=False).to(dev)
decoder.load_state_dict(torch.load(VAE_CKPT, map_location='cpu', weights_only=True))
decoder.eval()

dino = AutoModel.from_pretrained('facebook/dinov2-large', local_files_only=True).to(dev).eval()
im_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
im_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

# 输入图片 bld_0004.png
img_path = '/home/linux/redstone_ai/render_output/bld_0004.png'
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
                         steps=50, cfg_strength=3.0, verbose=False)

z = res.samples
m = torch.tensor(NORM['mean'], device=dev).view(1, 8, 1, 1, 1)
sd = torch.tensor(NORM['std'], device=dev).view(1, 8, 1, 1, 1)
z_scaled = z * sd + m

print(f'Diffusion z stats: min={z.min().item():.3f}, max={z.max().item():.3f}, mean={z.mean().item():.3f}, std={z.std().item():.3f}')

with torch.no_grad():
    logits = decoder(z_scaled) # [1, 163, 64, 64, 64]
    id_logits = logits[0, :128] # [128, 64, 64, 64]

air_logits = id_logits[0]
solid_logits, solid_cls = id_logits[1:].max(dim=0)
solid_cls = solid_cls + 1 # 偏移到真实类别

diff = solid_logits - air_logits
print(f'Air logits: min={air_logits.min().item():.2f}, max={air_logits.max().item():.2f}, mean={air_logits.mean().item():.2f}')
print(f'Solid max logits: min={solid_logits.min().item():.2f}, max={solid_logits.max().item():.2f}, mean={solid_logits.mean().item():.2f}')
print(f'diff (Solid - Air): max={diff.max().item():.2f}, min={diff.min().item():.2f}, mean={diff.mean().item():.2f}')

# 测试不同阈值下的方块数
for bias in [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]:
    occ = (diff > bias).sum().item()
    print(f'Threshold bias={bias:+.1f} -> non-zero blocks = {occ}')

# 保存一个带合适阈值的真实生成方块矩阵
# 如果取 diff 最大的 top-K 或者适当 bias
top_k_blocks = 1500
flat_diff = diff.flatten()
kth_val = torch.topk(flat_diff, top_k_blocks).values[-1].item()
print(f'Top {top_k_blocks} threshold = {kth_val:.3f}')

gen_occ = (diff >= kth_val)
gen_vox = torch.where(gen_occ, solid_cls, torch.zeros_like(solid_cls)).cpu().numpy().astype(np.int16)
np.save('/home/linux/redstone_ai/gen_outputs/gen_topk_idx4.npy', gen_vox)
print('PROBE DONE')
