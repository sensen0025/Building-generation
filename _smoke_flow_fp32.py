"""flow v3 fp32 smoke: 加载 denoiser(fp32) + 一步 flow-matching 训练式 forward/backward.
在 VAE 训练占用下跑 batch2, 报告显存峰值与 loss 有限性, 验证 fp32 配置可训练."""
import os, sys, json
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
import numpy as np
import torch

dev = 'cuda'
print('cuda:', torch.cuda.get_device_name(0), flush=True)
print('free before:', round(torch.cuda.mem_get_info()[0] / 1e9, 2), 'GB', flush=True)

cfg = json.load(open('/home/linux/redstone_ai/TRELLIS/configs/flow/ss_flow_mc_v3.json'))
margs = cfg['models']['denoiser']['args']
print('model args:', {k: margs[k] for k in ('resolution', 'in_channels', 'out_channels', 'model_channels', 'cond_channels', 'use_fp16')}, flush=True)
print('fp16_mode:', cfg['trainer']['args']['fp16_mode'], 'batch:', cfg['trainer']['args']['batch_size_per_gpu'], flush=True)

from trellis.models import SparseStructureFlowModel
denoiser = SparseStructureFlowModel(**margs).to(dev)
print('denoiser params:', sum(p.numel() for p in denoiser.parameters()) / 1e6, 'M', flush=True)
denoiser.train()
opt = torch.optim.AdamW(denoiser.parameters(), lr=1e-4)

B = 2
latent = torch.randn(B, margs['in_channels'], 16, 16, 16, device=dev)
# cond: dinov2-large last_hidden_state for 518px -> 37*37+1=1370 tokens x 1024
cond = torch.randn(B, 1370, margs['cond_channels'], device=dev)
noise = torch.randn_like(latent)
t = torch.rand(B, device=dev)

# flow matching 目标: z_t = (1-t)*noise + t*latent? 简化: 预测 target latent
zt = (1 - t.view(B, 1, 1, 1, 1)) * noise + t.view(B, 1, 1, 1, 1) * latent
pred = denoiser(zt, t * 1000, cond=cond)
print('pred:', tuple(pred.shape), flush=True)
target = latent - noise
loss = ((pred - target) ** 2).mean()
loss.backward()
opt.step()
print('loss:', float(loss.item()), flush=True)
print('grad finite:', all(torch.isfinite(p.grad).all().item() for p in denoiser.parameters() if p.grad is not None), flush=True)
print('free after:', round(torch.cuda.mem_get_info()[0] / 1e9, 2), 'GB', flush=True)
print('max allocated:', round(torch.cuda.max_memory_allocated() / 1e9, 2), 'GB', flush=True)
print('SMOKE OK', flush=True)
