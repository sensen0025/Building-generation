from mcfix.legacy import legacy_state_classes
import os, sys, json
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/home/linux/redstone_ai/TRELLIS")
import numpy as np, torch
from trellis.models import SparseStructureFlowModel, SparseStructureDecoder
from trellis.utils.mc_inpaint import sample_completion

cfg = json.load(open("/data/redstone_v4/work/ss_flow_mc_v4_comb.json"))
args = cfg["models"]["denoiser"]["args"]
mean = torch.tensor(cfg["dataset"]["args"]["normalize"]["mean"]).view(1,-1,1,1,1)
std = torch.tensor(cfg["dataset"]["args"]["normalize"]["std"]).view(1,-1,1,1,1)

dev = "cuda"
denoiser = SparseStructureFlowModel(**args).to(dev)
denoiser.load_state_dict(torch.load("/data/redstone_v4/work/v4_init.pt", map_location="cpu", weights_only=True))
denoiser.eval()
decoder = SparseStructureDecoder(out_channels=128+35, latent_channels=8, num_res_blocks=2,
                                 num_res_blocks_middle=2, channels=[512,128,32], use_fp16=False).to(dev)
decoder.load_state_dict(torch.load("/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/ckpts/decoder_step0050000.pt", map_location="cpu", weights_only=True))
decoder.eval()

ss = np.load("/data/redstone_v4/train/ss64_all.npy", mmap_mode="r")
st = np.load("/data/redstone_v4/train/ss64_state_all.npy", mmap_mode="r")
gt = np.asarray(ss[30000]); gtp = np.asarray(st[30000])
nz = np.argwhere(gt > 0)
print("sample solid:", nz.shape[0])
# bottom completion mask: top half unknown
mask = np.zeros_like(gt, dtype=bool); cut = int(nz[:,1].min() + (nz[:,1].max()-nz[:,1].min())*0.5)
mask[:, cut:, :] = True
known = ~mask
# block-only: zero cond tokens
B = 1
zero_cond = torch.zeros(B, 1370, 1024, device=dev)
with torch.no_grad():
    vox, fields = sample_completion(
        denoiser, decoder, torch.randn(B, 8, 16, 16, 16, device=dev),
        zero_cond,
        torch.from_numpy(gt).long().unsqueeze(0).to(dev),
        torch.from_numpy(known).unsqueeze(0).to(dev),
        torch.from_numpy(gtp).long().unsqueeze(0).to(dev),
        mean.to(dev), std.to(dev), steps=20, state_classes=legacy_state_classes())
vox = vox[0].cpu().numpy()
inter = ((gt>0)&(vox>0)&mask).sum(); union = (((gt>0)&mask)|((vox>0)&mask)).sum()
print("completed-region IoU:", round(inter/max(union,1),4))
known_ok = ((gt>0)==(vox>0))[known].mean()
print("known preserved (block equality):", round(float(known_ok),4))
print("COMPLETION SMOKE OK")
